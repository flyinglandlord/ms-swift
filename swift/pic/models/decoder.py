"""Concrete decoder-only adapter. No architecture assumptions leak into Trainer."""
import importlib
import json
import random
import torch
from contextlib import contextmanager
from pathlib import Path
from transformers.cache_utils import DynamicCache

from ..contracts import Architecture, Memory


def base_model(model):
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


@contextmanager
def evaluating(model):
    training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(training)


def pairs(cache):
    return tuple((layer.keys, layer.values) for layer in cache.layers)


class DecoderArchitecture(Architecture):
    capabilities = frozenset({'c2kv', 'c2kv_context', 'frozen_artifact', 'query_lora', 'full_reference'})

    def __init__(self,
                 name,
                 prefix,
                 module,
                 template,
                 backend='native',
                 model_config=None,
                 attn_impl='sdpa',
                 gist_gradient_checkpointing=False):
        self.name, self.prefix, self.module, self.template = name, prefix, module, template
        self.backend = backend
        self.model_config = model_config or {}
        self.attn_impl = attn_impl
        self.gist_gradient_checkpointing = gist_gradient_checkpointing
        self.query_targets = ('q_proj', )
        self.base_targets = ('q_proj', 'k_proj', 'v_proj')
        self.gist_targets = ('gist_q_proj', 'gist_k_proj', 'gist_v_proj')

    def classes(self):
        module = importlib.import_module(f'swift.pic.models.c2kv.{self.module}' if self.backend ==
                                         'c2kv' else 'transformers')
        return getattr(module, self.prefix + 'Config'), getattr(module, self.prefix + 'ForCausalLM')

    def prepare_c2kv(self):
        self.backend = 'c2kv'

    def configure_c2kv(self, model, options):
        from ..tuning import freeze, lora
        if options.tuning == 'lora':
            targets = {
                'gist': self.gist_targets,
                'base': self.base_targets,
                'both': self.gist_targets + self.base_targets
            }
            return lora(model, targets[options.lora_scope], options.rank, options.alpha)
        if not getattr(options, 'algorithm_config', {}).get('only_train_gist', True):
            model.requires_grad_(True)
            return model
        freeze(model)
        for name, parameter in model.named_parameters():
            if 'gist_' in name:
                parameter.requires_grad_(True)
        return model

    def configure_query_adapter(self, model, rank, alpha):
        from ..tuning import lora
        return lora(model, self.query_targets, rank, alpha)

    @contextmanager
    def offline_context(self, model):
        with model.disable_adapter(), evaluating(model), torch.no_grad():
            yield

    def snapshot_memory(self, memory):
        payload = tuple(tuple(t.detach().cpu().clone() for t in pair) for pair in memory.payload)
        size = sum(t.numel() * t.element_size() for pair in payload for t in pair)
        return Memory(payload, memory.token_count, memory.metadata), size

    def materialize_memory(self, memory, device):
        payload = tuple(tuple(t.to(device).clone() for t in pair) for pair in memory.payload)
        return Memory(payload, memory.token_count, memory.metadata)

    def generate_teacher(self, model, processor, record, max_new_tokens):
        prompt = record.get('system_ids', []) + sum(record['segments'], []) + record['query_ids']
        ids = torch.tensor([prompt], device=model.device)
        with torch.no_grad():
            result = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.eos_token_id)
        return result[0, len(prompt):].tolist()

    def load(self, path, dtype):
        from swift.model import ModelLoader, ModelMeta, get_model_processor, register_model
        spec = self

        class Loader(ModelLoader):

            def get_config(self, model_dir):
                config = spec.classes()[0].from_pretrained(model_dir, local_files_only=True, **spec.model_config)
                layer_types = getattr(config, 'layer_types', None) or []
                if any(kind != 'full_attention' for kind in layer_types):
                    raise ValueError('Sliding/hybrid attention needs a dedicated PIC architecture adapter')
                if spec.backend == 'c2kv':
                    saved = json.loads((Path(model_dir) / 'config.json').read_text())
                    defaults = dict(
                        gist_type='interleave-1',
                        gist_param='qkv',
                        gist_residual_type='embed-mean',
                        gist_overlap=64,
                        pad_token_id=None)
                    for key, default in defaults.items():
                        setattr(config, key, spec.model_config.get(key, saved.get(key, default)))
                    # Swift assigns tokenizer.pad_token_id to config after loading.
                    # It is a vocabulary ID, invalid for the tiny gist embedding
                    # table on reload. C2KV masks padding explicitly instead.
                    if spec.model_config.get('pad_token_id') is not None:
                        raise ValueError('C2KV requires pad_token_id=None; padding is handled by attention masks')
                    config.pad_token_id = None
                    kernel = importlib.import_module(f'swift.pic.models.c2kv.{spec.module}.modeling_{spec.name}')
                    kernel.GIST_GRADIENT_CHECKPOINTING = spec.gist_gradient_checkpointing
                config._attn_implementation = spec.attn_impl
                return config

            def get_model(self, model_dir, config, processor, model_kwargs):
                if spec.backend == 'c2kv':
                    config.gist_token_id = processor.eos_token_id
                return spec.classes()[1].from_pretrained(
                    model_dir, config=config, local_files_only=True, **model_kwargs)

        key = f'pic_{self.backend}_{self.name}'
        register_model(ModelMeta(key, [], loader=Loader, template=self.template), exist_ok=True)
        return get_model_processor(
            path, model_type=key, torch_dtype=dtype, attn_impl=self.attn_impl, use_hf=True, device_map=None)

    def context_forward(self, model, inputs):
        """Original combined C2KV forward, including compression/reconstruction loss."""
        return model(**inputs)

    def encode(self, model, tokens, prefix=None):
        """Native post-RoPE artifact KV at local positions, without request-specific context."""
        if self.backend == 'c2kv':
            return self.encode_many(model, tokens)[0]
        prefix = tokens[:, :0] if prefix is None else prefix
        ids = torch.cat([prefix, tokens], dim=1)
        output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        n = prefix.shape[1]
        kv = tuple((k[:, :, n:].detach(), v[:, :, n:].detach()) for k, v in pairs(output.past_key_values))
        return Memory(kv, tokens.shape[1], {'position_policy': 'local', 'cache_prefix_length': n})

    def encode_many(self, model, padded_tokens):
        if self.backend != 'c2kv':
            return super().encode_many(model, padded_tokens)
        decoder = base_model(model).model
        valid = padded_tokens != -100
        tokens = padded_tokens.masked_fill(~valid, decoder.gist_token_id)
        kwargs = {'ratio': random.choice([2, 4, 8])} if decoder.config.gist_type == 'dynamic-interleave' else {}
        output, mask, positions = decoder.generate_gist(tokens, valid, **kwargs)
        if decoder.config.gist_type == 'interleave-1' and not torch.equal(mask.sum(dim=1), valid.sum(dim=1)):
            raise ValueError('C2KV 1:1 extractor changed the valid token count')
        return [
            Memory(
                tuple((k[i:i + 1], v[i:i + 1]) for k, v in output.past_key_values), int(mask[i].sum()), {
                    'mask': mask[i:i + 1],
                    'positions': positions[i:i + 1],
                    'source_tokens': int(valid[i].sum()),
                    'position_advance': tokens.shape[1] if mask.all() else int(positions[i, int(mask[i].sum()) - 1]) + 1
                }) for i in range(tokens.shape[0])
        ]

    def assemble(self, model, memories, batch):
        if self.backend == 'c2kv':
            return self._assemble_c2kv(model, memories, batch)
        prefix = batch['system_input_ids']
        prefix_kv = None
        if prefix.numel():
            # Prefix is online: its adapted hidden states may carry gradients.
            output = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
            prefix_kv = pairs(output.past_key_values)
        joined = []
        for layer in range(len(memories[0].payload)):
            entries = ([] if prefix_kv is None else [prefix_kv[layer]]) + [m.payload[layer] for m in memories]
            joined.append(tuple(torch.cat([entry[i] for entry in entries], dim=2) for i in range(2)))
        # New cache owns concatenated tensors: reader mutation cannot modify stored artifacts.
        return Memory(
            DynamicCache(tuple(joined), config=base_model(model).config),
            prefix.shape[1] + sum(m.token_count for m in memories))

    def read(self, model, batch, memory):
        if self.backend == 'c2kv':
            return model(
                input_ids=batch['input_ids'],
                past_key_values=memory.payload,
                use_gist=True,
                use_cache=False,
                **memory.metadata)
        ids = batch['input_ids']
        position_ids = torch.arange(memory.token_count, memory.token_count + ids.shape[1], device=ids.device)[None]
        mask = torch.ones((1, memory.payload.get_seq_length() + ids.shape[1]), device=ids.device, dtype=torch.bool)
        return model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=memory.payload,
            use_cache=False)

    def full_reference(self, model, batch):
        segments = [x[x != -100][None] for x in batch['context_input_ids'][0]]
        ids = torch.cat([batch['system_input_ids'], *segments, batch['input_ids']], dim=1)
        return model(input_ids=ids, use_cache=False, logits_to_keep=batch['input_ids'].shape[1])

    def _assemble_c2kv(self, model, memories, batch):
        from .c2kv.gist_utils import apply_rotary_pos_emb
        decoder = base_model(model).model
        prefix = batch['system_input_ids']
        cache = None
        if prefix.numel():
            with evaluating(model), torch.no_grad():
                cache = model(input_ids=prefix, use_cache=True, logits_to_keep=1).past_key_values
        if cache is None:
            cache = DynamicCache(config=decoder.config)
        offset = prefix.shape[1]
        masks = [torch.ones_like(prefix, dtype=torch.bool)]
        for memory in memories:
            positions = memory.metadata['positions'] + offset
            cos, sin = decoder.rotary_emb(memory.payload[0][0], positions)
            for layer, (key, value) in enumerate(memory.payload):
                cache.update(apply_rotary_pos_emb(key, cos, sin), value, layer)
            masks.append(memory.metadata['mask'])
            offset += memory.metadata['position_advance']
        ids = batch['input_ids']
        suffix_start = prefix.shape[1] + sum(m.metadata['source_tokens'] for m in memories)
        positions = torch.arange(suffix_start, suffix_start + ids.shape[1], device=ids.device)[None]
        mask = torch.cat([*masks, batch['attention_mask']], dim=1)
        return Memory(cache, prefix.shape[1] + sum(m.token_count for m in memories), {
            'attention_mask': mask,
            'position_ids': positions
        })
