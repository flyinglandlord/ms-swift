"""Native MCore projections/MLP and TP collectives with explicit PIC attention.

The reference attention path deliberately excludes fused attention, activation
recomputation and pipeline parallelism until their cache lifetimes are validated.
"""
import os
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from types import MethodType

from .kernels import attention, gist_mask, hf_rms_norm, rotary


def mark_column_parallel_weight(weight):
    """Retain shard semantics even when random parameter initialization is disabled."""
    from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
    if not hasattr(weight, 'tensor_model_parallel'):
        set_tensor_model_parallel_attributes(weight, True, 0, 1)
    if not weight.tensor_model_parallel or weight.partition_dim != 0:
        raise ValueError('PIC column projection requires an output-dimension TP shard')


def install(model, args):
    from megatron.core.tensor_parallel import ColumnParallelLinear

    config = model.config
    if getattr(args, 'pic_input_norm', 'hf') not in ('hf', 'native'):
        raise ValueError('pic_input_norm must be hf or native')
    if getattr(args, 'pic_numerical_reference', False):
        if config.params_dtype != torch.float32 or os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
            raise ValueError('PIC numerical reference requires FP32 and NVIDIA_TF32_OVERRIDE=0 before launch')
    if config.num_query_groups % args.tensor_model_parallel_size:
        raise ValueError('PIC requires KV heads divisible by TP size')
    if config.hidden_dropout or config.attention_dropout:
        raise ValueError('PIC reference backend requires zero dropout')
    if config.num_moe_experts or getattr(config, 'multi_latent_attention', False):
        raise ValueError('PIC currently requires a dense standard GQA decoder')
    hf = config.hf_config
    if hasattr(hf, 'gist_type'):
        raise ValueError(
            'Importing HF C2KV gist weights is not implemented; use a vanilla backbone or native PIC checkpoint')
    if hf.model_type not in ('qwen3', 'qwen2', 'llama'):
        raise ValueError('PIC Megatron currently supports dense Qwen3/Qwen2/Llama architectures only')
    if getattr(hf, 'use_sliding_window', False) or getattr(config, 'window_size', None):
        raise ValueError('Sliding attention needs a dedicated PIC adapter')
    if getattr(config, 'layernorm_zero_centered_gamma', False) or getattr(config, 'rotary_interleaved', False):
        raise ValueError('PIC currently requires standard RMSNorm and half-split RoPE')
    rope = getattr(hf, 'rope_scaling', None) or getattr(hf, 'rope_parameters', None) or {}
    if rope.get('rope_type', 'default') != 'default':
        raise ValueError('Scaled RoPE needs a dedicated PIC architecture adapter')
    theta = getattr(hf, 'rope_theta', rope.get('rope_theta', 10000.))
    if getattr(config, 'rotary_percent', 1.) != 1.:
        raise ValueError('Partial RoPE is not supported')
    model.requires_grad_(False)
    runtime = Runtime(model, args, theta)
    for index, layer in enumerate(model.decoder.layers):
        attn = layer.self_attention
        base = attn.linear_qkv
        if hasattr(base, 'layer_norm_weight') and config.normalization != 'RMSNorm':
            raise ValueError('Fused PIC projections currently require RMSNorm')

        def linear(input_size, output_size, bias=False):
            projection = ColumnParallelLinear(
                input_size,
                output_size,
                config=config,
                init_method=config.init_method,
                bias=bias,
                gather_output=False,
                skip_bias_add=False,
                tp_group=base.tp_group)
            # Imported backbones disable random initialization. MCore only tags
            # ColumnParallelLinear.weight inside its initializer; without this
            # tag wrap_model treats shards as replicas and excludes TP rank > 0
            # from the native optimizer's global gradient norm.
            mark_column_parallel_weight(projection.weight)
            return projection

        if args.pic_algorithm == 'c2kv':
            attn.pic_gist_qkv = linear(config.hidden_size,
                                       (config.num_attention_heads + 2 * config.num_query_groups) * config.kv_channels,
                                       base.bias is not None and base.bias.numel() > 0)
            with torch.no_grad():
                attn.pic_gist_qkv.weight.copy_(base.weight)
                if base.bias is not None and base.bias.numel() > 0:
                    attn.pic_gist_qkv.bias.copy_(base.bias)
            if index == 0:
                model.pic_gist_embedding = torch.nn.Parameter(
                    # Match HF Embedding [1, hidden]; MCore exempts 1-D tensors
                    # from weight decay, which would silently change AdamW.
                    torch.zeros(1, config.hidden_size, device=base.weight.device, dtype=base.weight.dtype))
                with torch.random.fork_rng(devices=[base.weight.device]):
                    torch.manual_seed(args.seed + 10000)
                    torch.nn.init.normal_(model.pic_gist_embedding, std=config.init_method_std)
                torch.distributed.broadcast(
                    model.pic_gist_embedding.data,
                    src=torch.distributed.get_process_group_ranks(base.tp_group)[0],
                    group=base.tp_group)
        else:
            attn.pic_query_a = torch.nn.Linear(
                config.hidden_size, args.pic_rank, bias=False, device=base.weight.device, dtype=base.weight.dtype)
            with torch.random.fork_rng(devices=[base.weight.device]):
                torch.manual_seed(args.seed + 20000 + index)
                torch.nn.init.uniform_(attn.pic_query_a.weight, -config.hidden_size**-.5, config.hidden_size**-.5)
            # Replicated A must start identically across TP ranks.
            torch.distributed.broadcast(
                attn.pic_query_a.weight.data,
                src=torch.distributed.get_process_group_ranks(base.tp_group)[0],
                group=base.tp_group)
            attn.pic_query_b = linear(args.pic_rank, config.num_attention_heads * config.kv_channels)
            torch.nn.init.zeros_(attn.pic_query_b.weight)
        attn.forward = MethodType(runtime.attention_forward, attn)
    model.forward = MethodType(runtime.forward, model)
    return runtime


class Runtime:

    def __init__(self, model, args, theta):
        self.model, self.args, self.theta = model, args, theta
        self.native_forward = model.forward
        self.mode = None
        self.captured = {}
        self.memory = {}
        self.offset = 0
        self.length = 0

    def qkv(self, attn, hidden, gist=False, adapt=False):
        projection = attn.pic_gist_qkv if gist else attn.linear_qkv
        if not gist:
            from megatron.core.tensor_parallel import copy_to_tensor_model_parallel_region

            # Bypass TE fused normalization: it is applied once before selecting
            # original/gist streams, so both share exactly the same frozen norm.
            hidden_tp = copy_to_tensor_model_parallel_region(hidden)
            bias = projection.bias if projection.bias is not None and projection.bias.numel() else None
            mixed = F.linear(hidden_tp, projection.weight, bias)
        else:
            mixed, bias = projection(hidden)
            if bias is not None:
                mixed = mixed + bias
        dim = attn.hidden_size_per_attention_head
        groups = attn.num_query_groups_per_partition
        heads = attn.num_attention_heads_per_partition
        mixed = mixed.view(*mixed.shape[:2], groups, (heads // groups + 2) * dim)
        q, k, v = torch.split(mixed, [heads // groups * dim, dim, dim], dim=-1)
        q = q.reshape(*q.shape[:2], heads, dim)
        if adapt:
            delta, _ = attn.pic_query_b(attn.pic_query_a(hidden))
            q = q + delta.view_as(q) * (self.args.pic_alpha / self.args.pic_rank)
        if attn.q_layernorm is not None:
            q = attn.q_layernorm(q)
        if attn.k_layernorm is not None:
            k = attn.k_layernorm(k)
        return q, k, v

    def attention_forward(self, attn, hidden_states, attention_mask=None, **kwargs):
        mode = self.mode
        if mode is None:
            raise RuntimeError('PIC attention must execute inside its model runtime')
        device = hidden_states.device
        if hasattr(attn.linear_qkv, 'layer_norm_weight'):
            weight = attn.linear_qkv.layer_norm_weight
            eps = self.model.config.layernorm_epsilon
            if getattr(self.args, 'pic_input_norm', 'hf') == 'hf':
                hidden_states = hf_rms_norm(hidden_states, weight, eps)
            else:
                hidden_states = F.rms_norm(hidden_states, (hidden_states.shape[-1], ), weight, eps)
        size = hidden_states.shape[0]
        if mode == 'gist':
            n = self.length
            original, gist = hidden_states[:n], hidden_states[n:]
            if attn.layer_number == 1:
                gist = gist + original  # upstream embed-mean at ratio 1
            q, k, v = self.qkv(attn, original)
            gq, gk, gv = self.qkv(attn, gist, gist=True)
            self.captured[attn.layer_number] = (gk, gv)  # pre-RoPE, retains gradients
            q, k, v = (torch.cat(pair, dim=0) for pair in ((q, gq), (k, gk), (v, gv)))
            positions = torch.arange(n, device=device).repeat(2)
            allowed = gist_mask(n, self.args.pic_overlap, device)
        else:
            q, k, v = self.qkv(
                attn,
                hidden_states,
                # Upstream gist_param='qkv' trains encoder gist projections;
                # only uppercase 'QKV' replaces the reader projections.
                gist=False,
                adapt=mode in ('read', 'prefix') and self.args.pic_algorithm == 'attuner')
            positions = torch.arange(size, device=device) + self.offset
            allowed = torch.ones(size, size, device=device, dtype=torch.bool).tril()
        q, k = rotary(q, positions, self.theta), rotary(k, positions, self.theta)
        if mode in ('artifact', 'prefix'):
            self.captured[attn.layer_number] = (k, v)
        if mode == 'read' and self.memory:
            mk, mv = self.memory[attn.layer_number]
            k, v = torch.cat([mk, k]), torch.cat([mv, v])
            allowed = torch.cat([torch.ones(size, mk.shape[0], device=device, dtype=torch.bool), allowed], dim=1)
        return attn.linear_proj(attention(q, k, v, allowed))

    def encode(self, ids, mode):
        self.mode, self.offset, self.length = mode, 0, ids.shape[1]
        self.captured = {}
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        hidden = self.model.embedding(input_ids=ids, position_ids=positions)
        if mode == 'gist':
            embedding = self.model.pic_gist_embedding
            hidden = torch.cat([hidden, embedding.view(1, 1, -1).expand_as(hidden)])
        self.model.decoder(hidden_states=hidden, attention_mask=None)
        return self.captured

    def logits(self, ids, mode, offset=0):
        self.mode, self.offset = mode, offset
        positions = torch.arange(ids.shape[1], device=ids.device)[None] + offset
        return self.native_forward(
            input_ids=ids, position_ids=positions, attention_mask=None, runtime_gather_output=True)

    def forward(self, model, pic_batch):
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            context = (
                sdpa_kernel(SDPBackend.MATH) if getattr(self.args, 'pic_numerical_reference', False) else nullcontext())
            with context:
                return self.losses(pic_batch)
        finally:
            # Never persist autograd graphs across microbatches/checkpoint saves.
            self.mode, self.memory, self.captured = None, {}, {}

    def losses(self, batch):
        prefix, suffix = batch['system_input_ids'], batch['input_ids']
        docs = [row[row != -100][None] for row in batch['context_input_ids'][0]]
        if batch['cache_prefix_ids'].numel():
            raise ValueError('Megatron PIC does not yet support cache_prefix_ids')
        teacher = None
        if self.args.pic_algorithm == 'attuner':
            with torch.no_grad():
                full = torch.cat([prefix, *docs, suffix], dim=1)
                teacher = self.logits(full, 'teacher')[:, -suffix.shape[1]:].detach()
        parts = []
        if prefix.numel():
            if self.args.pic_algorithm == 'c2kv':
                with torch.no_grad():
                    parts.append(self.encode(prefix, 'prefix'))
            else:
                parts.append(self.encode(prefix, 'prefix'))
        offset = prefix.shape[1]
        for ids in docs:
            if self.args.pic_algorithm == 'c2kv':
                raw = self.encode(ids, 'gist')
                pos = torch.arange(ids.shape[1], device=ids.device) + offset
                parts.append({layer: (rotary(k, pos, self.theta), v) for layer, (k, v) in raw.items()})
            else:
                with torch.no_grad():
                    parts.append(self.encode(ids, 'artifact'))
            offset += ids.shape[1]
        self.memory = {
            layer: tuple(torch.cat([part[layer][i] for part in parts]) for i in (0, 1))
            for layer in parts[0]
        }
        student = self.logits(suffix, 'read', offset)[:, :-1].float()
        targets = batch['labels'][:, 1:]
        if teacher is None:
            return F.cross_entropy(student.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
        temperature = self.args.pic_temperature
        return F.kl_div(
            F.log_softmax(student / temperature, -1),
            F.softmax(teacher[:, :-1].float() / temperature, -1),
            reduction='none').sum(-1) * temperature**2
