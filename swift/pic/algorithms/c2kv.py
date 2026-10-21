import torch.nn.functional as F

from ..cache import C2KVExtractor
from ..contracts import Algorithm
from .c2kv_recipes import UpstreamCollator, prepare_chunked, prepare_mdoc, regularized_loss


class C2KV(Algorithm):
    required_capabilities = frozenset({'c2kv'})
    objective_name = 'answer_cross_entropy'
    preserve_explicit_checkpointing = True

    def __init__(self, architecture, options):
        super().__init__(architecture, options)
        self.config = getattr(options, 'algorithm_config', {})
        self.enabled = self.config.get('enable_gist', True)
        if self.enabled:
            architecture.prepare_c2kv()
        self.extractor = C2KVExtractor()
        self.adapter_checkpoint = options.tuning == 'lora'
        self.recipe = self.config.get('recipe', 'segmented')
        self.accepts_loss_kwargs = self.recipe != 'segmented'
        if self.recipe not in ('segmented', 'mdoc', 'sft', 'pretrain'):
            raise ValueError(f'Unknown C2KV recipe: {self.recipe}')
        if self.recipe != 'segmented' and 'c2kv_context' not in architecture.capabilities:
            raise ValueError('Upstream recipes require c2kv_context capability')
        if not self.enabled and self.recipe not in ('sft', 'pretrain'):
            raise ValueError('enable_gist=False requires sft or pretrain')
        if self.config.get('gist_self_distill_coef') is not None:
            raise ValueError('Self-distillation is not implemented for these C2KV recipes; do not silently ignore it')
        if self.config.get('gist_reconstruct_loss_coef') is not None:
            if self.recipe not in ('mdoc', 'pretrain'):
                raise ValueError('Reconstruction requires mdoc or pretrain recipe')
            architecture.model_config.setdefault('gist_extra_embed_num', 2)

    def collator(self):
        return super().collator() if self.recipe == 'segmented' else UpstreamCollator()

    def configure(self, model):
        if not self.enabled:
            if self.options.tuning == 'lora':
                raise ValueError('Vanilla LoRA is not a C2KV gist configuration')
            model.requires_grad_(True)
            return model
        return self.architecture.configure_c2kv(model, self.options)

    def trainer_loss(self, model, batch, num_items_in_batch=None):
        return self.loss(model, batch, num_items_in_batch)

    def loss(self, model, batch, num_items_in_batch=None):
        if not self.enabled:
            inputs = dict(batch)
            if self.recipe == 'pretrain':
                prepared = prepare_chunked(batch, self.config, 'pretrain')
                context_length = prepared['context_input_ids'].shape[1] * prepared['context_input_ids'].shape[2]
                inputs['labels'] = inputs['input_ids'].clone().masked_fill(~inputs['attention_mask'].bool(), -100)
                inputs['labels'][:, :context_length] = -100
            if num_items_in_batch is not None:
                inputs['num_items_in_batch'] = num_items_in_batch
            outputs = self.architecture.context_forward(model, inputs)
            return outputs.loss, outputs
        if self.recipe != 'segmented':
            inputs = (
                prepare_mdoc(model, batch, self.config) if self.recipe == 'mdoc' else prepare_chunked(
                    batch, self.config, self.recipe))
            if num_items_in_batch is not None:
                inputs['num_items_in_batch'] = num_items_in_batch
            outputs = self.architecture.context_forward(model, inputs)
            loss = regularized_loss(
                model, outputs.loss, self.config, reconstruction=getattr(outputs, 'reconstruct_loss', None) is not None)
            outputs.loss = loss
            return loss, outputs
        memory = self.extractor.extract(model, self.architecture, batch)
        outputs = self.architecture.read(model, batch, memory)
        logits, targets = self.architecture.response_logits(outputs, batch)
        loss = regularized_loss(model, F.cross_entropy(logits.float(), targets), self.config)
        outputs.loss = loss
        return loss, outputs
