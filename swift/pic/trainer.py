import time
import torch
from transformers import Trainer as HFTrainer

from swift.trainers import Seq2SeqTrainer
from swift.trainers.patcher import PrinterCallbackNew, ProgressCallbackNew


class PICTrainer(Seq2SeqTrainer):
    """Swift owns the optimizer/checkpoints; algorithms own model execution and loss."""

    def __init__(self, *args, algorithm, **kwargs):
        self.algorithm = algorithm
        self.pic_callbacks = kwargs.pop('callbacks', [])
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = algorithm.accepts_loss_kwargs
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, (ProgressCallbackNew, PrinterCallbackNew)):
                callback.start_time = time.time()
                callback.start_step = 0

    def _get_callbacks(self, args):
        return super()._get_callbacks(args) + self.pic_callbacks

    def _fix_gradient_checkpointing(self):
        # C2KV explicitly requests reentrant gist recomputation. Swift's global
        # checkpoint wrapper otherwise overrides even explicit use_reentrant=True
        # under torchrun, changing unused gradients (zero versus None) and AdamW
        # decay of the last encoder query projection.
        if not getattr(self.algorithm, 'preserve_explicit_checkpointing', False):
            super()._fix_gradient_checkpointing()

    def _get_data_collator(self, args, template):
        return self.algorithm.collator()

    def _prepare_inputs(self, inputs):
        return HFTrainer._prepare_inputs(self, inputs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = self.algorithm.trainer_loss(model, inputs, num_items_in_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite PIC objective')
        return (loss, outputs) if return_outputs else loss
