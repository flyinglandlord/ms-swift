"""Experimental ``megatron pic`` pipeline using Swift's native Megatron trainer."""
import torch
from dataclasses import dataclass
from datasets import Dataset
from functools import partial

from swift.megatron.arguments import MegatronSftArguments
from swift.megatron.model import get_mcore_model
from swift.megatron.pipelines.train.sft import MegatronSft
from swift.megatron.trainers import MegatronTrainer
from swift.megatron.utils import wrap_model
from swift.pic.data import SegmentCollator, assert_disjoint, read_records
from .model import install


def validate(args):
    if args.pic_algorithm not in ('c2kv', 'attuner'):
        raise ValueError('pic_algorithm must be c2kv or attuner')
    if args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1:
        raise ValueError('PIC currently requires PP=CP=1')
    if args.sequence_parallel or args.packing or args.padding_free:
        raise ValueError('PIC does not yet support SP, packing or padding-free')
    if args.recompute_granularity not in (None, 'none'):
        raise ValueError('PIC cache lifetime currently excludes activation recomputation')
    if args.micro_batch_size != 1 or args.tuner_type != 'full':
        raise ValueError('Use micro_batch_size=1 and tuner_type=full; PIC owns parameter freezing')
    if args.save_safetensors:
        raise ValueError('PIC extra parameters require native MCore checkpoints: save_safetensors=false')
    if args.pic_rank <= 0 or args.pic_temperature <= 0 or args.pic_overlap < 0:
        raise ValueError('Invalid PIC rank, temperature or overlap')
    if args.bridge_backend != 'mcore-bridge':
        raise ValueError('Use bridge_backend=mcore-bridge')
    if args.task_type != 'causal_lm':
        raise ValueError('PIC currently requires causal_lm')
    if args.adapters or args.mcore_adapter or args.fp8_format or args.fp4_format:
        raise ValueError('PIC currently requires full native checkpoints and non-FP8/FP4 weights')
    if args.pic_hf_gist_checkpoint and (args.mcore_model is not None or args.pic_algorithm != 'c2kv'):
        raise ValueError('HF gist import requires fresh C2KV initialization, not native resume')


@dataclass
class MegatronPICArguments(MegatronSftArguments):
    task_type: str = 'causal_lm'
    pic_algorithm: str = 'c2kv'
    pic_overlap: int = 64
    pic_rank: int = 8
    pic_alpha: float = 16.
    pic_temperature: float = 1.
    pic_allow_reference_responses: bool = False
    pic_hf_gist_checkpoint: str = None
    pic_numerical_reference: bool = False
    pic_input_norm: str = 'hf'
    padding_free: bool = False
    recompute_granularity: str = 'none'
    save_safetensors: bool = False

    def __post_init__(self):
        validate(self)
        super().__post_init__()
        if torch.distributed.get_world_size() != self.tensor_model_parallel_size:
            raise ValueError('PIC reference backend currently requires data parallel size 1')


class MegatronPICTrainer(MegatronTrainer):

    def prepare_model(self):
        self.unwrapped_models = get_mcore_model(self.args, self.template.config)
        if len(self.unwrapped_models) != 1:
            raise ValueError('PIC requires a single local model chunk')
        self.config = self.unwrapped_models[0].config
        self.bridge = self.config.bridge
        # Load the original backbone before adding algorithm-owned parameters.
        if self.args.mcore_model is None:
            self.bridge.load_weights(self.unwrapped_models, self.args.model_dir)
        self.pic_runtime = install(self.unwrapped_models[0], self.args)
        if self.args.pic_hf_gist_checkpoint:
            from .checkpoint import load_gist
            load_gist(self.unwrapped_models[0], self.args.pic_hf_gist_checkpoint, self.args.pic_overlap)
        self.peft_models = self.unwrapped_models
        self.wrapped_models = wrap_model(self.args, self.unwrapped_models)

    def _get_data_collator(self):
        return SegmentCollator()

    def forward_step(self, data_iterator, model):
        batch = {key: value.cuda(non_blocking=True) for key, value in next(data_iterator).items()}
        losses = model(pic_batch=batch)
        if not torch.isfinite(losses).all():
            raise FloatingPointError('Non-finite Megatron PIC loss')
        return losses, partial(self.loss_func, labels=batch['labels'][:, 1:])


class MegatronPIC(MegatronSft):
    args_class = MegatronPICArguments

    def __init__(self, args=None):
        super().__init__(args)
        # The base argument lifecycle can restore train_dataloader_shuffle after
        # applying dataset_shuffle. Honor the explicit dataset policy for PIC.
        if self.args.dataset_shuffle is not None:
            self.args.train_dataloader_shuffle = self.args.dataset_shuffle
        self.args.save_args(self.args.output_dir)

    def prepare_trainer(self):
        return MegatronPICTrainer(self.args, self.template)

    def _prepare_dataset(self):
        args = self.args
        if len(args.dataset) != 1 or len(args.val_dataset) != 1 or args.cached_dataset:
            raise ValueError('PIC requires one segmented JSONL train file and one explicit validation file')
        train, evaluation = read_records(args.dataset[0]), read_records(args.val_dataset[0])
        assert_disjoint(train, evaluation)
        for row in train + evaluation:
            length = len(row.get('system_ids', [])) + sum(map(len, row['segments']))
            length += len(row['query_ids']) + len(row['answer_ids'])
            if length > args.max_length:
                raise ValueError('PIC does not silently truncate: example exceeds max_length')
            if row.get('cache_prefix_ids'):
                raise ValueError('cache_prefix_ids is not yet supported by Megatron PIC')
            if args.pic_algorithm == 'attuner' and not args.pic_allow_reference_responses:
                if row.get('teacher_generated') is not True:
                    raise ValueError('ATTUNER requires teacher-generated responses or explicit reference ablation')
        return Dataset.from_list(train), Dataset.from_list(evaluation)


def main():
    MegatronPIC().main()
