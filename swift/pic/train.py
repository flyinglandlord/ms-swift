"""Single-process research entry point: swift pic --algorithm c2kv|attuner ..."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import torch
from datasets import Dataset
from pathlib import Path
from transformers import set_seed

from swift.template import get_template
from swift.trainers import Seq2SeqTrainingArguments
from .data import assert_disjoint, read_records
from .registry import ALGORITHMS, ARCHITECTURES, register_builtins
from .trainer import PICTrainer


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', default='c2kv')
    p.add_argument('--architecture', default='qwen3')
    p.add_argument('--plugin', action='append', default=[])
    p.add_argument('--architecture-config', type=json.loads, default={})
    p.add_argument('--algorithm-config', type=json.loads, default={})
    p.add_argument(
        '--training-config',
        type=json.loads,
        default={},
        help='Swift/HF TrainingArguments overrides (batch size, optimizer, warmup, schedule, etc.)')
    p.add_argument('--seed', type=int, default=2948)
    p.add_argument('--dtype', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    p.add_argument('--model', required=True)
    p.add_argument('--train-data', required=True)
    p.add_argument('--eval-data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--gradient-accumulation', type=int, default=1)
    p.add_argument('--tuning', choices=['gist', 'lora'], default='gist')
    p.add_argument('--lora-scope', choices=['gist', 'base', 'both'], default='gist')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=int, default=16)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--cache-bytes', type=int, default=64 * 1024 * 1024)
    p.add_argument('--allow-reference-responses', action='store_true')
    p.add_argument('--resume-from-checkpoint')
    p.add_argument('--reload-run', help='Evaluate a final checkpoint/adapter from a completed run')
    return p


def build(options, reload_path=None):
    register_builtins()
    for module in options.plugin:
        importlib.import_module(module)
    architecture_config = getattr(options, 'architecture_config', {})
    if not isinstance(architecture_config, dict) or not isinstance(getattr(options, 'algorithm_config', {}), dict):
        raise ValueError('Architecture/algorithm config must be JSON objects')
    architecture = ARCHITECTURES[options.architecture](**architecture_config)
    algorithm = ALGORITHMS[options.algorithm](architecture, options)
    set_seed(getattr(options, 'seed', 2948))
    adapter = algorithm.adapter_checkpoint
    source = str(reload_path / 'final') if reload_path and not adapter else options.model
    dtype = getattr(options, 'dtype', 'auto')
    dtype = (torch.bfloat16 if torch.cuda.is_available() else torch.float32) if dtype == 'auto' else {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32
    }[dtype]
    model, processor = architecture.load(source, dtype)
    if reload_path and adapter:
        model = algorithm.load_adapter(model, reload_path / 'final')
    else:
        model = algorithm.configure(model)
    template = get_template(processor, template_type=architecture.template, padding_free=False)
    template.set_mode('train')
    return model, template, algorithm, adapter


def main():
    options = parser().parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or os.environ.get('NPROC_PER_NODE', '1') != '1':
        raise ValueError('PIC initial backend is single-process; distributed execution is not yet validated')
    reload_path = Path(options.reload_run) if options.reload_run else None
    if reload_path:
        manifest = json.loads((reload_path / 'pic_manifest.json').read_text())
        options = argparse.Namespace(**manifest['options'])
        options.output = str(reload_path / 'reload_eval')
    if options.algorithm == 'c2kv' and options.algorithm_config.get('recipe', 'segmented') != 'segmented':

        def read_native(path):
            rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
            if not rows or len({x['id'] for x in rows}) != len(rows):
                raise ValueError('Empty upstream data or duplicate IDs')
            return rows

        train, evaluation = read_native(options.train_data), read_native(options.eval_data)
    else:
        train, evaluation = read_records(options.train_data), read_records(options.eval_data)
    assert_disjoint(train, evaluation)
    output = Path(options.output)
    if output.exists() and any(output.iterdir()) and not reload_path and not options.resume_from_checkpoint:
        raise ValueError('Output is nonempty; use a new directory or explicit resume')
    output.mkdir(parents=True, exist_ok=True)
    model, template, algorithm, adapter = build(options, reload_path)
    algorithm.validate_records(train, evaluation)
    training_config = dict(
        output_dir=str(output),
        max_steps=options.steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=options.gradient_accumulation,
        learning_rate=options.learning_rate,
        warmup_ratio=None,
        bf16=torch.cuda.is_available() and getattr(options, 'dtype', 'auto') in ('auto', 'bf16'),
        fp16=getattr(options, 'dtype', 'auto') == 'fp16',
        gradient_checkpointing=False,
        remove_unused_columns=False,
        logging_steps=1,
        save_steps=options.steps,
        eval_strategy='no',
        report_to=[],
        check_model=False,
        prediction_loss_only=True,
        tuner_type='lora' if adapter else 'full',
        seed=getattr(options, 'seed', 2948))
    overrides = getattr(options, 'training_config', {})
    if not isinstance(overrides, dict):
        raise ValueError('training-config must be a JSON object')
    if 'seed' in overrides and overrides['seed'] != getattr(options, 'seed', 2948):
        raise ValueError('Use --seed consistently for model initialization and Trainer')
    training_config.update(overrides)
    args = Seq2SeqTrainingArguments(**training_config)
    trainer = PICTrainer(
        model=model,
        args=args,
        template=template,
        algorithm=algorithm,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(evaluation))
    before = trainer.evaluate()
    if reload_path:
        difference = abs(before['eval_loss'] - manifest['eval_after']['eval_loss'])
        result = {'eval': before, 'difference': difference, 'passed': difference < 1e-5}
        (reload_path / 'reload_verification.json').write_text(json.dumps(result, indent=2))
        if not result['passed']:
            raise AssertionError(f'Reload mismatch: {difference}')
        return
    result = trainer.train(resume_from_checkpoint=options.resume_from_checkpoint)
    after = trainer.evaluate()
    trainer.save_model(str(output / 'final'))
    trainer.save_state()
    manifest = {
        'options': vars(options),
        'eval_before': before,
        'eval_after': after,
        'objective': algorithm.objective_name,
        'train_metrics': result.metrics,
        'resolved_training_args': args.to_dict(),
        'algorithm_statistics': algorithm.statistics(),
        'trainable_names': [name for name, param in model.named_parameters() if param.requires_grad],
        'trainable_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'versions': {
            n: importlib.metadata.version(n)
            for n in ['ms-swift', 'transformers', 'peft', 'torch']
        },
        'swift_source': str(Path(__file__).resolve()),
        'data_sha256': {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [options.train_data, options.eval_data]
        }
    }
    (output / 'pic_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
