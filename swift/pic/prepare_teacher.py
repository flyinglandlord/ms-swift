"""Build ATTUNER's fixed teacher-response dataset with an unadapted full-context model."""
import argparse
import importlib
import json
import torch
from pathlib import Path

from .data import read_records
from .registry import ARCHITECTURES, register_builtins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--architecture', default='qwen3')
    parser.add_argument('--plugin', action='append', default=[])
    parser.add_argument('--architecture-config', type=json.loads, default={})
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('Refusing to overwrite teacher data')
    register_builtins()
    for module in args.plugin:
        importlib.import_module(module)
    architecture = ARCHITECTURES[args.architecture](**args.architecture_config)
    model, processor = architecture.load(args.model, torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    rows = read_records(args.input)
    for row in rows:
        row['reference_answer_ids'] = row['answer_ids']
        row['answer_ids'] = architecture.generate_teacher(model, processor, row, args.max_new_tokens)
        row['teacher_generated'] = True
        row['teacher_model'] = str(Path(args.model).resolve())
        row['teacher_generation_limit'] = args.max_new_tokens
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(''.join(json.dumps(row) + '\n' for row in rows))


if __name__ == '__main__':
    main()
