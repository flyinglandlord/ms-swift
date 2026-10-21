"""Explicit HF gist import and canonical GQA projection layout conversion."""
import json
import torch
from pathlib import Path
from safetensors import safe_open


def pack_qkv(q, k, v, heads, groups, dim):
    tail = q.shape[1:]
    return torch.cat(
        [q.reshape(groups, heads // groups * dim, *tail),
         k.reshape(groups, dim, *tail),
         v.reshape(groups, dim, *tail)],
        dim=1).flatten(0, 1)


def unpack_qkv(value, heads, groups, dim):
    tail = value.shape[1:]
    values = value.reshape(groups, (heads // groups + 2) * dim, *tail)
    return tuple(x.flatten(0, 1) for x in values.split([heads // groups * dim, dim, dim], dim=1))


def load_gist(model, directory, overlap=64):
    """Load only supported, complete 1:1 gist tensors; never silently discard extras."""
    root = Path(directory)
    cfg = json.loads((root / 'config.json').read_text())
    expected = dict(gist_type='interleave-1', gist_param='qkv', gist_residual_type='embed-mean', gist_overlap=overlap)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError(f'HF gist import requires {expected}')
    tensors = {}
    for file in sorted(root.glob('*.safetensors')):
        with safe_open(file, framework='pt', device='cpu') as source:
            for key in source.keys():
                if 'gist' in key:
                    if key in tensors:
                        raise ValueError(f'Duplicate tensor: {key}')
                    tensors[key] = source.get_tensor(key)
    config = model.config
    with torch.no_grad():
        embedding = tensors.pop('model.gist_embed_tokens.weight')
        if embedding.shape != (1, config.hidden_size):
            raise ValueError('Only one gist embedding is supported')
        model.pic_gist_embedding.copy_(embedding.reshape_as(model.pic_gist_embedding))
        for layer in model.decoder.layers:
            attn = layer.self_attention
            group = attn.linear_qkv.tp_group
            rank, size = torch.distributed.get_rank(group), torch.distributed.get_world_size(group)
            for kind in ('weight', 'bias'):
                target = getattr(attn.pic_gist_qkv, kind)
                if target is None:
                    continue
                prefix = f'model.layers.{layer.layer_number - 1}.self_attn.gist_'
                values = [tensors.pop(f'{prefix}{x}_proj.{kind}') for x in 'qkv']
                packed = pack_qkv(*values, config.num_attention_heads, config.num_query_groups, config.kv_channels)
                shard = packed.chunk(size, dim=0)[rank]
                if target.shape != shard.shape:
                    raise ValueError(f'Gist shape mismatch: {target.shape} != {shard.shape}')
                target.copy_(shard)
    if tensors:
        raise ValueError(f'Unsupported gist tensors: {sorted(tensors)}')
