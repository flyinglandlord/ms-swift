"""C2KV recipe input transformations, ported from upstream python/train/trainer.py.

Reference revision: 832ccd9c985055a95c1e219383e043bb005a1b36.
Runtime/optimizer remain owned by Swift. This module never starts another Trainer.
"""
import torch


def prepare_chunked(batch, config, recipe):
    inputs = dict(batch)
    batch_size, length = inputs['input_ids'].shape
    sizes, max_chunks = config.get('gist_mode', '1024-16').split('-')
    lengths = inputs['attention_mask'].sum(dim=1)
    if recipe == 'sft':
        lengths = lengths - (inputs['labels'] != -100).sum(dim=1)
    minimum = int(lengths.min())
    sizes = [int(x) for x in sizes.split(',') if int(x) < minimum]
    if not sizes:
        raise ValueError('No gist chunk size smaller than the minimum prompt/sequence')
    size = sizes[minimum % len(sizes)]
    chunks = min((minimum - 1) // size, int(max_chunks))
    context_length = size * chunks
    inputs['context_input_ids'] = inputs['input_ids'][:, :context_length].reshape(batch_size, chunks, size)
    inputs['input_ids'] = inputs['input_ids'][:, context_length:]
    inputs['attention_mask'] = inputs['attention_mask'][:, context_length:].bool()
    inputs['position_ids'] = torch.arange(
        context_length, length, device=inputs['input_ids'].device)[None].expand(batch_size, -1)
    if recipe == 'sft':
        inputs['labels'] = inputs['labels'][:, context_length:]
    else:
        inputs['labels'] = inputs['input_ids'].clone().masked_fill(~inputs['attention_mask'], -100)
        inputs['reconstruct_loss_coef'] = config.get('gist_reconstruct_loss_coef')
    return inputs


def prepare_mdoc(model, batch, config):
    inputs = dict(batch)
    system = inputs.pop('system_input_ids')
    valid = system != -100
    lengths = valid.sum(dim=1)
    width = int(lengths.max())
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    ids = system.new_full((system.shape[0], width), pad)
    mask = system.new_zeros(ids.shape)
    for i, n in enumerate(lengths.tolist()):
        if n:
            ids[i, width - n:] = system[i][valid[i]]
            mask[i, width - n:] = 1
    was_training = model.training
    if config.get('system_attn_impl'):
        model.config._attn_implementation = config['system_attn_impl']
    model.eval()
    try:
        with torch.no_grad():
            cache = model(ids, attention_mask=mask, use_cache=True, logits_to_keep=1).past_key_values
    finally:
        model.train(was_training)
    context = inputs['context_input_ids']
    inputs['context_input_ids'] = context.reshape(context.shape[0], -1, config.get('max_doc_length', 1024))
    inputs['past_key_values'] = cache
    inputs['past_attention_mask'] = mask
    positions = torch.arange(inputs['input_ids'].shape[1], device=system.device)[None]
    inputs['position_ids'] = positions + width + (context != -100).sum(dim=1, keepdim=True)
    inputs['reconstruct_loss_coef'] = config.get('gist_reconstruct_loss_coef')
    if config.get('context_attn_impl'):
        model.config._attn_implementation = config['context_attn_impl']
    return inputs


class UpstreamCollator:
    """Already-tokenized upstream fields. Preserve fixed document/system padding."""

    def __call__(self, rows):
        result = {}
        for key in ('input_ids', 'labels', 'attention_mask', 'context_input_ids', 'system_input_ids'):
            if key in rows[0]:
                values = [torch.tensor(row[key], dtype=torch.long) for row in rows]
                if len({tuple(v.shape) for v in values}) != 1:
                    raise ValueError('Upstream records must use the original fixed-length padding')
                result[key] = torch.stack(values)
        return result


def regularized_loss(model, loss, config, reconstruction=False):
    parameters = config.get('gist_regularization')
    if parameters is None:
        return loss
    strength = config.get('gist_regularization_factor')
    if strength is None:
        raise ValueError('gist_regularization_factor is required')
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    penalties = []
    for layer in base.model.layers:
        for key in parameters:
            if key in 'qkv':
                original = getattr(layer.self_attn, f'{key}_proj').weight
                gist = getattr(layer.self_attn, f'gist_{key}_proj').weight
                penalties.append((original - gist).pow(2).mean())
    if not penalties:
        raise ValueError('gist_regularization must select q, k or v')
    # Upstream wraps model.loss_function, which is also called by reconstruction.
    multiplier = 1 + config['gist_reconstruct_loss_coef'] if reconstruction else 1
    return loss + torch.stack(penalties).mean() * strength * multiplier
