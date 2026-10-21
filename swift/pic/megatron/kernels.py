"""Layout-independent reference kernels for the first, uncompressed PIC backend."""
import torch
import torch.nn.functional as F


def hf_rms_norm(hidden, weight, eps):
    """Match C2KV/HF rounding: normalize in FP32, cast, then multiply weight.

    Do not fuse the weight multiplication into the FP32 normalization. That
    changes BF16 rounding and was the first divergent operator in the GPU audit.
    Preserve autograd through the cast, including gradients into frozen layers.
    """
    value = hidden.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return weight * value.to(hidden.dtype)


def rotary(x, positions, theta):
    # MCore local layout: [sequence, batch, heads, head_dim].
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError('PIC requires an even rotary head dimension')
    inv = theta**(-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    phase = positions.float()[:, None] * inv[None]
    phase = torch.cat([phase, phase], -1)[:, None, None]
    rotated = torch.cat([-x[..., dim // 2:], x[..., :dim // 2]], -1)
    return x * phase.cos().to(x.dtype) + rotated * phase.sin().to(x.dtype)


def gist_mask(length, overlap, device):
    i = torch.arange(length, device=device)
    causal = i[:, None] >= i[None, :]
    local = (i[None, :] <= i[:, None]) & (i[None, :] >= i[:, None] - overlap)
    local[:, 0] = True
    mask = torch.zeros(2 * length, 2 * length, device=device, dtype=torch.bool)
    mask[:length, :length] = causal
    mask[length:, :length] = local
    mask[length:, length:] = causal
    return mask


def attention(q, k, v, allowed):
    repeats = q.shape[2] // k.shape[2]
    if q.shape[2] % k.shape[2]:
        raise ValueError('Query heads must be divisible by KV heads')
    k, v = (x.repeat_interleave(repeats, dim=2) for x in (k, v))
    output = F.scaled_dot_product_attention(
        q.permute(1, 2, 0, 3),
        k.permute(1, 2, 0, 3),
        v.permute(1, 2, 0, 3),
        attn_mask=allowed[None, None],
        dropout_p=0.)
    return output.permute(2, 0, 1, 3).contiguous().flatten(2)
