import torch
from torch import nn, einsum
import math
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F

import time

from random import random
from functools import partial

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
def Downsample3d(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) (l p3) -> b (c p1 p2 p3) h w l', p1 = 2, p2 = 2, p3 = 2),
        nn.Conv3d(dim * 8, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class WeightStandardizedConv3d(nn.Conv3d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps)**(1/3.)

        return F.conv3d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LayerNorm3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm3d(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm3d(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class Block3d(nn.Module):
    def __init__(self, dim, dim_out, groups = 2):
        super().__init__()
        self.proj = WeightStandardizedConv3d(dim, dim_out, 3, padding = 1)
        self.dim=dim
        if self.dim%2==0:
            self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        if self.dim % 2 == 0:
            x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class ResnetBlock3d(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block3d(dim, dim_out, groups = groups)
        self.block2 = Block3d(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention3d(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, attn_type="linear"):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.attn_type=attn_type

        if attn_type=="linear":
            self.to_out = nn.Sequential(
                nn.Conv3d(hidden_dim, dim, 1),
                LayerNorm(dim)
            )
        else:
            self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):

        b, c, h, w, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h = self.heads), qkv)

        if self.attn_type=="linear":
            q = q.softmax(dim = -2)
            k = k.softmax(dim = -1)
            v = v / (h * w *l)
        print(q.size(), k.size(), v.size())
        q = q * self.scale
        if self.attn_type=="linear":
            sim = einsum('b h d i, b h d j -> b h i j', q, k)
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h d j -> b h i d', attn, v)
            out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x=h, y=w, z=l)

        else:
            context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
            out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
            out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w, z = l)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, attn_type="linear"):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.attn_type = attn_type

        if attn_type == "linear":
            self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))
        else:
            self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        if self.attn_type == "linear":
            q = q.softmax(dim=-2)
            k = k.softmax(dim=-1)
            v = v / (h * w)

        q = q * self.scale
        if self.attn_type == "linear":
            context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
            out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
            out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        else:
            sim = einsum('b h d i, b h d j -> b h i j', q, k)
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h d j -> b h i d', attn, v)
            out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class SqrSoftmax(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim=dim
    def forward(self, input_):
        input_=input_.clamp(-10,10)
        c=input_**2
        k=c.sum(self.dim).unsqueeze(self.dim)
        k[k==0]+=0.0001
        return c*k**(-1)

class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels=4, head_count=32, value_channels=32, SqSoftmax=False):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.sqsoftmax=SqSoftmax
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        if SqSoftmax:
            self.sqrsoftmax = SqrSoftmax()

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            if self.sqsoftmax:
                key = self.sqrsoftmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
                query = self.sqrsoftmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            else:
                key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
                query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class EfficientAttention3d(nn.Module):

    def __init__(self, in_channels, key_channels=8, head_count=8, value_channels=64, SqSoftmax=False):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.sqsoftmax=SqSoftmax
        self.keys = nn.Conv3d(in_channels, key_channels, 1)
        self.queries = nn.Conv3d(in_channels, key_channels, 1)
        self.values = nn.Conv3d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv3d(value_channels, in_channels, 1)
        if SqSoftmax:
            self.sqrsoftmax = SqrSoftmax()

    def forward(self, input_):
        n, _, h, w, l = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w * l))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w * l)
        values = self.values(input_).reshape((n, self.value_channels, h * w * l))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            if self.sqsoftmax:
                key = self.sqrsoftmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
                query = self.sqrsoftmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            else:
                key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
                query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w, l)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class SOFTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W, L = x.shape
        N = self.num_heads
        x = x.reshape()
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Conv1dtoConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth_dim, kernel=None, bias=False):
        super(Conv1dtoConv2d, self).__init__()
        self.depth_dim = depth_dim
        assert kernel%2==1
        if not kernel:
            kernel_size=1
            padding=0
        else:
            kernel_size=kernel
            padding = kernel//2
        self.entry1d = nn.Sequential(nn.Conv1d(in_channels, depth_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=bias))
        self.exit2d = nn.Sequential(nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
                                    nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.entry1d(x).unsqueeze(1)
        x = self.exit2d(x)
        return x

class Conv2dtoConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, depth_dim, kernel=None, bias=False):
        super(Conv2dtoConv3d, self).__init__()
        self.depth_dim = depth_dim
        assert kernel % 2 == 1
        if not kernel:
            kernel_size=1
            padding=0
        else:
            kernel_size=kernel
            padding = kernel//2
        self.entry2d = nn.Sequential(nn.Conv2d(in_channels, depth_dim, kernel_size=kernel_size, stride=1,
                                               padding=padding, bias=bias))
        self.exit3d = nn.Sequential(nn.Conv3d(1, out_channels, kernel_size=kernel_size, stride=1,
                                              padding=padding, bias=bias),
                                    nn.BatchNorm3d(out_channels))

    def forward(self, x):
        x = self.entry2d(x).unsqueeze(1)
        x = self.exit3d(x)
        return x


class Conv2dtoConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, depth_dim, kernel=None, bias=False):
        super(Conv2dtoConv1d, self).__init__()
        self.depth_dim=depth_dim
        assert kernel % 2 == 1
        if not kernel:
            kernel_size=1
            padding=0
        else:
            kernel_size=kernel
            padding = kernel//2
        self.entry2d = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=bias))
        self.exit1d = nn.Sequential(nn.Conv1d(depth_dim, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
                                    nn.BatchNorm1d(out_channels))

    def forward(self, x):
        x = self.entry2d(x).squeeze(1)
        x = self.exit1d(x)
        return x

class Conv3dtoConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth_dim, kernel=None, bias=False):
        super(Conv3dtoConv2d, self).__init__()
        self.depth_dim = depth_dim
        assert kernel % 2 == 1
        if not kernel:
            kernel_size=1
            padding=0
        else:
            kernel_size=kernel
            padding = kernel//2
        self.entry3d = nn.Sequential(nn.Conv3d(in_channels, 1, kernel_size=kernel_size, stride=1,
                                               padding=padding, bias=bias))
        self.exit2d = nn.Sequential(nn.Conv2d(depth_dim, out_channels, kernel_size=kernel_size, stride=1,
                                              padding=padding, bias=bias),
                                    nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.entry3d(x).squeeze(1)
        x = self.exit2d(x)
        return x

