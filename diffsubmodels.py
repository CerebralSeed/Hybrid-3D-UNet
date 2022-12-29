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

    def __init__(self, in_channels, key_channels=4, head_count=4, value_channels=32, SqSoftmax=False):
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

# models

class Unet2d(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, Attention(dim_in, attn_type="linear"))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, Attention(dim_out, attn_type="linear"))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

class Unet3d(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        self.init_dim = default(init_dim, dim)
        self.dim = dim
        self.time_dim = dim * 4
        self.learned_sinusoidal_dim = learned_sinusoidal_dim

        self.init_conv = nn.Conv2d(input_channels, self.init_dim, 7, padding = 3)

        dims = [self.init_dim, *map(lambda m: self.dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        in_out3d =[(1,4), (4,16), (16,64), (64, 64)]
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        block_klass3d = partial(ResnetBlock3d, groups = 2)
        # time embeddings


        self.time_embeddings(learned_sinusoidal_cond, random_fourier_features)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, ((dim_in, dim_out), (dim_in3d, dim_out3d)) in enumerate(zip(in_out,in_out3d)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                Residual(PreNorm(dim_in, EfficientAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1),
                #3d
                block_klass3d(dim_in3d, dim_in3d, time_emb_dim=self.time_dim),
                block_klass3d(dim_in3d, dim_in3d, time_emb_dim=self.time_dim),
                Residual(PreNorm3d(dim_in3d, EfficientAttention3d(dim_in3d))),
                Downsample3d(dim_in3d, dim_out3d) if not is_last else nn.Conv3d(dim_in3d, dim_out3d, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        mid_dim3d=in_out3d[-1][-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, EfficientAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        self.mid_block1_3d = block_klass3d(mid_dim3d, mid_dim3d, time_emb_dim = self.time_dim)
        self.mid_attn_3d = Residual(PreNorm3d(mid_dim3d, EfficientAttention3d(mid_dim3d)))
        self.mid_block2_3d = block_klass3d(mid_dim3d, mid_dim3d, time_emb_dim = self.time_dim)

        self.mid_block3 = block_klass(mid_dim*2, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn2 = Residual(PreNorm(mid_dim, EfficientAttention(mid_dim)))
        self.mid_block4 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        for ind, ((dim_in, dim_out),(dim_in3d, dim_out3d)) in enumerate(zip(reversed(in_out), reversed(in_out3d))):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([

                block_klass(dim_out*2 + dim_in, dim_out, time_emb_dim = self.time_dim),
                block_klass(dim_out*2 + dim_in, dim_out, time_emb_dim = self.time_dim),

                Residual(PreNorm(dim_out, EfficientAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = self.time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def time_embeddings(self, learned_sinusoidal_cond, random_fourier_features):
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(self.learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = self.learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(self.dim)
            fourier_dim = self.dim

        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, self.time_dim), nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim))

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        x = self.init_conv(x)
        r = x.clone()
        j = x.clone()
        t = self.time_mlp(time)
        h = []
        j = j.unsqueeze(1)
        for block1, block2, attn, downsample, block1_3d, block2_3d, attn3d, downsample3d in self.downs:

            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

            j = block1_3d(j, t)
            h.append(j)

            j = block2_3d(j, t)
            j = attn3d(j)
            h.append(j)

            j = downsample3d(j)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        j=self.mid_block1_3d(j,t)
        j= self.mid_attn_3d(j)
        j=self.mid_block2_3d(j,t)

        x=torch.cat([x, j.reshape(-1, x.size()[1], x.size()[2], x.size()[3])],dim=1)

        x = self.mid_block3(x, t)
        x = self.mid_attn2(x)
        x = self.mid_block4(x, t)

        for block1, block2, attn, upsample in self.ups:
            j1, j2, x1, x2 = h.pop(), h.pop(), h.pop(), h.pop()
            x = torch.cat((x,j1.reshape(-1, x.size()[1],x.size()[2], x.size()[3]), x1), dim = 1)
            x = block1(x, t)
            x = torch.cat((x, j2.reshape(-1, x.size()[1],x.size()[2], x.size()[3]), x2), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)





