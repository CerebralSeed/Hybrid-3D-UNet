import torch
from torch import nn, einsum
import math
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import diffsubmodules as dsm

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

        block_klass = partial(dsm.ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = dsm.RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = dsm.SinusoidalPosEmb(dim)
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
                dsm.Residual(dsm.PreNorm(dim_in, dsm.Attention(dim_in, attn_type="linear"))),
                dsm.Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = dsm.Residual(dsm.PreNorm(mid_dim, dsm.Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                dsm.Residual(dsm.PreNorm(dim_out, dsm.Attention(dim_out, attn_type="linear"))),
                dsm.Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
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
        block_klass = partial(dsm.ResnetBlock, groups = resnet_block_groups)
        block_klass3d = partial(dsm.ResnetBlock3d, groups = 2)
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
                dsm.Residual(dsm.PreNorm(dim_in, dsm.EfficientAttention(dim_in))),
                dsm.Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1),
                #3d
                block_klass3d(dim_in3d, dim_in3d, time_emb_dim=self.time_dim),
                block_klass3d(dim_in3d, dim_in3d, time_emb_dim=self.time_dim),
                dsm.Residual(dsm.PreNorm3d(dim_in3d, dsm.EfficientAttention3d(dim_in3d))),
                dsm.Downsample3d(dim_in3d, dim_out3d) if not is_last else nn.Conv3d(dim_in3d, dim_out3d, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        mid_dim3d=in_out3d[-1][-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn = dsm.Residual(dsm.PreNorm(mid_dim, dsm.EfficientAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        self.mid_block1_3d = block_klass3d(mid_dim3d, mid_dim3d, time_emb_dim = self.time_dim)
        self.mid_attn_3d = dsm.Residual(dsm.PreNorm3d(mid_dim3d, dsm.EfficientAttention3d(mid_dim3d)))
        self.mid_block2_3d = block_klass3d(mid_dim3d, mid_dim3d, time_emb_dim = self.time_dim)

        self.mid_block3 = block_klass(mid_dim*2, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn2 = dsm.Residual(dsm.PreNorm(mid_dim, dsm.EfficientAttention(mid_dim)))
        self.mid_block4 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        for ind, ((dim_in, dim_out),(dim_in3d, dim_out3d)) in enumerate(zip(reversed(in_out), reversed(in_out3d))):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([

                block_klass(dim_out*2 + dim_in, dim_out, time_emb_dim = self.time_dim),
                block_klass(dim_out*2 + dim_in, dim_out, time_emb_dim = self.time_dim),

                dsm.Residual(dsm.PreNorm(dim_out, dsm.EfficientAttention(dim_out))),
                dsm.Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = self.time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def time_embeddings(self, learned_sinusoidal_cond, random_fourier_features):
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = dsm.RandomOrLearnedSinusoidalPosEmb(self.learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = self.learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = dsm.SinusoidalPosEmb(self.dim)
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


class Unet3d_v2(nn.Module):
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

        block_klass = partial(dsm.ResnetBlock, groups = resnet_block_groups)
        block_klass3d = partial(dsm.ResnetBlock3d, groups = 2)
        # time embeddings


        self.time_embeddings(learned_sinusoidal_cond, random_fourier_features)

        # layers
        self.updim=dsm.Conv2dtoConv3d(self.init_dim, dims[0],dim, kernel=3)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                dsm.Residual(dsm.PreNorm(dim_in, dsm.EfficientAttention(dim_in))),
                dsm.Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1),
                #3d
                block_klass3d(dim_in, dim_in, time_emb_dim=self.time_dim),
                block_klass3d(dim_in, dim_in, time_emb_dim=self.time_dim),
                dsm.Residual(dsm.PreNorm3d(dim_in, dsm.EfficientAttention3d(dim_in))),
                dsm.Downsample3d(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn = dsm.Residual(dsm.PreNorm(mid_dim, dsm.EfficientAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        self.mid_block1_3d = block_klass3d(mid_dim, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn_3d = dsm.Residual(dsm.PreNorm3d(mid_dim, dsm.EfficientAttention3d(mid_dim)))
        self.mid_block2_3d = block_klass3d(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        self.mid_block3 = block_klass(mid_dim*2, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn2 = dsm.Residual(dsm.PreNorm(mid_dim, dsm.EfficientAttention(mid_dim)))
        self.mid_block4 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        self.middowndim=dsm.Conv3dtoConv2d(mid_dim,mid_dim , 4, 3)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                dsm.Conv3dtoConv2d(dim_in,dim_out,2*2**(ind+1), kernel=3),
                block_klass(dim_out*2 + dim_in, dim_out, time_emb_dim = self.time_dim),
                dsm.Conv3dtoConv2d(dim_in, dim_out, 2*2**(ind + 1), kernel=3),
                block_klass(dim_out*2 + dim_in, dim_out, time_emb_dim = self.time_dim),
                dsm.Residual(dsm.PreNorm(dim_out, dsm.EfficientAttention(dim_out))),
                dsm.Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = self.time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def time_embeddings(self, learned_sinusoidal_cond, random_fourier_features):
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = dsm.RandomOrLearnedSinusoidalPosEmb(self.learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = self.learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = dsm.SinusoidalPosEmb(self.dim)
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
        j = self.updim(j)

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
        j=self.middowndim(j)
        x=torch.cat([x, j.reshape(-1, x.size()[1], x.size()[2], x.size()[3])],dim=1)

        x = self.mid_block3(x, t)
        x = self.mid_attn2(x)
        x = self.mid_block4(x, t)

        for downdim3d1, block1, downdim3d2, block2, attn, upsample in self.ups:
            j1, j2, x1, x2 = h.pop(), h.pop(), h.pop(), h.pop()
            j1=downdim3d1(j1)
            x = torch.cat((x,j1, x1), dim = 1)
            x = block1(x, t)
            j2=downdim3d2(j2)
            x = torch.cat((x,j2, x2), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


