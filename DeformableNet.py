import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
from base_networks import *
import torchvision
import torchvision.ops

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
import pdb, os

import warnings
from einops import rearrange, repeat



class LayerNormChannel(nn.Module):
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

class ConvAttention(nn.Module):
    def __init__(self, dim, attn_dim, kernel_size=3, bias=False):
        super().__init__()
        self.attn_conv = nn.Conv2d(
                                dim, 
                                attn_dim, 
                                kernel_size=kernel_size, 
                                stride=1, 
                                padding=kernel_size // 2, 
                                bias=bias
                                )
        self.value_conv = nn.Conv2d(dim, attn_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias)
        self.proj = nn.Conv2d(attn_dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm = LayerNormChannel(attn_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        attn = self.attn_conv(x)
        x = self.value_conv(x)
        x = attn.sigmoid() * x
        x = self.act(self.norm(x))
        x = self.proj(x)
        return x


class NNSelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, w*h, c)
        x, _ = self.multihead_attn(x, x, x)
        x = x.view(b, w, h, c).permute(0, 3, 1, 2)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_head, sr_ratio=1, mlp_ratio=2):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.attn = ConvAttention(dim, attn_dim=(int(dim*mlp_ratio)), kernel_size=3)
        else:
            self.attn_norm = LayerNormChannel(dim)
            self.attn = NNSelfAttn(dim, num_heads=num_head)

        self.mlp_norm = LayerNormChannel(dim)
        self.mlp = FFN(in_features=dim, hidden_features=int(dim*mlp_ratio))
    
    def forward(self, x):
        if self.sr_ratio > 1:
            x = x + self.attn(x)
        else:
            x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class PatchStrideConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.layer = nn.Linear(kernel_size**2 * in_dim, out_dim)
        self.act = nn.GELU()
        self.layer2 = nn.Linear(out_dim, out_dim)
        self.norm = LayerNormChannel(out_dim)

    def forward(self, x):
        x = rearrange(x, 'b c (h p) (w q) -> b h w (c p q)', p=self.kernel_size, q=self.kernel_size)
        x = self.layer(x)
        x = self.act(x)
        x = self.layer2(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.norm(x)
        return x


class NewEncoder(nn.Module): # double patch stride
    def __init__(self, in_channels=3, dims=[64, 128, 320, 512], strides=[4, 2, 2, 2], num_heads=[1, 2, 4, 8], sr_ratios=[4, 2, 1, 1], depth=[2, 2, 2, 2], mlp_ratios=[2, 2, 2, 2]):
        super().__init__()
        self.stages_len = len(strides)
        self.patch_stride_stages_raw = []
        self.normal_stride_stages = []
        self.transform_stages = []
        self.pw_conv_stages = []

        stride_accumulate = 1
        for i in range(len(strides)):
            stride_accumulate *= strides[i]
            self.patch_stride_stages_raw.append(PatchStrideConv(in_channels, dims[i], stride_accumulate))
            self.transform_stages.append(nn.ModuleList([
                AttentionBlock(dims[i], num_heads[i], sr_ratios[i], mlp_ratios[i]) for _ in range(depth[i])
            ]))
            if i > 0:
                self.normal_stride_stages.append(PatchStrideConv(dims[i-1], dims[i], strides[i]))
                self.pw_conv_stages.append(LayerNormChannel(dims[i]))
        
        self.patch_stride_stages_raw = nn.ModuleList(self.patch_stride_stages_raw)
        self.normal_stride_stages = nn.ModuleList(self.normal_stride_stages)
        self.pw_conv_stages = nn.ModuleList(self.pw_conv_stages)
        self.transform_stages = nn.ModuleList(self.transform_stages)

    def forward(self, x):
        ans = []
        x_in = x
        for i in range(self.stages_len):
            if i == 0:
                x = self.patch_stride_stages_raw[i](x_in)
            else:
                x0 = self.patch_stride_stages_raw[i](x_in)
                x2 = self.normal_stride_stages[i-1](x)
                x = self.pw_conv_stages[i-1](x0 + x2)
            for j in range(len(self.transform_stages[i])):
                x = self.transform_stages[i][j](x)
            ans.append(x)
        return ans


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_dec(nn.Module):
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

        self.task_query = nn.Parameter(torch.randn(1,48,dim))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        task_q = self.task_query
        if B>1:
            task_q = task_q.unsqueeze(0).repeat(B,1,1,1)
            task_q = task_q.squeeze(1)
        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q,size= (v.shape[2],v.shape[3]))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block_dec(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_dec(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DecoderTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size//16, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[3])

        # transformer decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block_dec(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[3])
        cur += depths[0]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    
    def forward_features(self, x):
        x=x[3]
        B = x.shape[0]
        outs = []
        
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Tdec(DecoderTransformer):
    def __init__(self, **kwargs):
        super(Tdec, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()
        assert type(kernel_size) == tuple or type(kernel_size) == int
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class DeformableConvLayer4(nn.Module):
    def __init__(self, 
                dim, 
                kernel_size=3,
                stride=1,
                mlp_ratio=4., 
                drop=0., 
                layer_scale_init_value=1e-5, 
                ):
        super().__init__()
        self.deformconv1 = DeformableConv2d(
                                        dim,
                                        int(mlp_ratio*dim),
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int((kernel_size-1)/2),
                                        bias=False
                                    )
        self.norm1 = LayerNormChannel(int(mlp_ratio*dim))
        self.act1 = nn.GELU()

        self.deformconv2 = DeformableConv2d(
                                        int(mlp_ratio*dim),
                                        dim,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int((kernel_size-1)/2),
                                        bias=False
                                    )
        self.norm2 = LayerNormChannel(dim)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.deformconv1(x)
        x = self.act1(self.norm1(x))
        
        x = self.deformconv2(x)
        x = self.act2(self.norm2(x))
        return x


class DeformableConvBlock4(nn.Module):
    def __init__(self, 
                dim, 
                mlp_ratio=4., 
                drop=0., 
                drop_path=0., 
                layer_scale_init_value=1e-5, 
                ):

        super().__init__()
        self.layer = DeformableConvLayer4(
                                            dim=dim,
                                            kernel_size=7,
                                            stride=1,
                                            mlp_ratio=1,
                                            drop=drop,
                                        )

        self.layer_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        x = x + self.layer_drop_path(self.layer(x))
        return x
    
    def forward_everything(self, x):
        res, offset1, modulator1, offset2, modulator2 = self.layer.forward_everything(x)
        x = x + self.layer_drop_path(res)
        return x, offset1, modulator1, offset2, modulator2


class convprojection(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection,self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=5, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=5, stride=2)
        self.dense_4 = nn.Sequential(DeformableConvBlock4(320, mlp_ratio=4))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=5, stride=2)
        self.dense_3 = nn.Sequential(DeformableConvBlock4(128, mlp_ratio=4))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=5, stride=2)
        self.dense_2 = nn.Sequential(DeformableConvBlock4(64 , mlp_ratio=4))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=5, stride=2)
        self.dense_1 = nn.Sequential(DeformableConvBlock4(16, mlp_ratio=4))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=5, stride=2)
        self.conv_output = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()        

    def forward(self,x1,x2):
        res32x = self.convd32x(x2[0])
        if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,-1,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)
        elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
            p2d = (0,-1,0,0)
            res32x = F.pad(res32x,p2d,"constant",0)
        elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,0,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)
        res16x = res32x + x1[3]
        res16x = self.convd16x(res16x) 
        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,-1,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0,-1,0,0)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,0,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x) 
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)
        return x


class DeformDeweatherNet(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(DeformDeweatherNet, self).__init__()

        self.Tenc = NewEncoder()
        
        self.Tdec = Tdec()
        
        self.convtail = convprojection()

        self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        if path is not None:
            self.load_pretrained(path)

    def forward(self, x):
        x1 = self.Tenc(x)
        
        x2 = self.Tdec(x1)

        x = self.convtail(x1,x2)

        pred = self.clean(x)

        return pred

    
    def load_pretrained(self, path):
        """
        Load checkpoint.
        """
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model'])
            print("======> Load pretrain model '{}'".format(path))
        else:
            print("======> No pretrain model found.")