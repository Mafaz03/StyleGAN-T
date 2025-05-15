import torch
from torch import nn
from helper import supress_tracer_warnings, assert_shape, is_list_of_str, normalise_2nd_moment, count_parameters

from typing import Optional, Any, List, Union
import numpy as np
import matplotlib.pyplot as plt

from shared import FullyConnectedLayers, MLP
from clip import CLIP

import sys
import os

cur_path = ''.join(os.getcwd().split('/')[:-1])
sys.path.insert(0, f'{cur_path}/torch_utils/ops')
sys.path.insert(0, f'{cur_path}/torch_utils')

# TODO: code this from scrtch
import conv2d_resample
import upfirdn2d
import bias_act
import fma


def modulated_conv2d(x: torch.Tensor, weight: torch.Tensor, styles: torch.Tensor, noise: Optional[torch.Tensor] = None,
                     up: int = 1, down: int = 1, padding: int = 0, resample_filter: Optional[List[int]] = None, 
                     demodulate: bool = True, flip_weight: bool = True, fused_mod_cov: bool = True) -> torch.Tensor:
    
    # x:      [B, inC, H, W]
    # weight: [outC, inC, kh, kw]
    # styles: [B, inC]

    batch_size = x.shape[0]
    out_channels, in_channels, kw, kh = weight.shape
    assert_shape(weight, [out_channels, in_channels, kw, kh])
    assert_shape(x, [batch_size, in_channels, None, None])    # x's & weight's batch_size and In channels must remain same
    assert_shape(styles, [batch_size, in_channels])

    if x.dtype == torch.float16 and demodulate:
        a = 1 / np.sqrt(in_channels * kh * kw)
        b = weight.norm(p = float("inf"), dim = [1,2,3], keepdim=True)          # max of inC, kh, kw
        weight = weight * (a / b)

        styles = styles / styles.norm(p = float('inf'), dim = [1], keep_dim = True) # max of inC
    
    w = None
    dcoef = None
    if demodulate or fused_mod_cov:
        w = weight.unsqueeze(0)                                    # w:      [1, outC, inC, kh, kw]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)            # styles: [B,  1,   inC, 1,  1]
                                                                   # w:      [B, outC, inC, kh, kw]
    if demodulate:
        dcoef = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt()       # dcoef:  [B, outC]
       
    if demodulate and fused_mod_cov:
        w = w * dcoef.reshape(batch_size, -1, 1, 1, 1)             # w:      [B, outC, inC, kh, kw]
         # w_hat_i = modulated_weight_i / sqrt(sum_j(modulated_weight_{i,j} ** 2) + epsilon)

    if not fused_mod_cov:
        x = x * styles.reshape(batch_size, -1, 1, 1)               # style: [B, inC x kh x kw, C, 1]
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        x = x.reshape(batch_size, -1, *x.shape[2:])

        if demodulate and noise is not None:
            x = fma.fma(x, dcoef.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoef.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))                          # noise: [B, inC, H, W]                            
        return x
    
    # if fused_mod_cov is True
    # Execute as one fused op using grouped convolution.

    with supress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    assert_shape(x, [batch_size, in_channels, None, None])

    x = x.reshape(1, -1, *x.shape[2:])                            # x: [1, B x inC, H, W]
    w = w.reshape(-1, in_channels, kh, kw)                        # w: [B x outC, inC, kh, kw]

    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])                   # x: [B, C, H, W]
    if noise is not None:
        x = x.add_(noise)
    return x

class GroupNorm_float32(nn.GroupNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).type(x.dtype) # Converts x to float32 -> applies Group norm -> converts back to original type


class StyleSplit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        """
        •	Suppress or amplify features (gate)
        •	Scale them globally (scale)
        •	Shift them (bias)
        """
        super().__init__()

        self.fcl = FullyConnectedLayers(in_features=in_channels, out_features = 3*out_channels, **kwargs)
    
    def forward(self, x: torch.Tensor):
        x = self.fcl(x)
        m1, m2, m3 = x.chunk(chunks=3, dim=1)
        
        return m1 * m2 + m3 # scale . gate + bias
    
class SynthesisInput(nn.Module):
    """
    Latent w → SynthesisInput → Feature map [B, 64, 64, 64]

    sin(2pi (f . x + phi))
    
    Where:
	•	f = 2D frequency vector (learned)
	•	x = pixel location
	•	phi = phase offset (learned)
	•	All of this is modulated by latent vector w through an affine transformation
    """

    def __init__(self, w_dim: int, channels: int, size: int, sampling_rate: int, bandwidth: int):
        super().__init__()

        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(size, [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.rand([self.channels, 2])                      # Gausian Cloud in 2D space
        radii = freqs.square().sum(dim = 1, keepdim=True).sqrt()    # r = sqrt(x^2 + y^2)
        freqs = freqs / (radii * radii.square().exp().pow(0.25))    # freqs = freqs / r x exp(0.25 x r^2)}
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5                  # values are in range [-0.5, 0.5]

        self.weights = nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayers(in_features=w_dim, 
                                           out_features=4,
                                           weight_init = 0,
                                           bias_init=[1,0,0,0])
        
        self.register_buffer("transforms", torch.eye(3,3))
        self.register_buffer("freqs", freqs)
        self.register_buffer("phases", phases)

    def forward(self, w: torch.Tensor): 
        # w = [B, w_dim]
        transforms = self.transforms.unsqueeze(0)                                   # [batch, row, col]
        freqs = self.freqs.unsqueeze(0)                                             # [batch, channels, 2] since 2D space, xy
        phases = self.phases.unsqueeze(0)                                           # [batch, channels]

        t = self.affine(w)                                                          # t: [batch, 4]
        t = t / t[:, :2].norm(dim = 1, keepdim = True)                              # t: [batch, 4] / [5, 1] -> [batch, 4]

        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # m_r = [batch, 3, 3] Identity matrix     ROTATION
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c

        # m_r:
        # [cos(theta)  -sin(theta)  0]
        # [sin(theta)   cos(theta)  0]
        # [0            0           1]

        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # m_t = [batch, 3, 3] Identity matrix     TRANSFORM
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        # m_t:
        # [1  0  -tx]
        # [0  1  -ty]
        # [0  0    1]

        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.
        # transforms: [batch, 3, 3]

        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)
        
        theta = torch.eye(2, 3)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate

        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)  # [1, size, size, 2]
        
        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weights / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x.contiguous()

    def extra_repr(self) -> str:
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])
    
class SynthesisLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, w_dim: int, resolution: int, kernel_size: int = 3, up: int = 1,
                 use_noise: bool = True,                    # Enable noise input?
                 acivation: str = "lrelu",                  # Activation function: 'relu', 'lrelu', etc.
                 resample_filer: List[int] = [1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp: Optional[int] = None,          # Clamp the output of convolution layers to +-X, None = disable clamping.
                 channels_last: bool = False,               # Use channels_last format for the weights?
                 layer_scale_init: float = 1e-5,            # Initial value of layer scale.
                 residual: bool = False,                    # Residual convolution?
                 gn_groups: int = 32,                       # Number of groups for GroupNorm
                 ):
        super().__init__()

        if residual: assert in_channels == out_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.kernel_size = kernel_size
        self.use_noise = use_noise
        self.acivation = acivation
        self.conv_clamp = conv_clamp
        self.residual = residual
        self.up = up
        self.register_buffer("resample_filer", upfirdn2d.setup_filter(resample_filer))
        self.padding = kernel_size // 2
        self.act_gain = bias_def_gain[acivation]

        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = nn.Parameter(torch.zeros([]))

        self.affine = StyleSplit(
            in_channels = w_dim,
            out_channels = in_channels,
            bias_init = 1
        )

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        # contiguous_format |   [N, C, H, W]  |  Default, works everywhere
        # channels_last     |   [N, H, W, C]  |  Faster on GPUs with mixed-precision (AMP)

        self.weight = nn.Parameter(
            torch.randn([
                out_channels, in_channels, kernel_size, kernel_size
                ])
        ).to(memory_format = memory_format)
        
        self.bias = nn.Parameter(torch.zeros([out_channels]))

        if self.residual:
            assert up == 1
            self.norm = GroupNorm_float32(gn_groups, out_channels)
            self.gamma = nn.Parameter(layer_scale_init * torch.ones([1, out_channels, 1, 1])).to(memory_format = memory_format)    # gamma: [1, OutC, 1, 1]
            
    def forward(self, x: torch.Tensor, w: torch.Tensor, noise_mode: str = 'random', fused_mod_cov: bool = True, gain: int = 2):
        in_resolution = self.resolution // self.up
        assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])

        noise = None
        if self.use_noise and noise_mode == "random":
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution],  device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength
        
        flip_weight = (self.up == 1)  
        styles = self.affine(w)

        if self.residual: x = self.norm(x)
        y = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, fused_mod_cov=fused_mod_cov,
                            padding=self.padding, resample_filter=self.resample_filer, flip_weight=flip_weight)
        y = y.to(x.dtype)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None

        y = bias_act.bias_act(y, self.bias.to(x.dtype), act=self.acivation, gain=act_gain, clamp=act_clamp)

        if self.residual:
            y = self.gamma * y
            y = y.to(x.dtype).add_(x).mul(np.sqrt(2))
        
        return y
        
    def extra_repr(self) -> str:
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.acivation:s}'])

class ToRGBLayer(nn.Module):
    """
    Last layer that converts feature vector to rgb image: [B, 3, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int, w_dim: int, kernel_size: int = 1, conv_clamp: Optional[int] = None, channels_last: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = StyleSplit(
            in_channels = w_dim,
            out_channels = in_channels,
            bias_init = 1
        )

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = nn.Parameter(0.1 * torch.randn([self.out_channels, self.in_channels, kernel_size, kernel_size])).to(memory_format = memory_format) 
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x: torch.Tensor, w: torch.Tensor, fused_mod_cov: bool = True) -> torch.Tensor:
                                                                                          # x:      [B, Cs, H, W]
                                                                                          # w:      [B, w_dim]
        styles = self.affine(w) * self.weight_gain                                        # styles: [B, inC]
        x = modulated_conv2d(x = x, weight=self.weight, styles=styles, demodulate=False)  # x:      [B, outC, H, W]
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

class SynthesisBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, w_dim: int, resolution: int, img_channels: int,
                 is_last: bool,                                  # Is this the last block?
                 num_res_blocks: int = 3,                        # Number of conv layers per block
                 architecture: str = "orig",                     # Architecture: 'orig', 'skip'.
                 resample_filter: list[int] = [1,3,3,1],         # Low-pass filter to apply when resampling activations.
                 conv_clamp: int = 256,                          # Clamp the output of convolution layers to +-X, None = disable clamping.
                 use_fp16: bool = False,                         # Use FP16 for this block?
                 fp16_channels_last: bool = False,               # Use channels-last memory format with FP16?
                 fused_modconv_default: Any = 'inference_only',  # Default value of fused_modconv.
                 **layer_kwargs,                                 # Arguments for SynthesisLayer.
                 ):
        super().__init__()

        assert architecture in ['orig', 'skip'], f"{architecture} not defined."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.fused_modconv_default = fused_modconv_default
        
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.input = SynthesisInput(w_dim = self.w_dim, channels = self.out_channels, size = resolution, sampling_rate = resolution, bandwidth = 2)
            # input: [B, outC, resolution, resolution])
            self.num_conv += 1
        
        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels = in_channels, out_channels = out_channels, w_dim = self.w_dim, resolution = resolution,
                                        up = 2, resample_filer = resample_filter, conv_clamp = conv_clamp, channels_last = self.channels_last, **layer_kwargs)
            # conv0: [B, outC, resolution, resolution]
            self.num_conv += 1
        
        convs = []
        for _ in range(num_res_blocks):
            convs.append(
                SynthesisLayer(in_channels = out_channels, out_channels = out_channels, resolution = resolution, conv_clamp = conv_clamp, 
                               channels_last = self.channels_last, w_dim = w_dim, **layer_kwargs)
            )
            convs.append(
                SynthesisLayer(in_channels = out_channels, out_channels = out_channels, resolution = resolution, conv_clamp = conv_clamp, 
                               channels_last = self.channels_last, w_dim = w_dim, **layer_kwargs, residual = True)
            )
        
        self.convs1 = nn.ModuleList(convs)
        self.num_conv += len(convs)

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1
    
    def forward(self, x: torch.Tensor, img: Optional[torch.Tensor], ws: torch.Tensor, force_fp32: bool = False, fused_mod_cov: bool = True, **layer_kwargs):
        w_iter = iter(ws.unbind(dim=1)) # ws: [B, num w, w_dim]
        if ws.device.type != "cuda": 
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32                                # use float16 is 'cpu' and use_fp16 is True
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format   # Channel last if 'cpu' and channel_last
        
        if fused_mod_cov is None:
            fused_mod_cov = self.fused_modconv_default
        if fused_mod_cov == "inference_only":
            fused_mod_cov = (not self.training)
        
        # Input
        if self.in_channels == 0:
            x = self.input(next(w_iter)) # x: [B, w_dim]
        else:
            assert_shape(x, [None, self.in_channels, self.resolution//2, self.resolution//2])
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main Layers
        if self.in_channels == 0:
            for conv in self.convs1:
                x = conv(x, next(w_iter), fused_mod_cov = fused_mod_cov, gain = np.sqrt(0.5), **layer_kwargs)    # x: [B, outC, resolution, resolution]
        else:
            x = self.conv0(x, next(w_iter), fused_mod_cov = fused_mod_cov, **layer_kwargs)                       # x: [B, outC, resolution, resolution]
            for conv in self.convs1:
                x = conv(x, next(w_iter), fused_mod_cov = fused_mod_cov, gain=np.sqrt(0.5), **layer_kwargs)
        # x: [B, outC, resolution // 2, resolution // 2]
    
        # ToRGB
        if img is not None:
            assert_shape(img, [None, self.img_channels, self.resolution//2, self.resolution//2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)      # img: [B, imgC, resolution, resolution] Doubles the H and W
        
        if self.is_last or self.architecture == "skip":
            y = self.torgb(x, next(w_iter), fused_mod_cov = fused_mod_cov)        # y: [B, 3, resolution, resolution]
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self) -> str:
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

class SynthesisNetwork(torch.nn.Module):
    """
    style vectors (ws)
            ↓
    block 1 (make 8x8 base)
            ↓
    block 2 (upsample to 16x16, add details)
            ↓
    block 3 (upsample to 32x32, add more)
            ↓
    ...
    block N (final resolution)
            ↓
    ToRGB → Final Image [B, 3, img_resolution, img_resolution]
    """
    def __init__(
        self,
        w_dim: int,                 # Intermediate latent (W) dimensionality.
        img_resolution: int,        # Output image resolution.
        img_channels: int = 3,      # Number of color channels.
        channel_base: int = 32768,  # Overall multiplier for the number of channels.
        channel_max: int = 512,     # Maximum number of channels in any layer.
        num_fp16_res: int = 4,      # Use FP16 for the N highest resolutions.
        base_mult: int = 3,         # Start resolution (SG2: 2, SG3: 4, SG-T: 3).
        num_res_blocks: int = 3,    # Number of residual blocks.
        verbose: bool = False,
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
    
        super().__init__()

        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0 # A bitwise trick to check if img_resolution is a power of 2

        self.verbose = verbose
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res

        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.block_resolutions = [2 ** i for i in range(base_mult, self.img_resolution_log2 + 1)]        # [8, 16, 32, 64, 128....]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}   # {8: 512, 16: 512, 32: 512, 64: 512, 128: 256...}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)                     # Ensures the minimum value of fp16_resolution is 8.
        
        self.num_ws = 0
        blocks = []
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > self.block_resolutions[0] else 0 # 0, 512, 512, 512, 512 ...
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, 
                                   w_dim = w_dim, 
                                   resolution = res, 
                                   img_channels = img_channels, 
                                   is_last = is_last, 
                                   use_fp16 = use_fp16, 
                                   num_res_blocks = num_res_blocks, 
                                   **block_kwargs)
            
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb

            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, ws: torch.Tensor, **block_kwargs) -> torch.Tensor:
        # ws: [B, num_ws, w_dim]
        ws = ws.to(torch.float32)
        block_ws = []
        w_idx = 0
        for block in self.blocks:
            a = ws.narrow(dim = 1, start = w_idx, length = block.num_conv + block.num_torgb) # a: [B, num_conv + num_torgb, w_dim]
            block_ws.append(a)
            w_idx += block.num_conv
        
        x = img = None
        for res, cur_ws, block in zip(self.block_resolutions, block_ws, self.blocks):
            x, img = block(x, img, cur_ws, **block_kwargs)
            if self.verbose: print(x.shape)
        
        return img 
    
    def extra_repr(self) -> str:
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])
    
class MappingNetwork(nn.Module):
    def __init__(self, 
                 z_dim: int,                   # Latent space
                 conditional: bool,            # Text Conditional?
                 num_layers: int = 2,          # Number of mapping layers.
                 activation: str = "lrelu",
                 lr_multiplier: float = 0.01,
                 x_avg_beta: float = 0.995,    # Decay for tracking the moving average of W during training.
                 ):
        super().__init__()

        self.z_dim = z_dim
        self.x_avg_beta = x_avg_beta
        self.num_ws = None

        self.mlp = MLP(feature_list = [z_dim]*(num_layers-1), # FIX
                       activation = activation,
                       lr_multiplier=lr_multiplier, 
                       linear_out=True)
        if conditional:
            self.clip = CLIP()
            del self.clip.model.visual # only using the text encoder
            self.c_dim = self.clip.txt_dim
        else:
            self.cdim = 0
        
        self.w_dim = self.c_dim + self.z_dim
        self.register_buffer('x_avg', torch.zeros([self.z_dim]))

    def forward(self, z: torch.Tensor, c: Union[None, torch.Tensor, list[str]], truncation_psi: float = 1.0) -> torch.Tensor:
        # z: [B, z_dim]
        assert_shape(z, [None, self.z_dim])

        x = self.mlp(normalise_2nd_moment(z))               # z: [B, [z_dim]*num_layers-1]; Doesnt change shape after mlp layer

        if self.x_avg_beta is not None and self.training:
            a = x.detach().mean(0)                          # a:     [z_dim]
            a = a.lerp(self.x_avg, self.x_avg_beta)         # a:     [z_dim]; Does a linear interpolation of two tensors
            self.x_avg.copy_(a)                             # x_avg: [z_dim]
        
            # lerp: start(i) + [weight(i) x (end(i) - start(i))]
        if truncation_psi != 1:
            assert self.x_avg_beta is not None
            x = self.x_avg_beta.lerp(x, truncation_psi)    # x_avg: [B, z_dim]; Doesnt change

        # Build latent
        if len(c) >- 1:
            assert c is not None
            c = self.clip.encode_text(c) if is_list_of_str(c) else c
            w = torch.cat([x, c], dim=1)                  # w: [B, z_dim + 768]
        else:
            w = x
        
        # Broadcast latent codes
        if self.num_ws is not None:
            w = w.unsqueeze(1)                           # w: [B, 1,      z_dim + 768]
            w = w.repeat([1, self.num_ws, 1])            # w: [B, num_ws, z_dim + 768]
        
        return w
    
class Generator(nn.Module):
    def __init__(self, z_dim: int, conditional: bool, img_resolution: int, img_channels: int = 3, 
                 train_mode: str = "all", # Control which layers are trainable
                 synthesis_kwargs: dict = {},
                 ):
        super().__init__()
        
        self.z_dim = z_dim
        self.conditional = conditional
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.mapping = MappingNetwork(z_dim = z_dim, conditional = conditional)
        self.synthesis = SynthesisNetwork(w_dim=self.mapping.w_dim, img_resolution=img_resolution,
                                          img_channels=img_channels, **synthesis_kwargs)
        
        self.w_dim = self.synthesis.w_dim
        self.c_dim = self.mapping.c_dim
        self.num_ws = self.synthesis.num_ws
        self.mapping.num_ws = self.num_ws
        
        # Set trainable layers.
        self.train_mode = train_mode
        if train_mode == 'all':
            self.trainable_layers = ['synthesis', 'mapping.mlp']
        elif train_mode == 'text_encoder':
            self.trainable_layers = ['clip']
        elif train_mode == 'freeze64':
            self.trainable_layers = [f"b{x}" for x in self.synthesis.block_resolutions if x > 64]
            self.trainable_layers += ['torgb']
        
    def forward(self, 
                z: torch.Tensor, 
                c: Union[None, torch.Tensor, List[str]],
                truncation_psi: float = 1.0, 
                **synthesis_kwargs
                ):
        
        ws = self.mapping(z, c, truncation_psi=truncation_psi)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img


