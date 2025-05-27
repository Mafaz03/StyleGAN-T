import torch
from torch import nn
import numpy as np

from typing import Dict

from torch.nn.utils.spectral_norm import SpectralNorm

from helper import assert_shape
from shared import FullyConnectedLayers, ResidualBlock
from vit_utils import make_vit_backbone, forward_vit
from diff_aug import DiffAugment

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import RandomCrop, Normalize

class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name = "weight", n_power_iterations=1, dim = 0, eps = 1e-12)

class LocalBatchNorm(nn.Module):
    # When using large batch sizes, the variance across the batch can be very high, especially in early training.
    # It may cause the normalization to overreact, resulting in instability in the discriminator’s learning.
    # So we use virtual_bs for smaller batch size to normalize it through.
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-8):
        super().__init__()

        self.num_features = num_features
        self.affine = affine             # learn weight and biases?
        self.virtual_bs = virtual_bs
        self.eps = eps

        if self.affine:
            self.weights = nn.Parameter(torch.ones(num_features))
            self.bias    = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape

        G = np.ceil(x.shape[0] / self.virtual_bs).astype(int) # G = B / 8 
        x = x.view(G, -1, x.shape[1], x.shape[2])             # x: [G, -1, N, C]
        # Normalizing per group, per channel
        mean = x.mean([1, 3], keepdim=True)                   # mean: [G, 1, N, 1] 
        var = x.var([1, 3], keepdim=True)                     # var : [G, 1, N, 1] 

        x = (x - mean) / (torch.sqrt(var) + self.eps)            # x: [G, -1, N, C]

        if self.affine:
            x = x * self.weights[None, :, None]               # weight: [1, N, 1]
                                                              # x     : [G, -1, N, C]

            x = x + self.bias[None, :, None]                  # bias  : [1, N, 1]
                                                              # x     : [G, -1, N, C]
        return x.view(shape)


def make_block(channels: int, kernel_size: int):
    return nn.Sequential(
        SpectralConv1d(in_channels  = channels, 
                       out_channels = channels, 
                       kernel_size  = kernel_size, 
                       padding      = kernel_size//2, 
                       padding_mode = "circular"),
        
        LocalBatchNorm(num_features = channels),
        nn.LeakyReLU(0.2, True)
    )

class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()

        self.channels = channels # DINO ViT-S output
        self.c_dim = c_dim       # Text embedding from CLIP
        self.cmap_dim = cmap_dim # Projection space for conditional score

        self.main = nn.Sequential(
            make_block(channels = channels, kernel_size = 1),
            ResidualBlock(make_block(channels = channels, kernel_size = 9))
        )   # x shape will remain same as long as kernel is odd

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayers(in_features = c_dim, out_features = cmap_dim)
            self.cls = SpectralConv1d(in_channels = channels, out_channels = cmap_dim, kernel_size = 1, padding = 0)
        else:
            self.cls = SpectralConv1d(in_channels = channels, out_channels = 1, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, channels, dim]
        # c: [B, c_dim]

        h = self.main(x)                         # h  : [B, channels, dim]
        out = self.cls(h)                        # out: [B, cmap_dim, dim]
                                            #         or
                                            # out: [B, 1,        dim]
        if self.c_dim > 0:
            cmap = self.cmapper(c)                   # cmap: [B, cmap_dim]
            cmap = cmap.unsqueeze(-1)           # cmap: [B, cmap_dim, 1]
            out = out * cmap                    # out:  [B, cmap_dim, cmap_dim]
            out = out.sum(1, keepdim=True)      # out:  [B, 1,        cmap_dim]
            out = out * np.sqrt(1 / self.cmap_dim)   # out:  [B, 1,        cmap_dim]
        
        return out

class DINO(nn.Module):
    def __init__(self, hooks: int = [2, 5, 8, 11], hook_patch: bool = True):
        super().__init__()
        
        self.n_hooks = len(hooks) + int(hook_patch) # n_hooks: 5

        self.model = make_vit_backbone(
                    timm.create_model('vit_small_patch16_224_dino', pretrained=True),
                    patch_size=[16,16], hooks=hooks, hook_patch=hook_patch,
                )
        self.model = self.model.model.eval().requires_grad_(False)
        self.img_res = self.model.model.patch_embed.img_size[0]                 # img_res  : 224
        self.embed_dim = self.model.model.embed_dim                             # embed_dim: 384
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    
    def forward(self, x: torch.Tensor) -> Dict: # FIX
        x = nn.functional.interpolate(x, self.img_res, mode='area')  # x: [B, channel, 224, 224]
        x = self.norm(x)
        features = forward_vit(self.model, x)
        return features
    
class ProjectedDiscriminator(nn.Module):
    def __init__(self, c_dim: int, diff_aug: bool = True, p_crop: bool = True):
        super().__init__()

        self.c_dim = c_dim
        self.p_crop = p_crop
        self.diff_aug = diff_aug

        self.dino = DINO()

        heads = []

        for i in range(self.dino.n_hooks):
            heads += [
                str(i), DiscHead(channels = self.dino.embed_dim,
                                c_dim     = c_dim,
                                cmap_dim  = 64)
                                ],

        self.heads = nn.ModuleDict(heads)

    def train(self, mode: bool = True):
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        
        if self.diff_aug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # Transform to [0, 1].
        x = x.add(1).div(2)

        # Take crops with probablity p_crop if the image is larger.
        if x.size(-1) > self.dino.img_res and np.random.random() < self.p_crop:
            x = RandomCrop(self.dino.img_resolution)(x)

        # Forward pass through DINO ViT.
        features = self.dino(x)

        # Apply discriminator heads.
        logits = []
        for k, head in self.heads.items():
            logit = head(features[k], c).squeeze(1)
            logits.append(logit)

        return torch.cat(logits, dim = 1)

