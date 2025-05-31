import types
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

class Readout(nn.Module):
    """
    Adds CLS and/or DIST Tokens to the patches
    """
    def __init__(self, start_index = 1):
        super().__init__()

        self.start_index = start_index
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X [B, N, C]
        # Every local patch token gains global context
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2 # (CLS + DIST) / 2
        else:
            readout = x[:, 0]                 # CLS
        
        # readout: [B,             C]
        # x      : [B, N - 2 or 1, C]
        return x[: , self.start_index: ] + readout.unsqueeze(1)

class Transpose(nn.Module):
    """
    from [B, N, C] to [B, C, N]
    """
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim1, self.dim2).contiguous()

def _resize_pos_embed(self, posemb: torch.Tensor, gs_h: int, gs_w: int) -> torch.Tensor:
    posemb_tok = posemb[:, : self.start_index]                    # posemb_tok:  [B,          1 or 2, C]
    posemb_grid = posemb[0, self.start_index :]                   # posemb_grid: [N - 1 or 2, C]

    gs_old = int(math.sqrt(len(posemb_grid)))                   # gs_old, 14 or 16 .....
    
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1)    # posemb_grid: [1, 14, 14, C]
    posemb_grid = posemb_grid.permute(0, 3, 1, 2)               # posemb_grid: [1, C, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), 
                                mode="bilinear", 
                                align_corners=False)            # posemb_grid: [1, C, new_gs_w, new_gs_w]
    
    posemb_grid = posemb_grid.permute(0, 2, 3, 1)               # posemb_grid: [1, new_gs_w, new_gs_w, C]
    posemb_grid = posemb_grid.reshape(1, gs_h * gs_w, -1)       # posemb_grid: [1, new_gs_w x new_gs_w, C]
    posemb_grid = posemb_grid.expand(posemb.shape[0], -1, -1)   # posemb_grid: [B, new_gs_w x new_gs_w, C]    # FIX
    posemb = torch.cat([posemb_tok, posemb_grid], dim = 1)      # posemb_grid: [B, new_gs_w x new_gs_w + 1 or 2, C] added Batch dim back + 1 or 2 CLS token
    return posemb


def forward_flex(self, x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    x = self.patch_embed.proj(x)             # x: [B, embedding dimension, H // 16   W // 16]
    x = x.flatten(2)                         # x: [B, embedding dimension, H // 16 x W // 16]
    x = x.transpose(1,2)                     # x: [B, H // 16 x W // 16, embedding dimension]
    # x: [B, patch_dim, embedding dimension]
    # x: [B, N,         C]

    pos_embed = _resize_pos_embed(self = self, posemb = self.pos_embed, gs_h = H // self.patch_size[0], gs_w = W // self.patch_size[1])

    # # Adding CLS Tokens
    cls_tokens = self.cls_token             # cls_tokens: [1, 1, embedding dimension]
    cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)  # cls_tokens: [B, 1, embedding dimension]

    x = torch.cat([cls_tokens, x], dim=1)               # x:   [B, N + 1, C]

    # assert x.shape == pos_embed.shape, f"x shape: {x.shape}, pos_embed: {pos_embed.shape}"
    x = x + pos_embed
    x = self.pos_drop(x)                # x:   [B, N + 1, C]
    for blk in self.blocks:
        x = blk(x)                      # x:   [B, N + 1, C]
    x = self.norm(x)                    # x:   [B, N + 1, C]
    return x


activations = {}

def get_activation(name: str) -> Callable:
    def hook(model, inputs, outputs):
        activations[name] = outputs
    return hook

def make_vit_backbone(model: nn.Module, 
                      patch_size = [16, 16],
                      hooks = [2, 5, 8, 11],
                      hook_patch = True,
                      start_index = 1):
    assert len(hooks) == 4

    pretrained = nn.Module
    pretrained.model = model

    for i in range(len(hooks)):

        pretrained.model.blocks[hooks[i]].register_forward_hook(get_activation(f'{i}')) # Get shape of 2nd, 5th, 6th... Block of ViT
    if hook_patch: pretrained.model.pos_drop.register_forward_hook(get_activation('4'))

    pretrained.rearrange = nn.Sequential(
                                        Readout(start_index=start_index), 
                                        Transpose(1, 2)                  # [B, C, N]
                                        )
    
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = patch_size

    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)

    return pretrained
    
def forward_vit(pretrained: nn.Module, x: torch.Tensor) -> dict:
    _ = pretrained.model.forward_flex(x)
    return {k: pretrained.rearrange(v) for k, v in activations.items()}