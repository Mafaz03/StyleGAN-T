import torch
from torch import nn
from typing import Callable
import numpy as np

import sys
sys.path.insert(0, '/Users/mohamedmafaz/Desktop/StyleGAN-T/notebooks/torch_utils/ops')
sys.path.insert(0, '/Users/mohamedmafaz/Desktop/StyleGAN-T/notebooks/torch_utils')
import bias_act

class ResidualBlock(nn.Module):
    def __init__(self, layer: Callable):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.layer(x)) / np.sqrt(2)

class FullyConnectedLayers(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: str = 'linear',
                       lr_multiplier: float = 1.0, weight_init: float = 1.0, bias_init: float = 0.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.weights = nn.Parameter(
            torch.randn([out_features, in_features])
            ) * (weight_init / lr_multiplier)
        
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), shape=[out_features])
        self.bias = nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(x.dtype) * self.weight_gain
        b = self.bias

        if b is not None:
            if self.bias_gain != 1: b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())  # b + x @ Wᵀ
            # x: [batch_size, in_features]
	        # W: [out_features, in_features]
        else:
            x = torch.matmul(x, w.t())
            bias_act.bias_act(x = x, b = b, act = self.activation)
        
        return x
    
    def extra_repr(self):
        return f"In Features: {self.in_features}\nOut Features: {self.out_features}\nActivation Function: {self.activation}"


class MLP(nn.Module):
    def __init__(self, feature_list: list[int], activation: str = 'linear', lr_multiplier: float = 1.0, linear_out: bool = False):
        super().__init__()

        self.num_layers = len(feature_list) - 1
        self.out_dim = feature_list[-1]

        self.layers = nn.ModuleList()

        for idx in range(self.num_layers):
            in_features = feature_list[idx]
            out_features = feature_list[idx+1]
            if linear_out and idx == self.num_layers-1:
                activation = 'linear'
            layer = FullyConnectedLayers(in_features=in_features, out_features=out_features, activation=activation, lr_multiplier=lr_multiplier)
            # print(layer)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift2Batch = (x.ndim == 3)

        if shift2Batch:
            B, K, L = x.shape
            x = x.flatten(0,1) # B, K, L -> B x K, L
        
        for layer in self.layers:
            x = layer(x)
        
        if shift2Batch:
            x = x.reshape(B, K, -1)
        
        return x
    