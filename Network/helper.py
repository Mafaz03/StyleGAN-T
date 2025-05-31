import contextlib
import torch
import warnings
from typing import Union, Any, Optional, List, Iterator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from torch import nn

def show_one(x: torch.Tensor, title = None):
    if x.ndim == 4:
        x = x[0]  # first batch
    assert x.shape[0] <= 4
    if title: plt.title(title)
    plt.imshow(x.permute(1,2,0).detach().cpu().numpy())

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

@contextlib.contextmanager
def supress_tracer_warnings():
    flt  = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape): raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')

    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None: pass
        elif isinstance(ref_size, torch.Tensor):
            with supress_tracer_warnings():
                torch._assert(
                    torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}'
                )
        elif isinstance(size, torch.Tensor):
            with supress_tracer_warnings():
                torch.equal(
                    torch._assert(size, torch.as_tensor(ref_shape)), f'Wrong size for dimension {idx}'
                )
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

def is_list_of_str(arr: Any):
    if arr is None: return None
    is_list = isinstance(arr, list) or isinstance(arr, np.ndarray) or isinstance(arr, tuple)
    is_str = isinstance(arr[0], str)
    return is_list and is_str


def normalise_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    a = x.square().mean(dim=dim, keepdim=True) + eps
    a = a.rsqrt()
    a = x * a
    return x

def save_image_grid(dl, name = "sample.png", num_grid = 2, px = 600, has_labels = True):
    if has_labels:
        images, labels = next(iter(dl))
    else: images = next(iter(dl))

    fig = plt.figure(figsize=(px/100, px/100))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(num_grid, num_grid),
                    axes_pad=0.3,
                    )
    if has_labels:
        for ax, img, label in zip(grid, images, labels):
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()

            ax.imshow(img_np)
            ax.axis("off")
            ax.set_title(label)
    else:
        for ax, img in zip(grid, images):
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()

            ax.imshow(img_np)
            ax.axis("off")

    for ax in grid[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(name)

def split(arr: Union[List, np.ndarray, torch.Tensor], chunk_size: int, dim: int = 0):
    splits = (len(arr) / chunk_size)
    return np.split(arr, splits, dim)

def fetch_data(training_set_iterator: Iterator, batch_size: int, z_dim: int, device: torch.device):
    real_images, real_labels = next(training_set_iterator)
    real_images = (real_images/(255/2)) - 1           # normalizing images from [-1, 1]
    gen_zs = torch.randn([batch_size, z_dim], device=device)
    
    return real_images, real_labels, gen_zs


def partial_freeze(gen_or_disc: nn.Module) -> None:
    phase = gen_or_disc.name

    if phase == "G":

        trainable_layers = gen_or_disc.trainable_layers
        # Freeze all layers first
        gen_or_disc.requires_grad_(False)

        # Then selectively unfreeze based on substring match
        for name, layer in gen_or_disc.named_modules():
            should_train = any(layer_type in name for layer_type in trainable_layers)
            layer.requires_grad_(should_train)
    
    elif phase == "D":
        gen_or_disc.dino.requires_grad_(False)
    
    else: raise NotImplemented
    
