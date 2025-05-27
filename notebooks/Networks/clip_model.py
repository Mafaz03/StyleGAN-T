import torch
import torch.nn as nn
from torchvision.transforms import Normalize
import torch.nn.functional as F
import open_clip
from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from helper import count_parameters


# class CLIP_model(nn.Module):
#     def __init__(self, name='ViT-L/14', pretrained='openai'):
#         super().__init__()
#         self.model = open_clip.create_model(name, pretrained=pretrained)
#         self.model = self.model.eval().requires_grad_(False)
#         self.img_resolution = self.model.visual.image_size[0]
#         self.norm = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
#         self.im_dim = self.txt_dim = self.model.ln_final.normalized_shape[0]

#     @property
#     def device(self) -> torch.device:
#         return next(self.model.parameters()).device

#     def encode_image(self, images: torch.Tensor, div255: bool = False) -> torch.Tensor:
#         if div255:
#             images = images.to(torch.float32) / 255.
#         images = F.interpolate(images, self.img_resolution, mode='bicubic', align_corners=False)
#         images = self.norm(images)
#         image_features = self.model.encode_image(images)
#         image_features = F.normalize(image_features, dim=-1)
#         return image_features

#     def encode_texts(self, texts: list[str]) -> torch.Tensor:
#         text = open_clip.tokenize(texts).to(self.device)
#         text_features = self.model.encode_text(text)
#         text_features = F.normalize(text_features, dim=-1)
#         return text_features

#     def forward(self, images: torch.Tensor, texts: list[str], div255: bool = False) -> torch.Tensor:
#         assert len(images) == len(texts)
#         image_features = self.encode_image(images, div255=div255)
#         text_features = self.encode_text(texts)
#         joint_features = torch.cat([image_features, text_features], 1)
#         return joint_features


class CLIP_model(nn.Module):
    def __init__(self, name='ViT-L/14', pretrained='openai'):
        super().__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.clip_model = self.clip_model.eval().requires_grad_(False)
        self.img_resolution = self.clip_model.visual.image_size[0]
        self.norm = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        self.im_dim = self.txt_dim = self.clip_model.ln_final.normalized_shape[0]

    @property
    def device(self) -> torch.device:
        return next(self.clip_model.parameters()).device

    def encode_image(self, images: torch.Tensor, div255: bool = False) -> torch.Tensor:
        if div255:
            images = images.to(torch.float32) / 255.
        images = F.interpolate(images, self.img_resolution, mode='bicubic', align_corners=False)
        images = self.norm(images)
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        text = open_clip.tokenize(texts).to(self.device)
        text_features = self.clip_model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(self, images: torch.Tensor, texts: list[str], div255: bool = False) -> torch.Tensor:
        assert len(images) == len(texts)
        image_features = self.encode_image(images, div255=div255)
        text_features = self.encode_text(texts)
        joint_features = torch.cat([image_features, text_features], 1)
        return joint_features