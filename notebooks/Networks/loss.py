from clip_model import CLIP_model
from Generator import Generator
from Discriminator import ProjectedDiscriminator

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomCrop
# from torch_utils import training_stats

from helper import show_one

import sys
import os

from typing import List

cur_path = '/'.join(os.getcwd().split('/')[:-1])
sys.path.insert(0, f'{cur_path}/torch_utils/ops')
sys.path.insert(0, f'{cur_path}/torch_utils')

import conv2d_resample
import upfirdn2d
import bias_act
import fma



class ProjectedGANLoss:
    def __init__(
        self,
        device: torch.device,
        G: nn.Module,
        D: nn.Module,
        blur_init_sigma: int = 2,
        blur_fade_kimg: int = 0,
        clip_weight: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_curr_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.train_text_encoder = 'clip' in G.trainable_layers
        self.clip = CLIP_model().eval().to(self.device).requires_grad_(False)
        self.clip_weight = clip_weight

        @staticmethod
        def spherical_distance(x: torch.Tensor, y: torch.Tensor):
            x = F.normalize(x, dim = -1)
            y = F.normalize(y, dim = -1)

            # Smaller angle -> more similar
            # Larger angle  -> more dissimilar
            return (x * y).sum(-1).arccos().pow(2)
    
        @staticmethod
        def set_blur_sigma(self, cur_nimg: int):
            # cur_nimg is basically num images sees
            if self.blur_fade_kimg > 1:
                self.blur_curr_sigma = max(1 - cur_nimg / (self.blur_fade_kimg  * 1000), 0) * self.blur_init_sigma
            else: 
                self.blur_curr_sigma = 0
            return self.blur_curr_sigma
        
        def blur(img: torch.Tensor, blur_sigma: float) -> torch.Tensor:
            # Applies Blur
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=img.device, dtype = torch.float32) # e.g., [-3, -2, ..., 3]
                f = f.div(blur_sigma).square().neg().exp2()                                           # exp(-x^2 / (2σ^2))
                img = upfirdn2d.filter2d(img, f / f.sum())
            return img

        def run_Generator(self, z: torch.Tensor, c: torch.Tensor):
            ws = self.G.mapping(z, c)
            imgs = self.G.synthesis(ws)
            return imgs

        def run_Discriminator(self, imgs: torch.Tensor, c: torch.Tensor):
            if imgs.shape[-1] > self.G.img_resolution:
                imgs = F.interpolate(imgs, self.G.img_resolution, mode='area')
            imgs = self.blur(imgs, blur_sigma = self.blur_curr_sigma)
            return self.D(imgs, c)
        

        def accumulate_gradients(self,
                                phase: str, 
                                real_imgs: torch.Tensor, 
                                c_raw: List[str], 
                                gen_z: torch.Tensor,
                                cur_nimg: int,
                                verbose: bool = False):
            
            # gen_z    : Fake Images
            # real_imgs: Real Images
            self.set_blur_sigma(cur_nimg)
            batch_size = real_imgs.shape[0]
            self.training_stats = {}

            c_enc = None
            if isinstance(c_raw[0], str):
                c_enc = self.clip.encode_texts(c_raw)

            if phase == 'D':
                # Minimize logits for generated images
                fake_images      = self.run_Generator(gen_z, c_raw)
                fake_images_disc = self.run_Discriminator(fake_images, c_enc)

                fake_images_loss = (F.relu(torch.ones_like(fake_images_disc) + fake_images_disc)).mean() / batch_size
                # fake_images_loss.backward()
                                # 1 + -fake_logits; if disciminator is confident; fake_logits = 2 (above 1)
                                #                  (1 + -2) = -1; Relu(-1) = 0    NO PENALTY IS DISCRIMINATOR IS CONFIDENT

                                # 1 - fake_logits; if disciminator is NOT confident; fake_logits = 0 (below 0)
                                #                  (1 + -0) = 1; Relu(1) = 1         PENALTY IS DISCRIMINATOR IS NOT CONFIDENT

                real_images = real_imgs.detach().requires_grad_(False)
                real_images_disc = self.run_Discriminator(real_images, c_enc)
                real_images_loss = (F.relu(torch.ones_like(real_images_disc) - real_images_disc)).mean() / batch_size
                # real_images_loss.backward()
                                # 1 - real_logits; if disciminator is confident; real_logits = 2 (above 1)
                                #                  (1 - 2) = -1; Relu(-1) = 0    NO PENALTY IF DISCRIMINATOR IS CONFIDENT

                                # 1 - real_logits; if disciminator is NOT confident; real_logits = 0 (below 0)
                                #                  (1 - 0) = 1; Relu(1) = 1         PENALTY IF DISCRIMINATOR IS NOT CONFIDENT
                (fake_images_loss + real_images_loss).backward()
                self.training_stats = {
                    "Discriminator Score for Fake Images": round(fake_images_loss.item(),4),
                    "Discriminator Score for Real Images": round(real_images_loss.item(),4),
                    "Discriminator Total Loss"           : round(fake_images_loss.item() + real_images_loss.item(),4)
                }
                if verbose: print(training_stats)

            elif phase == "G":
                gen_img          = self.run_Generator(gen_z, c_raw)
                fake_images_disc = self.run_Discriminator(gen_img, c_enc)

                generator_loss = (-fake_images_disc).mean() / batch_size
                # If G is doing a good job -> gen_logits will be positive

                # Minimize spherical distance between image and text features
                clip_loss = 0
                if self.clip_weight > 0:
                    if gen_img.shape[-1] > 64:
                        gen_img = RandomCrop(64)(gen_img)
                    gen_img = F.interpolate(gen_img, 224, mode='area')
                    gen_img_features = self.clip.encode_image(gen_img.add(1).div(2))
                    clip_loss = self.spherical_distance(gen_img_features, c_enc).mean()

                total_generator_loss = generator_loss + self.clip_weight * clip_loss
                total_generator_loss.backward()

                training_stats = {
                    "Generator Loss"       : round(generator_loss.item(),4),
                    "CLIP Loss"            : round(clip_loss.item(),4),
                    "Generator Total Loss" : round(total_generator_loss.item(),4)
                }
                if self.clip_weight > 0:
                    self.training_stats["Clip Loss"] = clip_loss.item()
                if verbose: print(training_stats)
            