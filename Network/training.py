import torch
from torch import nn

from Generator import Generator
from Discriminator import ProjectedDiscriminator

from dataset_creation import StyleDataset
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from helper import fetch_data
from loss import ProjectedGANLoss

from config import *


# INITIALIZING
generator     = Generator(z_dim = Z_DIM, conditional = True, img_resolution = IMG_RESOLUTION)
discriminator = ProjectedDiscriminator(c_dim = C_DIM)

discriminator.name = "D"
generator.name     = "G"

discriminator.opt = torch.optim.Adam(generator.parameters(), lr = 0.002, betas=[0, 0.99])
generator.opt = torch.optim.Adam(discriminator.parameters(), lr = 0.002, betas=[0, 0.99])

optimizer_gen = torch.optim.Adam(generator.parameters(), lr = 0.002, betas=[0, 0.99])
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr = 0.002, betas=[0, 0.99])

# FREEZING
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

# TORCH INITIALIZING    
cudnn_benchmark = True

torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.


# EMA gives more weight to recent values but still considers past history. 
# It’s like a “soft average” that forgets old data slowly.
# EMA(t) = beta . EMA(t) - 1 + (1 - beta) . xt

#  beta in [0, 1): decay rate (e.g., 0.99 or 0.999)
#  xt            : the current value (e.g., a model parameter or loss)
#  EMA(t)        : the new smoothed value
#  EMA(t-1)      : the previous smoothed value

# DATASET
sd = StyleDataset(path="/Users/mohamedmafaz/Desktop/StyleGAN-T/notebooks/Networks/dataset/", resolution = IMG_RESOLUTION)
sdl = DataLoader(sd, BATCH_SIZE, shuffle=True)

# LOSS
loss = ProjectedGANLoss(G = generator, 
                        D = discriminator, 
                        blur_fade_kimg = BLUR_FADE_KIMG, # after 100 image there will be 0 Blur,
                        clip_weight = CLIP_wWEIGHT,
                        device = DEVICE
                        )

for epoch in tqdm(range(EPOCHS)):
    chance = 0
    cur_nimg = 0
    for real_images, real_labels in sdl:
        real_images = (real_images/(255/2)) - 1           # normalizing images from [-1, 1]
        batch_size = real_images.shape[0]
        all_gen_z = torch.randn([batch_size, Z_DIM], device = DEVICE)

        if chance % 2 == 0:
            phase = discriminator
        else: phase = generator

        # Train Discriminator and Generator
        phase.requires_grad_(True)
        partial_freeze(phase)
        loss.accumulate_gradients(phase = phase.name, cur_nimg = cur_nimg, real_imgs = real_images, c_raw = real_labels, gen_z = all_gen_z, verbose = False)
        
        training_stats = loss.training_stats
        if phase.name == "G": print("Generator Status")
        if phase.name == "D": print("Discriminator Status")
        print('-'*20)
        for key in training_stats:
            print(f"{key}: {training_stats[key]}", end = " || ")
        
        phase.opt.step()
        phase.opt.zero_grad()

        phase.requires_grad_(False)

        chance += 1
        cur_nimg += batch_size
