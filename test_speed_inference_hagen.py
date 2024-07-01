import sys
import os
# BASE_PATH = "/opt/app/"
BASE_PATH = "/home/schaudhary/siva_projects/ai4life_mdc24/"
sys.path.append(os.path.join(BASE_PATH, 'DifFace'))

from datetime import datetime
from DifFace.custom_samplers_v2 import DiffusionSampler
from functools import partial
from omegaconf import OmegaConf
from pathlib import Path
import random
import tifffile
import torch
import torchvision as thv
import numpy as np


class TensorMinMaxNormalize(object):
    def __init__(self) -> None:
        pass

    def min_max_normalize_image(self, image):
        if image.ndim == 2:
            img = image
            return (img - torch.min(img))/(torch.max(img) - torch.min(img))
        elif image.ndim == 3:
            norm_img = []
            for c in range(image.shape[0]):
                img = image[c]
                norm_img.append((img - torch.min(img))/(torch.max(img) - torch.min(img)))
            norm_img = torch.stack(norm_img, axis=0)
            return norm_img
    
    def __call__(self, img):
        return self.min_max_normalize_image(img)


class Patchify(object):
    def __init__(self, size, stride, width, height):
        self.size = size
        self.stride = stride
        self.width = width
        self.height = height
        self.num_patch_x = int((width - size)/stride + 1)
        self.num_patch_y = int((height - size)/stride + 1)

    
    def unpatchify(self, img):
        img = torch.reshape(img, (-1, self.num_patch_y, self.num_patch_x, self.size, self.size))
        img = torch.permute(img, (0, 1, 3, 2, 4))
        img = torch.reshape(img, (-1, self.height, self.width))
        return img
    
    def __call__(self, img):
        patches = img.unfold(1, self.size, self.stride).unfold(2, self.size, self.stride)
        patches = patches.reshape(-1, self.size, self.size)
        return patches


def img_transform(img, patch_fn):
    transforms = thv.transforms.Compose([
            TensorMinMaxNormalize(),
            patch_fn,
            thv.transforms.Normalize(mean=(0.5), std=(0.5)),
        ])
    return transforms(img)

def get_sample_fn():
    cfg_path = os.path.join(BASE_PATH, 'DifFace/configs/sample/smaller_iddpm_hagen_actin_confocal_highsnr.yaml')
    gpu_id = 0
    timestep_respacing = '1000'
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = gpu_id
    configs.diffusion.params.timestep_respacing = timestep_respacing
    configs.model.ckpt_path = os.path.join(BASE_PATH, 'trained_models/smaller_difface_run_hagen_actin_confocal_highsnr/ema_ckpts/ema0999_model_280000.pth')
    sampler_dist = DiffusionSampler(configs)
    sampler_fn = partial(
        sampler_dist.repaint_style_sample,
        tN=120,
        repeat_timestep=5,
        num_repeats=5,
        mixing=True,
        mixing_stop=10,
        alpha_start=0.05,
        alpha_end=0.01,
        paint_stop=True
    )
    return sampler_fn


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = 4

    input_files = ['/home/schaudhary/siva_projects/Datasets/AI4Life/hagen_noisy.tiff']
    print(f"Found files: {len(input_files)}")
    
    sampler_fn = get_sample_fn()

    for input_file in input_files:
        img = tifffile.imread(input_file)
        img = img.astype(np.float32)
        print(img.shape)
        patch_fn = Patchify(256, 256, img.shape[1], img.shape[2])

        all_diff_pred = []
        if img.shape[0]%bs == 0:
            num_iters = img.shape[0]//bs
        else:
            num_iters = img.shape[0]//bs + 1
        for i in range(num_iters):
            tic = datetime.now()
            curr_img = torch.tensor(img[i*bs:min((i+1)*bs, img.shape[0])])
            X = img_transform(curr_img, patch_fn)
            print(X.shape)
            X = torch.unsqueeze(X, dim=1).to(device)
            diff_pred = sampler_fn(X, X)
            diff_pred = patch_fn.unpatchify(diff_pred[:, 0])
            print(diff_pred.shape)
            all_diff_pred.append(diff_pred.cpu().numpy())
            print(datetime.now() - tic)
        all_diff_pred = np.concatenate(all_diff_pred, axis=0)