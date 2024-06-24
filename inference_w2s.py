import sys
import os
BASE_PATH = "/opt/app/"
# BASE_PATH = "/Users/schaudhary/siva_projects/ai4life_mdc24/"
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


INPUT_PATH = Path('/input/images/image-stack-unstructured-noise/')
OUTPUT_PATH = Path('/output/images/image-stack-denoised/')
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

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
        img = torch.reshape(img, (self.num_patch_y, self.num_patch_x, self.size, self.size))
        img = torch.permute(img, (0, 2, 1, 3))
        img = torch.reshape(img, (self.height, self.width))
        return img
    
    def __call__(self, img):
        patches = img.unfold(1, self.size, self.stride).unfold(2, self.size, self.stride)
        patches = patches.reshape(-1, self.size, self.size)
        return patches


def img_transform(img, patch_fn):
    transforms = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            TensorMinMaxNormalize(),
            patch_fn,
            thv.transforms.Normalize(mean=(0.5), std=(0.5)),
        ])
    return transforms(img)

def get_sample_fn(ch):
    if ch == 0:
        cfg_path = os.path.join(BASE_PATH, 'DifFace/configs/sample/iddpm_w2s_wf_channel0_mean.yaml')
        gpu_id = 0
        timestep_respacing = '1000'
        configs = OmegaConf.load(cfg_path)
        configs.gpu_id = gpu_id
        configs.diffusion.params.timestep_respacing = timestep_respacing
        configs.model.ckpt_path = os.path.join(BASE_PATH, 'trained_models/difface_run_w2s_wf_channel0_mean/ema_ckpts/ema0999_model_800000.pth')
        sampler_dist = DiffusionSampler(configs)
        sampler_fn = partial(
            sampler_dist.repaint_style_sample,
            tN=100,
            repeat_timestep=5,
            num_repeats=3,
            mixing=True,
            mixing_stop=20,
            alpha_start=0.1,
            alpha_end=0.05,
            paint_stop=False
        )
    elif ch == 1:
        cfg_path = os.path.join(BASE_PATH, 'DifFace/configs/sample/iddpm_w2s_wf_channel1_mean.yaml')
        gpu_id = 0
        timestep_respacing = '1000'
        configs = OmegaConf.load(cfg_path)
        configs.gpu_id = gpu_id
        configs.diffusion.params.timestep_respacing = timestep_respacing
        configs.model.ckpt_path = os.path.join(BASE_PATH, 'trained_models/difface_run_w2s_wf_channel1_mean/ema_ckpts/ema0999_model_800000.pth')
        sampler_dist = DiffusionSampler(configs)
        sampler_fn = partial(
            sampler_dist.repaint_style_sample,
            tN=100,
            repeat_timestep=2,
            num_repeats=3,
            mixing=True,
            mixing_stop=20,
            alpha_start=0.1,
            alpha_end=0.1,
            paint_stop=False
        )
    elif ch == 2:
        cfg_path = os.path.join(BASE_PATH, 'DifFace/configs/sample/iddpm_w2s_wf_channel2_mean.yaml')
        gpu_id = 0
        timestep_respacing = '1000'
        configs = OmegaConf.load(cfg_path)
        configs.gpu_id = gpu_id
        configs.diffusion.params.timestep_respacing = timestep_respacing
        configs.model.ckpt_path = os.path.join(BASE_PATH, 'trained_models/difface_run_w2s_wf_channel2_mean/ema_ckpts/ema0999_model_800000.pth')
        sampler_dist = DiffusionSampler(configs)
        sampler_fn = partial(
            sampler_dist.repaint_style_sample,
            tN=90,
            repeat_timestep=2,
            num_repeats=5,
            mixing=True,
            mixing_stop=20,
            alpha_start=0.1,
            alpha_end=0.01,
            paint_stop=False
        )
    return sampler_fn


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(10)

    input_files = sorted(INPUT_PATH.glob(f"*.tif*"))
    print(f"Found files: {len(input_files)}")
    
    all_sampler_fn = {
        0: get_sample_fn(0),
        1: get_sample_fn(1),
        2: get_sample_fn(2),
    }
    for input_file in input_files:
        img = tifffile.imread(input_file)
        img = img.astype(np.float32)

        if img.ndim == 4:
            patch_fn = Patchify(256, 256, img.shape[2], img.shape[3])
            all_diff_pred = []
            for n in range(img.shape[0]):
                curr_img = img[n]
                curr_diff_pred = []
                for ch in range(curr_img.shape[0]):
                    X = img_transform(img[ch], patch_fn)
                    X = torch.unsqueeze(X, dim=1).to(device)
                    diff_pred = all_sampler_fn[ch](X, X)
                    diff_pred = patch_fn.unpatchify(diff_pred[:, 0])
                    diff_pred = TensorMinMaxNormalize().min_max_normalize_image(diff_pred)
                    curr_diff_pred.append(diff_pred.cpu().numpy())
                curr_diff_pred = np.stack(curr_diff_pred, axis=0)
                all_diff_pred.append(curr_diff_pred)
            all_diff_pred = np.stack(all_diff_pred, axis=0)
            tifffile.imwrite(os.path.join(OUTPUT_PATH, f"{input_file.stem}.tif"), all_diff_pred)
        elif img.ndim == 3:
            patch_fn = Patchify(256, 256, img.shape[1], img.shape[2])
            all_diff_pred = []
            for ch in range(img.shape[0]):
                X = img_transform(img[ch], patch_fn)
                X = torch.unsqueeze(X, dim=1).to(device)
                diff_pred = all_sampler_fn[ch](X, X)
                diff_pred = patch_fn.unpatchify(diff_pred[:, 0])
                diff_pred = TensorMinMaxNormalize().min_max_normalize_image(diff_pred)
                all_diff_pred.append(diff_pred.cpu().numpy())
            all_diff_pred = np.stack(all_diff_pred, axis=0)
            tifffile.imwrite(os.path.join(OUTPUT_PATH, f"{input_file.stem}.tif"), all_diff_pred)

 
 