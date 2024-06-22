from math import ceil
from sampler import BaseSampler
from utils import util_image
import torch
import torchvision as thv
import numpy as np

class DiffusionSampler(BaseSampler):
    def sample_func(self, noise, start_timesteps=None, bs=4, num_images=1000, save_dir=None):
        print('Begining sampling:')

        h = w = self.configs.im_size
        if noise == None:
            if bs == None:
                bs = 1
        else:
            bs = noise.shape[0]
        if self.num_gpus != 0:
            total_iters = ceil(num_images / (bs * self.num_gpus))
        else:
            total_iters = ceil(num_images / (bs * 1))
        for ii in range(total_iters):
            if self.rank == 0 and self.display:
                print(f'Processing: {ii+1}/{total_iters}')
            if noise == None:
                if self.num_gpus != 0:
                    noise = torch.randn((bs, 1, h, w), dtype=torch.float32).cuda()
                else:
                    noise = torch.randn((bs, 1, h, w), dtype=torch.float32)
            else:
                noise = self.diffusion.q_sample(noise, torch.tensor(start_timesteps, device=noise.device))
            if 'ddim' in self.configs.diffusion.params.timestep_respacing:
                sample = self.diffusion.ddim_sample_loop(
                        self.model,
                        shape=(bs, 1, h, w),
                        noise=noise,
                        start_timesteps=start_timesteps,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        device=None,
                        progress=False,
                        eta=0.0,
                        )
            else:
                sample = self.diffusion.p_sample_loop(
                        self.model,
                        shape=(bs, 1, h, w),
                        noise=noise,
                        start_timesteps=start_timesteps,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        device=None,
                        progress=False,
                        )
            sample = util_image.normalize_th(sample, reverse=True).clamp(0.0, 1.0)
        return sample
    
    
    def repaint_style_sample(self, start_input, mixing_input, tN=100, repeat_timestep=1, num_repeats=1, mixing=False, mixing_stop=50, alpha_start=0.5, alpha_end=0.5, paint_stop=False, return_all=False):
        print('Begining sampling:')
        
        def linear_mixing_alphas(alpha_start, alpha_end, n_steps):
            mixing_alphas = np.linspace(alpha_end, alpha_start, n_steps)
            return mixing_alphas
            
        n_steps = max(tN - mixing_stop, 1)
        mixing_t_steps = {mixing_stop + i*int((tN - mixing_stop)/n_steps):i for i in range(0, n_steps)}
        mixing_alphas = linear_mixing_alphas(alpha_start, alpha_end, n_steps)

        all_imgs = []
        h = w = self.configs.im_size
        bs = start_input.shape[0]
        shape = (bs, 1, h, w)
        if start_input != None:
            img = self.diffusion.q_sample(start_input, torch.tensor(tN, device=start_input.device))
        else:
            img = torch.randn((bs, 1, h, w))
        if return_all == True:
            all_imgs.append(img)

        while tN > 0:
            indices = list(range(tN)[::-1])[0:repeat_timestep]
            print(f'repainting between {indices[0]}-{indices[-1]} for {num_repeats} times')
            for rp in range(num_repeats):
                # img = self.diffusion.q_sample(img, indices[0])
                for i in indices:
                    t = torch.tensor([i] * shape[0], device=img.device)
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            img,
                            t,
                            clip_denoised=True,
                            denoised_fn=None,
                            model_kwargs=None,
                        )
                        img = out["sample"]
                    if mixing == True and i >= mixing_stop:
                        xN_given_y0 = self.diffusion.q_sample(mixing_input, torch.tensor(i, device=mixing_input.device))
                        img = (1 - mixing_alphas[mixing_t_steps[i]])*img + mixing_alphas[mixing_t_steps[i]]*xN_given_y0
                        # img = (1 - alpha)*img + alpha*xN_given_y0
                    if return_all == True and rp == num_repeats - 1:
                        all_imgs.append(img)
            tN -= repeat_timestep
            if paint_stop == True and tN <= mixing_stop:
                num_repeats = 1
        sample = util_image.normalize_th(img, reverse=True).clamp(0.0, 1.0)
        all_imgs.append(sample)
        if return_all:
            return torch.stack(all_imgs, dim=0)
        else:
            return sample
    
    def mixing_sample(self, start_input, mixing_input, tN=100, mix_start=100, mix_stop=50, n_steps=1, alpha_start=1, alpha_end=0.1, scheduler='linear', return_all=False):
        def linear_mixing_alphas(alpha_start, alpha_end, n_steps):
            mixing_alphas = np.linspace(alpha_end, alpha_start, n_steps)
            return mixing_alphas

        def exponential_alphas(alpha_start, alpha_end, mix_start, mix_stop, n_steps):
            fn = lambda x: np.exp(0.1*(x - mix_start))*(alpha_start - alpha_end) + alpha_end
            t = np.linspace(mix_stop, mix_start, n_steps)
            mixing_alphas = fn(t)
            return mixing_alphas
        
        def cosine_alphas(alpha_start, alpha_end, mix_start, mix_stop, n_steps):
            fn = lambda x: np.cos((x - mix_start)/(mix_stop-mix_start)*np.pi/2)*(alpha_start - alpha_end) + alpha_end
            t = np.linspace(mix_stop, mix_start, n_steps)
            mixing_alphas = fn(t)
            return mixing_alphas

        assert (mix_start - mix_stop) % n_steps == 0
        mixing_t_steps = {mix_stop + i*int((mix_start - mix_stop)/n_steps):i for i in range(0, n_steps)}
        if scheduler == 'linear':
            mixing_alphas = linear_mixing_alphas(alpha_start, alpha_end, n_steps)
        elif scheduler == 'exponential':
            mixing_alphas = exponential_alphas(alpha_start, alpha_end, mix_start, mix_stop, n_steps)
        elif scheduler == 'cosine':
            mixing_alphas = cosine_alphas(alpha_start, alpha_end, mix_start, mix_stop, n_steps)
        xN_given_y0 = self.diffusion.q_sample(start_input, torch.tensor(tN, device=start_input.device))
        img = xN_given_y0

        all_imgs = []
        all_imgs.append(img)
        h = w = self.configs.im_size
        bs = img.shape[0]
        shape = (bs, 1, h, w)
        indices = list(range(tN)[::-1])
        for i in indices:
            t = torch.tensor([i] * shape[0], device=start_input.device)
            with torch.no_grad():
                out = self.diffusion.p_sample(
                    self.model,
                    img,
                    t,
                    clip_denoised=True,
                    denoised_fn=None,
                    model_kwargs=None,
                )
                img = out["sample"]
            if i in mixing_t_steps:
                print(f'mixing at {i}', (1 - mixing_alphas[mixing_t_steps[i]]), mixing_alphas[mixing_t_steps[i]])
                mixing_xN_given_y0 = self.diffusion.q_sample(mixing_input, torch.tensor(i, device=mixing_input.device))
                img = (1 - mixing_alphas[mixing_t_steps[i]])*img + mixing_alphas[mixing_t_steps[i]]*mixing_xN_given_y0
                # img = (1 - mixing_alphas[mixing_t_steps[i]])*img + mixing_alphas[mixing_t_steps[i]]*mixing_xN_given_y0[mixing_t_steps[i]]
            if return_all == True:
                all_imgs.append(img)
        sample = util_image.normalize_th(img, reverse=True).clamp(0.0, 1.0)
        all_imgs.append(sample)
        if return_all == True:
            return torch.stack(all_imgs, dim=0)
        else:
            return sample