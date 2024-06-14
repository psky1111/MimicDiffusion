import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
import torchvision

from utils import diff2clf, clf2diff, normalize, resize

class inverse_high_resolution_projection(torch.nn.Module):
    def __init__(self,scale_factor=4) -> None:
        super().__init__()
        self.upsample = partial(torch.nn.functional.interpolate, scale_factor=scale_factor)
        self.downsample = resize
        self.scale_factor = scale_factor
    #@torch.no_grad()
    def forward(self,x):
        return self.downsample(x,1/self.scale_factor)
    #@torch.no_grad()
    def transpose(self,x):
        return self.upsample(x)
def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()


class PurificationForward(torch.nn.Module):
    def __init__(self, clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.max_timestep = max_timestep
        self.attack_steps = attack_steps
        self.sampling_method = sampling_method
        self.projection = inverse_high_resolution_projection(16)
        assert sampling_method in ['ddim', 'ddpm']
        if self.sampling_method == 'ddim':
            self.eta = 0
        elif self.sampling_method == 'ddpm':
            self.eta = 1
        self.is_imagenet = is_imagenet

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if type(t) == int:
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x

    def denoising_process(self, x, seq, ref, rho_scale=7.5):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        ori_x = ref
        xt = x
        count = 0
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            count += 1
        return xt

    def preprocess(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            #noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            noised_x = torch.randn_like(x_diff)
            x_diff = self.denoising_process(noised_x, self.attack_steps[i], ref=x_diff)

        x_clf = diff2clf(x_diff)
        return x_clf

    def classify(self, x):
        logits = self.clf(x)
        return logits

    def forward(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(noised_x, self.attack_steps[i],ref=x_diff)


        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        else:
            x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return logits

# our method
class PurificationForward_mimic(torch.nn.Module):
    def __init__(self, clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.max_timestep = max_timestep
        self.attack_steps = attack_steps
        self.sampling_method = sampling_method
        self.projection = inverse_high_resolution_projection(4)
        self.phi = 8/255
        assert sampling_method in ['ddim', 'ddpm']
        if self.sampling_method == 'ddim':
            self.eta = 0
        elif self.sampling_method == 'ddpm':
            self.eta = 1
        self.is_imagenet = is_imagenet

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if type(t) == int:
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x

# main algorithm
    def denoising_process(self, x, seq, ref, rho_scale=3000):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        ori_x = ref
        xt = x
        count = 0
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            #two guidance
            guidances = 0.
            # freedom strategy
            if 90>count>20:
                xt.requires_grad_()
                with torch.enable_grad():
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    measure_norm= torch.norm(x0_t - ori_x,1,1).mean()
                    measure_super = torch.norm(self.projection.transpose(x0_t) - self.projection.transpose(ori_x),1,1).mean()

                norm_gradient = torch.autograd.grad(measure_norm,[xt],retain_graph=True)[0].detach()
                super_norm_gradient = torch.autograd.grad(measure_super,[xt],retain_graph=True)[0].detach()
                

                print(f"the norm is {measure_norm.item()}")
                rho_norm = rho_scale * at.sqrt()
                rho_super =  rho_scale * at.sqrt()
                

                guidances = rho_norm*norm_gradient + rho_super*super_norm_gradient
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et - guidances
            count += 1
        return xt

    def preprocess(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            #noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            noised_x = torch.randn_like(x_diff)
            x_diff = self.denoising_process(noised_x, self.attack_steps[i], ref=x_diff)

        x_clf = diff2clf(x_diff)
        return x_clf

    def classify(self, x):
        logits = self.clf(x)
        return logits

    def forward(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(torch.randn_like(x_diff), self.attack_steps[i],ref=x_diff)



        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        else:
            x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return logits
