import abc
from pathlib import Path
from typing import List, Optional, Callable, Mapping

import torch
import torchaudio
import tqdm
from math import sqrt, ceil

from audio_diffusion_pytorch.diffusion import Schedule
from torch.utils.data import DataLoader

from main.data import assert_is_audio, SeparationDataset
from main.module_base import Model

class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        
    @abc.abstractmethod
    def separate(mixture, num_steps) -> Mapping[str, torch.Tensor]:
        ...
    
    
class MSDMSeparator_ProjDiff(Separator):
    def __init__(self, model: Model, stems: List[str], sigma_schedule: Schedule, **kwargs):
        super().__init__()
        self.model = model
        self.stems = stems
        self.sigma_schedule = sigma_schedule
        self.separation_kwargs = kwargs
    
    def separate(self, mixture: torch.Tensor, num_steps:int = 100):
        device = self.model.device
        mixture = mixture.to(device)
        batch_size, _, length_samples = mixture.shape
        
        y = separate_mixture_DDIM(
            mixture=mixture,
            denoise_fn=self.model.model.diffusion.denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(self.stems), length_samples).to(device),
            **self.separation_kwargs,
        )
        return {stem:y[:,i:i+1,:] for i,stem in enumerate(self.stems)}


def differential_with_dirac(x, sigma, denoise_fn, mixture, source_id=0):
    num_sources = x.shape[1]
    x[:, [source_id], :] = mixture - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
    score = (x - denoise_fn(x, sigma=sigma)) / sigma
    scores = [score[:, si] for si in range(num_sources)]
    ds = [s - score[:, source_id] for s in scores]
    return torch.stack(ds, dim=1)

@torch.no_grad()
def separate_mixture_DDIM(
    mixture: torch.Tensor, 
    denoise_fn: Callable,
    sigmas: torch.Tensor,
    noises: Optional[torch.Tensor],
    use_tqdm: bool = False,
    beta: float=0.5,
    n: int=5,
    lr: float=None,
):      
    def prox(x, mixture):
        prox_x = x + 0.25 * (mixture - torch.sum(x, dim=1, keepdim=True)).repeat(1,4,1)
        return prox_x
    # Set initial noise
    x = sigmas[0] * noises # [batch_size, num-sources, sample-length]
    source_id = 0
    vis_wrapper  = tqdm.tqdm if use_tqdm else lambda x:x 
    # print(sigmas)
    x_0 = None
    momentum = None
    for i in vis_wrapper(range(len(sigmas) - 1)):
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        for k in range(n):
            A = torch.ones([1, 4]).cuda()
            num_sources = x.shape[1]
            if x_0 is None:
                x_0_pred = denoise_fn(x, sigma=sigma)
                x_0 = denoise_fn(torch.randn_like(x), sigma=sigma)
                diff = -(torch.sum(x_0, dim=1, keepdim=True) - mixture).repeat(1,4,1)
            else:
                x = x_0 + sigma * torch.randn_like(x_0)
                x_0_pred = denoise_fn(x, sigma=sigma)
            
            diff = (x_0_pred - x_0)
            if momentum is None:
                momentum=diff
            else:
                momentum = beta * momentum + (1-beta) * diff
            x_0 += lr * momentum
            x_0 = prox(x_0, mixture)
            # loss_obs = torch.mean(torch.norm(torch.sum(x_0, dim=1, keepdim=True) - mixture, dim=2)).item()
            # loss_cons = torch.mean(torch.norm(x_0-x_0_pred, dim=[1,2])).item()
            # print('obs loss:{}, cons loss:{}, lr:{}, sigma:{}'.format(loss_obs, loss_cons, lr, sigma))
    return x_0.cpu().detach()