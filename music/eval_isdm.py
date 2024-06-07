import os
import functools

from evaluation.experiments import separate_slakh_msdm
from main.data import ChunkedSupervisedDataset
from main.module_base import Model
from audio_diffusion_pytorch import KarrasSchedule
from main.separation import separate_dataset, MSDMSeparator
from core.sep_isdm_ProjDiff import ISDMSeparator_ProjDiff
import torch
from metrics.cal_sisdr import calculate_sisdr

def differential_with_dirac(x, sigma, denoise_fn, mixture, source_id=0):
    num_sources = x.shape[1]
    x[:, [source_id], :] = mixture - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
    score = (x - denoise_fn(x, sigma=sigma)) / sigma
    scores = [score[:, si] for si in range(num_sources)]
    ds = [s - score[:, source_id] for s in scores]
    return torch.stack(ds, dim=1)

def main():
    dataset_path = 'data/slakh2100/test'
    model_path = ['ckpts/laced-dream-329/epoch=443-valid_loss=0.002.ckpt',\
            'ckpts/ancient-voice-289/epoch=258-valid_loss=0.019.ckpt',\
            'ckpts/honest-fog-332/epoch=407-valid_loss=0.007.ckpt',\
            'ckpts/ruby-dew-290/epoch=236-valid_loss=0.010.ckpt']
    output_dir = 'output/separations/ISDM_ProjDiff'
    source_id = 0
    sigma_min = 1e-4
    sigma_max = 1.0
    num_steps = 150
    batch_size = 32
    n_repeats=5
    lr=0.1
    beta=0.5
    resume = True
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset = ChunkedSupervisedDataset(
        audio_dir=dataset_path,
        stems=["bass", "drums", "guitar", "piano"],
        sample_rate=22050,
        max_chunk_size=262144,
        min_chunk_size=262144,
    )
    stems = ["bass", "drums", "guitar", "piano"]
    stem_to_model = {stems[k]: Model.load_from_checkpoint(model_path[k]).cuda() for k in range(4)}
    separator = ISDMSeparator_ProjDiff(
        stem_to_model=stem_to_model,
        sigma_schedule=KarrasSchedule(sigma_min=1e-4, sigma_max=1.0, rho=7.0),
        use_tqdm=True,
        n=n_repeats,
        lr=lr,
        beta=beta
    )
    separate_dataset(
        dataset=dataset,
        separator=separator,
        save_path=output_dir,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume
    )
    calculate_sisdr(dataset_path, output_dir)





if __name__ == '__main__':
    main()