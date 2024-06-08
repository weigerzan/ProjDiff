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
import argparse
import yaml
import json

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--config",
        type=str,
        default="eval_weakly_msdm.yaml",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ISDM_projdiff",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Step-size"
    )
    parser.add_argument(
        "--N", type=int, default=5, help="N repeats"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Momentum"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last run"
    )
    args = parser.parse_args()
    with open(os.sep.join(['exp', args.config]), "r") as file:
        config = yaml.safe_load(file)
    return args, config


def main():
    args, config = parse_args_and_config()
    dataset_path = config.dataset_path
    model_path = [config.separation.model_paths.bass,\
         config.separation.model_paths.drums,\
        config.separation.model_paths.guitar,\
        config.separation.model_paths.piano]
    output_dir = os.sep.join(['output/separations', args.output_dir])
    sigma_min = config.separation.sigma_min
    sigma_max = config.separation.sigma_max
    num_steps = config.separation.num_steps
    batch_size = config.separation.batch_size
    n_repeats=args.N
    lr=args.lr
    beta=args.beta
    resume = args.resume
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

    chunk_data = []
    for i in range(len(dataset)):
        start_sample, end_sample = dataset.get_chunk_indices(i)
        chunk_data.append(
            {
                "chunk_index": i,
                "track": dataset.get_chunk_track(i),
                "start_chunk_sample": start_sample,
                "end_chunk_sample": end_sample,
                "track_sample_rate": dataset.sample_rate,
                "start_chunk_seconds": start_sample / dataset.sample_rate,
                "end_chunk_in_seconds": end_sample / dataset.sample_rate,
            }
        )

    # Save chunk metadata
    with open(output_dir / "chunk_data.json", "w") as f:
        json.dump(chunk_data, f, indent=1)
        
    calculate_sisdr(dataset_path, output_dir)





if __name__ == '__main__':
    main()