import os
import functools

from evaluation.experiments import separate_slakh_msdm
from main.data import ChunkedSupervisedDataset, assert_is_audio
from main.module_base import Model
from audio_diffusion_pytorch import KarrasSchedule
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torchaudio
import math


@torch.no_grad()
def generate_track(
    denoise_fn,
    sigmas,
    noises,
    source=None,
    mask=None,
    lr=0.05,
    n_repeats=1,
    beta=0.9
) -> torch.Tensor:

    def prox(x, source, mask):
        prox_x = x * (1-mask) + source * mask
        return prox_x
    x = sigmas[0] * noises
    
    _, num_sources, _  = x.shape    
    # Initialize default values
    source = torch.zeros_like(x) if source is None else source
    mask = torch.zeros_like(x) if mask is None else mask
    
    sigmas = sigmas.to(x.device)
    
    x_0 = None
    momentum = None
    # Noise source to current noise level

    # Iterate over all timesteps
    last_noises = noises
    for i in tqdm(range(len(sigmas) - 1)):
        # print(i)
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        for k in range(n):
            # print(k)
            num_sources = x.shape[1]
            if x_0 is None:
                x_0_pred = denoise_fn(x, sigma=sigma)
                x_0 = denoise_fn(torch.randn_like(x), sigma=sigma)
            else:
                x = x_0 + sigma_next * noises + (sigma**2 - sigma_next**2)**0.5 * torch.randn_like(x_0) # Using Restricted Encoding
                x_0_pred = denoise_fn(x, sigma=sigma)
            diff = (x_0_pred - x_0)
            # diff = (x_0_pred - x) / sigma
            if momentum is None:
                momentum=diff
            else:
                momentum = beta * momentum + (1-beta) * diff
            # print(momentum[0])
            x_0 += lr * momentum
            # x += momentum * (sigma - sigma_next)
            x_0 = prox(x_0, source, mask)
            loss_cons = torch.mean(torch.norm(x_0-x_0_pred, dim=[1,2])).item()
            # print(torch.norm(source[0]))
            print('cons loss:{}, lr:{}, sigma:{}, norm:{}'.format(loss_cons, lr, sigma, torch.norm(x[0])))
    return x_0

@torch.no_grad()
def generate_inpaint_mask(sources, stem_to_inpaint):
    mask = torch.ones_like(sources)
    for stem_idx in stem_to_inpaint:
        mask[:,stem_idx,:] = 0.0
    return mask

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--config",
        type=str,
        default="eval_generation.yaml",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="projdiff",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--stems_to_inpaint",
        type=str,
        default="BDG",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="Step-size"
    )
    parser.add_argument(
        "--N", type=int, default=1, help="N repeats"
    )
    parser.add_argument(
        "--beta", type=float, default=0.9, help="Momentum"
    )
    parser.add_argument(
        "--resume", type=store_true, help="Resume from last run"
    )
    args = parser.parse_args()
    with open(os.sep.join(['exp', args.config]), "r") as file:
        config = yaml.safe_load(file)
    return args, config


def main():
    args, config = parse_args_and_config()
    model_path = config.separation.model_path
    output_dir = os.sep.join(['output/partial_generating', args.stems_to_inpaint, args.output_dir])
    model = Model.load_from_checkpoint(configs.generation.model_path).cuda()
    sigma_min = configs.generation.sigma_min
    sigma_max = configs.generation.sigma_max
    num_steps = configs.generation.num_steps
    batch_size = configs.generation.batch_size
    sample_rate=22050
    lr=args.lr
    n_repeats=args.n_repeats
    beta=args.beta
    stems = ["bass", "drums", "guitar", "piano"]
    stems_to_inpaint = []
    if 'B' in args.stems_to_inpaint:
        stems_to_inpaint.append("bass")
    if 'D' in args.stems_to_inpaint:
        stems_to_inpaint.append("drums")
    if 'G' in args.stems_to_inpaint:
        stems_to_inpaint.append("guitar")
    if 'P' in args.stems_to_inpaint:
        stems_to_inpaint.append("piano")
    resume = args.resume
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset = ChunkedSupervisedDataset(
        audio_dir=dataset_path,
        stems=["bass", "drums", "guitar", "piano"],
        sample_rate=sample_rate,
        max_chunk_size=262144,
        min_chunk_size=262144,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)
    # schedule = KarrasSchedule(sigma_min=1e-4, sigma_max=20.0, rho=7)(num_steps, model.device)
    schedule = KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7)(num_steps, model.device)
    denoise_fn = model.model.diffusion.denoise_fn

    stemidx_to_inpaint = [i for i,s in enumerate(stems) if s in stems_to_inpaint]
    inpaint_mask = None
    chunk_id = 0
    for idx, batch_data in enumerate(loader):
        # batch_data: List, 4 * bs * 1 * lens
        # print(batch_data)
        data = torch.cat([batch_data[0], batch_data[1], batch_data[2], batch_data[3]], dim=1).cuda()
        # torchaudio.save('test.wav', data[0, [0], :].cpu(), sample_rate=sample_rate)
        # print(data.shape)
        if inpaint_mask is None or inpaint_mask.shape[0] != data.shape[0]:
            inpaint_mask = generate_inpaint_mask(data, stem_to_inpaint=stemidx_to_inpaint)
        # print(inpaint_mask[0])
        inpainted_tracks = generate_track(
            source=data,
            mask=inpaint_mask,
            denoise_fn=denoise_fn,
            sigmas=schedule,
            noises=torch.randn_like(data),
            lr=lr,
            n_repeats=n_repeats,
            beta=beta
        )
        # inpainted_tracks = {"bass": inpainted_tracks[:, 0, :], "drums", "guitar", "piano"}
        num_samples = inpainted_tracks.shape[0]
        for i in range(num_samples):
            chunk_path_separate = os.sep.join([output_dir, 'separate', str(chunk_id)])
            chunk_path_sum = os.sep.join([output_dir, 'sum', str(chunk_id)])
            os.makedirs(chunk_path_separate, exist_ok=True)
            os.makedirs(chunk_path_sum, exist_ok=True)
            one_track = {'bass': inpainted_tracks[i, [0], :], 'drums': inpainted_tracks[i, [1], :], 'guitar': inpainted_tracks[i, [2], :],\
                     'piano': inpainted_tracks[i, [3], :], 'mixture': torch.sum(inpainted_tracks[i, :, :], dim=0, keepdim=True), \
                    'gt_mixture': torch.sum(data[i, :, :], dim=0, keepdim=True)}
            for stem, separated_track in one_track.items():
                assert_is_audio(separated_track)
                torchaudio.save(os.sep.join([chunk_path_separate, '{}.wav'.format(stem)]), separated_track.cpu(), sample_rate=sample_rate)
            assert_is_audio(one_track['mixture'])
            torchaudio.save(os.sep.join([chunk_path_sum, 'mixture.wav']), one_track['mixture'].cpu(), sample_rate=sample_rate)
            torchaudio.save(os.sep.join([chunk_path_sum, 'gt_mixture.wav']), one_track['gt_mixture'].cpu(), sample_rate=sample_rate)

            chunk_id += 1

if __name__ == '__main__':
    main()