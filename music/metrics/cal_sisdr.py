import os
import json
import torchaudio
import torch
import math
import numpy as np
from tqdm import tqdm
from torchaudio.functional import resample
from evaluation.evaluate_separation import get_full_tracks
from main.data import is_silent


def sisnr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    target_scaled = alpha * target
    noise = target_scaled - preds
    s_target = torch.sum(target_scaled**2, dim=-1) + eps
    s_error = torch.sum(noise**2, dim=-1) + eps
    return 10 * torch.log10(s_target / s_error)


def load_chunks(chunk_folder, stems):
    separated_tracks_and_rate = {s: torchaudio.load(os.sep.join([chunk_folder, '{}.wav'.format(s)])) for s in stems}
    separated_tracks = {k:t for k, (t,_) in separated_tracks_and_rate.items()}
    sample_rates_sep = [s for (_,s) in separated_tracks_and_rate.values()]

    assert len({*sample_rates_sep}) == 1, print(sample_rates_sep)
    sr = sample_rates_sep[0]

    return separated_tracks, sr

def load_and_resample_track(track_path, stems, resample_sr: int):
    
    def _load_and_resample_stem(stem):
        # stem_path = track_path/f"{stem}.wav"
        stem_path = os.sep.join([track_path, '{}.wav'.format(stem)])
        # if stem_path.exists():
        if os.path.exists(stem_path):
            wav, sr = torchaudio.load(stem_path)
            return resample(wav, sr, resample_sr)
        else:
            return None
    
    # Load and resample track stems
    stem_to_track = {s:_load_and_resample_stem(s) for s in stems}
    
    # Assert it contains at least a single source
    assert set(stem_to_track.values()) != {None}
    
    # Get sources dimensionality
    shapes = {wav.shape for s, wav in stem_to_track.items() if wav is not None}
    num_channels = {channels for (channels,length) in shapes}
    sample_lengths = {length for (channels,length) in shapes} 
    
    # Assert the existing sources have same dimensionality (up to certaian threshold)
    assert len(num_channels) == 1, f"{num_channels}"
    num_channels, = num_channels
    assert max(sample_lengths) - min(sample_lengths) <= 0.1 * resample_sr, f"{(max(sample_lengths) - min(sample_lengths))/resample_sr}"
    
    for s, wav in stem_to_track.items():
        # Initialize missing sources to zero
        if wav is None:
            stem_to_track[s] = torch.zeros(size=(num_channels, min(sample_lengths)) )
        
        # Crop sources
        else:
            stem_to_track[s] = stem_to_track[s][:,:min(sample_lengths)]
    
    return stem_to_track

def calculate_sisdr(gt_path, pred_path):
    expected_sample_rate = 22050
    separation_sr = 22050
    stems = ("bass","drums","guitar","piano")
    chunk_duration = 4.0
    overlap_duration = 2.0
    eps = 1e-8
    filter_single_source = True
    with open(os.sep.join([pred_path, 'chunk_data.json']), 'r') as f:
        chunk_data = json.load(f)
    track_to_chunks = {}
    for one_chunk in  chunk_data:
        track = one_chunk["track"]
        chunk_idx = one_chunk["chunk_index"]
        start_sample = one_chunk["start_chunk_sample"]
        if track not in track_to_chunks.keys():
            track_to_chunks[track] = [(start_sample, chunk_idx)]
        else:
            track_to_chunks[track].append( (start_sample, chunk_idx) )
    sisdr_list = {s:[] for s in stems}
    idx = 0
    for track, chunks in tqdm(track_to_chunks.items()):
        idx += 1
        sorted_chunks = sorted(chunks)
        separated_wavs = {s:[] for s in stems}
        for _, chunk_idx in sorted_chunks:
            chunk_folder = os.sep.join([pred_path, str(chunk_idx)])
            
            separated_chunks, sr = load_chunks(chunk_folder, stems)
            assert sr == expected_sample_rate, f"{sr} different from expected sample-rate {expected_sample_rate}"
            for s in separated_chunks.keys():
                separated_wavs[s].append(separated_chunks[s])
        for s in stems:
            separated_wavs[s] = torch.cat(separated_wavs[s], dim=-1)
        separated_track = separated_wavs
        original_track = load_and_resample_track(os.sep.join([gt_path, str(track)]), stems, 22050)
        for s in stems:
            max_length = separated_track[s].shape[-1]
            original_track[s] = original_track[s][:,:max_length]
        
        # Compute mixture
        mixture = sum([owav for owav in original_track.values()])
        chunk_samples = int(chunk_duration * separation_sr)
        overlap_samples = int(overlap_duration * separation_sr)

        # Calculate the step size between consecutive sub-chunks
        step_size = chunk_samples - overlap_samples

        # Determine the number of evaluation chunks based on step_size
        num_eval_chunks = math.ceil((mixture.shape[-1] - overlap_samples) / step_size)
        for i in range(num_eval_chunks):
            start_sample = i * step_size
            end_sample = start_sample + chunk_samples
            
            # Determine number of active signals in sub-chunk
            num_active_signals = 0
            for k in separated_track.keys():
                o = original_track[k][:,start_sample:end_sample]
                if not is_silent(o):
                    num_active_signals += 1
            
            # Skip sub-chunk if necessary
            if filter_single_source and num_active_signals <= 1:
                continue

            # Compute SI-SNRi for each stem
            for k in separated_track.keys():
                o = original_track[k][:,start_sample:end_sample]
                s = separated_track[k][:,start_sample:end_sample]
                m = mixture[:,start_sample:end_sample]
                sisdr = (sisnr(s, o, eps) - sisnr(m, o, eps)).item()
                # df_entries[k].append((sisnr(s, o, eps) - sisnr(m, o, eps)).item())
                sisdr_list[k].append(sisdr)
    for k in sisdr_list.keys():
        print('{}:{}'.format(k, np.mean(sisdr_list[k])))
if __name__ == '__main__':
    gt_path = 'data/slakh2100/test'
    pred_path = 'output/separations/PorjDiff'
    main(gt_path, pred_path)