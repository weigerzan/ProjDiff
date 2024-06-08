# Source separation and partial generation of paper: Unleashing the Denoising Capability of Diffusion Prior for Solving Inverse Problems

This project is based on:

\- https://github.com/gladia-research-group/multi-source-diffusion-models (MSDM),

## Environment

You can set up the environment using the `env.yaml`. Run

```bash
conda env create -f env.yaml
conda activate projdiff_music
```

## Test dataset

The test dataset can be downloaded from https://drive.google.com/file/d/1Xo-bGORndJhenHvzf3eY5Lt0XCTgElZP/view?usp=sharing.

## Pre-trained checkpoints

Please follow the README.md in `./ckpts` to prepare the checkpoints.

## Experiments

### Separation

#### MSDM

Please run the following command

```bash
python eval_msdm.py --config eval_msdm.yaml --output_dir {output_dir} --lr {lr} --N {N} --beta {beta}
```

`output_dir` is the save folder of the results.

`lr` is the step size.

`N` is the number of repetitions.

`beta` is the momentum.

To reproduce the results in the paper, you can leave all these parameters default. If you would like to tune `N` or `lr`, it is recommended to set them as `lr`=0.5/`N`.

#### ISDM

Please run the following command

```bash
python eval_msdm.py --config eval_weakly_msdm.yaml --output_dir {output_dir} --lr {lr} --N {N} --beta {beta}
```

### Partial Generation

Please run the following command

```bash
python partial_generate.py --config eval_generation.yaml --output_dir {output_dir} --lr {lr} --N {N} --beta {beta} --stems_to_inpaint {stems_to_inpaint}
```

`stems_to_inpaint` is the stems to be generated, represented by their first letters. For example, to generate bass, drums, and guitar based on piano, `stems_to_inpaint ` should be "BDG".

### Calculate metrics

For Separation tasks, the SI-SDR_i metrics will be calculated after all the tracks are separated. If you'd like to adjust the calculation process, please refer to `metrics/cal_sisdr.py` for more details.

For generation tasks, first, you need to arrange all the generated tracks into a generation folder and all the ground truth tracks into a gt folder, respectively. Then use the frechet_audio_distance package to calculate the FAD metrics. We provide example codes in the `metrics/mv_files.py` and `metrics/calculate_fad.py`.

