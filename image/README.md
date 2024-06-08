# Image restoration of paper: Unleashing the Denoising Capability of Diffusion Prior for Solving Inverse Problems

This project is based on:

\- https://github.com/bahjat-kawar/ddrm (DDRM),

\- https://github.com/wyhuai/DDNM (DDNM), and

\- https://github.com/DPS2022/diffusion-posterior-sampling (DPS)

## Environment

You can set up the environment using the `environment.yml`. Run

```bash
conda env create -f environment.yml
conda activate projdiff_ir
```

## Experiments in the paper

### Pre-trained models

We use pre-trained models on ImageNet, CelebA-HQ, and FFHQ. Please download the pre-trained models from https://github.com/openai/guided-diffusion for ImageNet ([256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)), from https://github.com/ermongroup/SDEdit for CelebA-HQ (https://drive.google.com/file/d/1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21/view?usp=share_link), and from https://github.com/DPS2022/diffusion-posterior-sampling  for FFHQ (https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh). Place them into exp/logs/[dataset_name] respectively.

### Test datasets

We use 1000 test samples for each dataset. Download the ImageNet validation set for ImageNet and place it into exp/datasets/imagenet. Download the CelebA-HQ test set from https://github.com/wyhuai/DDNM (https://drive.google.com/drive/folders/1cSCTaBtnL7OIKXT4SVME88Vtk4uDd_u4) and place it into exp/datasets/celeba. Download the FFHQ test set from https://github.com/NVlabs/ffhq-dataset and place it into exp/datasets/ffhq (Note: you only need to download the folder 00000 for testing). Thus the exp/ folder should look as follows:

```bash
exp
├── logs
├── datasets
│   ├── celeba # all CelebA files
│   ├── imagenet # all ImageNet files
│   ├── ffhq # out of distribution ImageNet images
└── imagenet_val_1k.txt # list of the 1k images used in ImageNet-1K.
```

### Reproduce the results

Please run the following code:

```
python main.py --ni --config {CONFIG}.yml --doc {DATASET} --timesteps {STEPS} --lr {lr} --default_lr --deg {DEGRADATION} --sigma_0 {SIGMA_0} -i {IMAGE_FOLDER}
```

where the following are options

- `STEPS` controls how many timesteps used in the process (recommend 1000 for phase retrieval and 100 for other experiments).
- `DEGREDATION` choose from: `deblur_gauss`, `sr4`,  `inp`, `phase`, `hdr`)
- `SIGMA_0` is the noise observed in y.
- `CONFIG` is the name of the config file. Choose from `imagenet_256.yml`, `celeba_hq.yml`, and `ffhq.yml`.
- `DATASET` is the name of the dataset used. Choose from `imagenet`, `celeba`, and `ffhq` (though only ImageNet requires this term).
- `IMAGE_FOLDER` is the name of the folder the resulting images will be placed in (default: `images`).
- `lr` is the step size (or learning rate) for ProjDiff algorithm.
- `default_lr` provides the default learning rate for reproducing the results in the paper. Set this term to use the tuned best step sizes.

e.g., for noise-free super-resolution experiment on ImageNet, run

```bash
python main.py --ni --config imagenet_256.yml --doc imagenet --timesteps 100 --default_lr --deg sr4 --sigma_0 0.00
```

### Calculate the metrics

We provide the example code for calculating the metrics in `calculate_metrics/`. Please first copy all the images using `mv_files.py` (mainly for FID calculation), and then run `cal_metrics.py` to calculate the metrics. Note that you may need to adjust the paths in the files.

## References and Acknowledgements