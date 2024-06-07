from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import skimage
import numpy as np
import os
from skimage.color import rgb2ycbcr
import torch_fidelity
import tqdm
import lpips
import torch

orig_path = 'exp/proxdiff_100steps/orig'
generated_path = 'exp/proxdiff_100steps/generated'
N = len(os.listdir(generated_path))
assert N == 1000
# Calculated SSIM
SSIM_sum = 0
PSNR_sum = 0
LPIPS_sum = 0
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
print('calculating PSNR, SSIM & LPIPS')
with torch.no_grad():
    for k in tqdm.tqdm(range(N)):
        source_path = os.sep.join([orig_path, 'orig_{}.png'.format(k)])
        source_image = skimage.io.imread(source_path)/255.0
        # source_image = rgb2ycbcr(source_image/255.0)[:, :, 0]

        denoise_path = os.sep.join([generated_path, '{}_-1.png'.format(k)])
        generated_image = skimage.io.imread(denoise_path)/255.0
        # print(source_image)
        # generated_image = rgb2ycbcr(generated_image/255.0)[:, :, 0]
        # print(source_image.shape)
        # print(generated_image)
        # SSIM = ssim(source_image, generated_image, data_range=generated_image.max() - generated_image.min(), channel_axis=-1)
        SSIM = ssim(source_image, generated_image, data_range=generated_image.max() - generated_image.min(), channel_axis=-1)
        SSIM_sum += SSIM
        PSNR = psnr(source_image, generated_image)
        PSNR_sum += PSNR
        source_image = source_image * 2 - 1
        generated_image = generated_image * 2 - 1
        LPIPS = loss_fn_vgg(torch.tensor(source_image).permute(2,0,1).to(torch.float32).cuda(), torch.tensor(generated_image).permute(2,0,1).to(torch.float32).cuda())
        LPIPS_sum += LPIPS[0,0,0,0]
        # print(SSIM_sum/(k+1))
        # print(PSNR_sum/(k+1))
    # print('Average SSIM: {}'.format(SSIM_sum/N))
    print('Average LPIPS: {}'.format(LPIPS_sum/N))
    print('calculating KID & FID')
    Results = torch_fidelity.calculate_metrics(input1=orig_path, input2=generated_path, kid=True, fid=True)
    print('PSNR:{}, SSIM:{}, LPIPS:{}, KID:{}, FID:{}'.format(PSNR_sum/N, SSIM_sum/N, LPIPS_sum/N, Results['kernel_inception_distance_mean'], Results['frechet_inception_distance']))