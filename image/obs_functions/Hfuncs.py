import numpy as np
import torch
from .fastmri_utils import fft2c_new, ifft2c_new
from torch.nn import functional as F
import math


def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
  """ IFFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        singular_inverse = singulars
        singular_inverse[singulars != 0] = 1 / singulars[singulars != 0]
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * singular_inverse
        return self.V(self.add_zeros(temp))

class SuperResolution:
    def __init__(self, channels, img_dim, ratio, device): #ratio = 2 or 4
        self.channels=channels
        self.img_dim=img_dim
        self.ratio=ratio
        self.device=device
    
    def downsampling(self, img):
        assert img.shape[1] == 3
        down_img = torch.zeros([img.shape[0], img.shape[1], int(img.shape[2]/self.ratio), int(img.shape[3]/self.ratio)]).to(self.device)
        for k in range(self.ratio):
            for j in range(self.ratio):
                down_img += img[:, :, k::self.ratio, j::self.ratio]
        down_img /= self.ratio**2
        return down_img

    
    def upsampling(self, img):
        up_img = torch.zeros([img.shape[0], img.shape[1], int(img.shape[2] * self.ratio), int(img.shape[3] * self.ratio)]).to(self.device)
        for k in range(self.ratio):
            for j in range(self.ratio):
                up_img[:, :, k::self.ratio, j::self.ratio] = img
        return up_img

    def forward(self, x):
        return self.downsampling(x)
    
    def H_pinv(self, y):
        return self.upsampling(y)

    def proj(self, x, y, alpha_obs=1.0):
        y = y * math.sqrt(alpha_obs)
        return x + self.upsampling(y - self.downsampling(x))
    
    def eq_var(self, var):
        return self.ratio ** 2 * var
    
    def get_type(self):
        return 'simple'

class Inpainting:
    def __init__(self, channels, img_dim, missing_r, device):
        self.channels = channels
        self.img_dim = img_dim
        indices = torch.zeros(img_dim**2)
        indices[missing_r] = 1
        self.mask = indices.reshape([img_dim, img_dim]).unsqueeze(0).unsqueeze(0).to(device)
    
    def forward(self, x):
        return x * (1-self.mask)

    def H_pinv(self, y):
        return y * (1-self.mask)

    def proj(self, x, y, alpha_obs=1.0):
        y = y * math.sqrt(alpha_obs)
        return x * self.mask + y * (1-self.mask)

    def eq_var(self, var):
        return var

    def get_type(self):
        return 'simple'

class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO = 3e-2):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        #ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
    
    def forward(self, x):
        return self.H(x)

    def proj(self, x, y, alpha_obs=1.0):
        return x + self.H_pinv(y - self.H(x)).view(y.shape[0], 3, x.shape[2], x.shape[3])
    
    def eq_var(self, var):
        print('This function should not be called')
        return

    def get_type(self):
        return 'SVD'

class PhaseRetrievalOperator:
    def __init__(self, oversample, device):
        # print(oversample)
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

    def H_pinv(self, x):
        x = ifft2_m(x).abs()
        x = self.undo_padding(x, self.pad, self.pad, self.pad, self.pad)
        return x
    
    def undo_padding(self, tensor, pad_left, pad_right, pad_top, pad_bottom):
        # Assuming 'tensor' is the 4D tensor
        # 'pad_left', 'pad_right', 'pad_top', 'pad_bottom' are the padding values
        if tensor.dim() != 4:
            raise ValueError("Input tensor should have 4 dimensions.")
        return tensor[:, :, pad_top : -pad_bottom, pad_left : -pad_right]

    def proj(self, x, y, alpha_obs=1.0):
        # print(self.pad)
        y = y * math.sqrt(alpha_obs)
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        fx = fft2_m(x_pad)
        # print(torch.min(fx.abs()))
        # fx_abs = fx.abs()
        # fx_abs[fx_abs<1e-5]=1e-5
        epsilon = 1e-8
        fx_prox = fx * y / (fx.abs() + epsilon)
        prox_x = ifft2_m(fx_prox)[:, :, self.pad:-self.pad, self.pad:-self.pad].real
        x = prox_x
        return prox_x

    def eq_var(self, var): 
        # print(256+2*self.pad)
        return var * (256+2*self.pad)**2/256**2

    def get_type(self):
        return 'simple'

class HDR(H_functions):
    def __init__(self):
        pass

    def forward(self, image):
        # Assert that image is in range [-1, 1]
        x = image
        x = torch.clip(x / 0.5, -1, 1)
        return x

    def H_pinv(self, x):
        return x * 0.5

    def proj(self, x, y, alpha_obs=1.0):
        # y = y * math.sqrt(alpha_obs)
        output = torch.zeros_like(x) + x
        # thre = alpha_obs.sqrt()
        thre = 1.0
        mask1 = torch.logical_and(torch.abs(y) >= thre, torch.abs(x) < thre/2)
        # mask1 = torch.logical_or(y > 2 * x, y < 2 * x)
        # print(mask1)
        if alpha_obs == 1.0:
            mask2 = torch.abs(y) < 1
        else:
            mask2 = torch.abs(y) < thre/2 # interesting
        output[mask1] = y[mask1] / 2
        output[mask2] = y[mask2] / 2
        return output

    def eq_var(self, var):
        return var / 4

    def get_type(self):
        return 'simple'