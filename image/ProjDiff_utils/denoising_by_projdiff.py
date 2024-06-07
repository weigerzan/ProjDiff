import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os
import numpy as np

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, lr, N, cls_fn=None, classes=None):
    # torch.cuda.empty_cache()
    with torch.no_grad():
        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        
        #setup iteration variables
        # x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]


        t = (torch.ones(n) * seq[-1]).to(x.device)
        at = compute_alpha(b, t.long())
        noise = torch.randn_like(x)
        x_T = noise * (1 - at).sqrt()
        et = model(x_T, t)
        if et.size(1) == 6:
            et = et[:, :3]
        x0_t = (x_T - et * (1 - at).sqrt()) / at.sqrt()
        v = None
        beta=0.0
        et = None
        # init_noise = torch.randn_like(x0_t)
        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            for _ in range(N):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                xt = at.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at).sqrt()
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
                
                if et.size(1) == 6:
                    et = et[:, :3]
                x0_t_new = (xt - et * (1 - at).sqrt()) / at.sqrt()
                # stochastic gradient
                diff = x0_t_new - x0_t
                d = diff
                if v is None:
                    v = d
                else:
                    v = beta * v + (1-beta) * d
                # print(v)
                # print(lr)
                x0_t += lr * v
                x0_t = H_funcs.proj(x0_t, y_0)
                # random_noise = torch.randn_like(x0_t)
                xt_next = x0_t
                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))

    return xs, x0_preds

# It is quite interesting that I found after the submission that this function was used for the noise-free phase retrieval task. Note that alpha_obs is always 1, thus this function is equalivant to the above one except 
# it performs the projection operation twice, which may yield better accuracy. Therefore, I have decided to keep this function here. One can switch to the above function and get a slightly lower results (PSNR 30~31).
def efficient_generalized_steps_phase(x, seq, model, b, H_funcs, y_0, sigma_0, lr, N, cls_fn=None, classes=None):
    with torch.no_grad():
        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        
        #setup iteration variables
        # x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]


        t = (torch.ones(n) * seq[-1]).to(x.device)
        at = compute_alpha(b, t.long())
        noise = torch.randn_like(x)
        x_T = noise * (1 - at).sqrt()
        et = model(x_T, t)
        if et.size(1) == 6:
            et = et[:, :3]
        x0_t = (x_T - et * (1 - at).sqrt()) / at.sqrt()
        v = None
        beta=0.0
        et = None
        init_noise = torch.randn_like(x0_t)
        # alpha_obs is always 1
        alpha_obs=torch.tensor(1)
        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            for _ in range(N):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                # add noise
                x_obs_t = alpha_obs.sqrt() * x0_t + (1-alpha_obs).sqrt() * torch.randn_like(x0_t)
                x_obs_t = H_funcs.proj(x_obs_t, y_0, alpha_obs)
                if at[0,0,0,0] <= alpha_obs:
                    xt = (at/alpha_obs).sqrt() * x_obs_t + (1-at/alpha_obs).sqrt() * torch.randn_like(x0_t)
                else:
                    xt = at.sqrt() * x0_t + (1-at).sqrt() * (x_obs_t - alpha_obs.sqrt() * x0_t) / (1-alpha_obs).sqrt()
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
                
                if et.size(1) == 6:
                    et = et[:, :3]
                
                x0_t_new = (xt - et * (1 - at).sqrt()) / at.sqrt()
                diff = x0_t_new - x0_t
                d = diff
                if v is None:
                    v = d
                else:
                    v = beta * v + (1-beta) * d
                x0_t_last = x0_t
                x0_t += lr * v
                x0_t = H_funcs.proj(x0_t, y_0, alpha_obs)

                xt_next = x0_t
                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
    return xs, x0_preds


def efficient_generalized_steps_noisy(x, seq, model, b, H_funcs, y_0, sigma_0, lr, N, cls_fn=None, classes=None):
    with torch.no_grad():
        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        var_obs = H_funcs.eq_var(sigma_0 ** 2)
        alpha_obs = 1 / torch.tensor(1+var_obs)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]


        t = (torch.ones(n) * seq[-1]).to(x.device)
        at = compute_alpha(b, t.long())
        noise = torch.randn_like(x)
        x_T = noise * (1 - at).sqrt()
        et = model(x_T, t)
        if et.size(1) == 6:
            et = et[:, :3]
        # x_obs_t = (x_T - et * (1 - at/alpha_obs).sqrt()) / (at/alpha_obs).sqrt()
        x0_t = (x_T - et * (1 - at).sqrt()) / at.sqrt()
        x_obs_t = alpha_obs.sqrt() * x0_t + (1-alpha_obs).sqrt() * torch.randn_like(x0_t)
        # x_obs_t = alpha_obs.sqrt() * x0_t + (1-alpha_obs).sqrt() * noise
        xt = x_T
        v = None
        beta=0.0
        lr_obs=1.0
        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            for _ in range(N):
                # print(x_obs_t)
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                if at[0,0,0,0] <= alpha_obs:
                    noise = torch.randn_like(x0_t)
                    xt = (at/alpha_obs).sqrt() * x_obs_t + (1-at/alpha_obs).sqrt() * noise
                    et = model(xt, t)
                    if et.size(1) == 6:
                        et = et[:, :3]
                    x0_t_new = x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    x_obs_t_new = alpha_obs.sqrt() * x0_t_new + (1-alpha_obs).sqrt() * torch.randn_like(x0_t_new)
                else:
                    sigma_t_tilde = 0
                    xt = at.sqrt() * x0_t + (1-at - sigma_t_tilde**2).sqrt() * (x_obs_t - alpha_obs.sqrt() * x0_t) / (1-alpha_obs).sqrt()
                    et = model(xt, t)
                    if et.size(1) == 6:
                        et = et[:, :3]
                    x0_t_new = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    x_obs_t_new = x_obs_t
                x0_t += lr * (x0_t_new - x0_t)
                x_obs_t += lr_obs * (x_obs_t_new - x_obs_t)
                if at[0,0,0,0] <= alpha_obs:
                    x_obs_t = H_funcs.proj(x_obs_t, y_0, alpha_obs)
                
                xt_next = x0_t


                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
    return xs, x0_preds



def efficient_generalized_steps_noisy_SVD(x, seq, model, b, H_funcs, y_0, sigma_0, lr, N, cls_fn=None, classes=None):
    with torch.no_grad():
        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        
        #setup iteration variables
        singulars = H_funcs.singulars()
        # print(singulars.shape)
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        alpha_obs = torch.ones_like(Sigma)
        # alpha_obs = torch.zeros_like(Sigma) 
        alpha_obs[Sigma > 0] = 1 / (1 + (sigma_0 / Sigma[Sigma > 0])**2).unsqueeze(0)
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]] * alpha_obs.sqrt()
        # print(Sig_inv_U_t_y.shape)
        alpha_obs = alpha_obs.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
        Sig_inv_U_t_y = Sig_inv_U_t_y.view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
        Sigma = Sigma.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
        # print(torch.sum(Sigma==0))
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]


        t = (torch.ones(n) * seq[-1]).to(x.device)
        at = compute_alpha(b, t.long())
        noise = torch.randn_like(x)
        x_T = noise * (1 - at).sqrt()
        et = model(x_T, t)
        if et.size(1) == 6:
            et = et[:, :3]
        x0_t = (x_T - et * (1 - at).sqrt()) / at.sqrt()
        V_t_x0 = H_funcs.Vt(x0_t).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
        V_t_x_obs = alpha_obs.sqrt() * V_t_x0 + (1-alpha_obs).sqrt() * torch.randn_like(V_t_x0)
        x_obs_t = H_funcs.V(V_t_x_obs.view([V_t_x_obs.shape[0], -1])).view(x.shape)
        
        # print(x0_t)
        # print(y_upsampling)
        v = None
        beta=0.0
        lr_obs = 1.0
        init_noise = torch.randn_like(x0_t)
        et = None
        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            for _ in range(N):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                V_t_x0 = H_funcs.Vt(x0_t).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
                V_t_x_obs = H_funcs.Vt(x_obs_t).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
                smaller_idx = (alpha_obs < at[0,0,0,0])
                # print(smaller_idx)
                larger_idx = (alpha_obs >= at[0,0,0,0])
                V_t_x_t = torch.zeros_like(V_t_x_obs)
                V_t_x_t[larger_idx] = (at[0,0,0,0]/alpha_obs[larger_idx]).sqrt() * V_t_x_obs[larger_idx] + (1-at[0,0,0,0]/alpha_obs[larger_idx]).sqrt() * torch.randn_like(V_t_x_obs[larger_idx])
                V_t_x_t[smaller_idx] = at[0,0,0,0].sqrt() * V_t_x0[smaller_idx] + (1-at[0,0,0,0]).sqrt() * (V_t_x_obs[smaller_idx] - V_t_x0[smaller_idx] * alpha_obs[smaller_idx].sqrt())/(1-alpha_obs[smaller_idx]).sqrt()
                xt = H_funcs.V(V_t_x_t.view([V_t_x_t.shape[0], -1])).view(x.shape)
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
                
                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t_new = (xt - et * (1 - at).sqrt()) / at.sqrt()

                V_t_x0_new = H_funcs.Vt(x0_t_new).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
                V_t_x_obs_new = alpha_obs.sqrt() * V_t_x0_new + (1-alpha_obs).sqrt() * torch.randn_like(V_t_x0_new)
                V_t_x0[larger_idx] = V_t_x0_new[larger_idx]
                V_t_x_obs_new[smaller_idx] = V_t_x_obs[smaller_idx]
                x0_t = H_funcs.V(V_t_x0.view([V_t_x0.shape[0], -1])).view(x.shape)
                x_obs_t_new = H_funcs.V(V_t_x_obs_new.view([V_t_x_obs_new.shape[0], -1])).view(x.shape)
                x0_t += lr * (x0_t_new - x0_t)
                x_obs_t += lr_obs * (x_obs_t_new - x_obs_t)
                V_t_x_obs = H_funcs.Vt(x_obs_t).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
                V_t_x_obs[Sigma > 0] = Sig_inv_U_t_y[Sigma > 0]
                x_obs_t = H_funcs.V(V_t_x_obs.view([V_t_x_obs.shape[0], -1])).view(x.shape)

                xt_next = x0_t
                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))

    return xs, x0_preds