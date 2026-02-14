import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import math
class Diffusion(torch.nn.Module):
    def __init__(self, denoiser, timesteps=1000, sqrt_s=0.008):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        
        # t from 1 to T
        t = torch.arange(1, timesteps + 1, dtype=torch.float32)
        # 1 - sqrt(t / (T + s))
        self.register_buffer('alphas_cumprod', 1. - torch.sqrt(t / (timesteps + sqrt_s)))
        
        betas = []
        one_minus_beta = 0
        for i in range(timesteps):
            alpha_cumprod_t = self.alphas_cumprod[i]
            if i == 0:
                betas.append(1 - alpha_cumprod_t)
                one_minus_beta = alpha_cumprod_t
            else:
                temp = alpha_cumprod_t / one_minus_beta
                betas.append(1 - temp)
                one_minus_beta = alpha_cumprod_t
        
        self.register_buffer('betas', torch.stack(betas))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('sqrt_alphas', torch.sqrt(self.alphas))
        
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alphas_cumprod_prev', torch.sqrt(self.alphas_cumprod_prev))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cuda())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    # --- 2. Loss ---
    def loss(self, x_start, y, attention_mask):
        batch_size = x_start.shape[0]
        device = x_start.device
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = self.extract(torch.sqrt(self.alphas_cumprod), t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_noisy = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        
        predicted_x0 = self.denoiser(x_noisy, t + 1, y, attention_mask)
        
        loss = F.mse_loss(x_start, predicted_x0, reduction='none')
        loss = (loss * attention_mask.unsqueeze(-1)).mean()
        return loss

    # --- 3. Sample---
    @torch.no_grad()
    def sample(self, num_samples, max_len, embedding_dim, length_sampler, cond_configs):
        device = self.betas.device
        self.denoiser.eval()
        
        sampled_lengths = length_sampler.sample(num_samples)
        attention_mask = torch.zeros(num_samples, max_len, device=device)
        for i, length in enumerate(sampled_lengths):
            attention_mask[i, :length] = 1
            
        x_t = torch.randn(num_samples, max_len, embedding_dim, device=device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", leave=False):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            model_t = t + 1 
            
            # --- Dual Guidance ---
            # Unconditional (Label 3)
            uncond_id = 3 
            y_uncond = torch.full((num_samples,), uncond_id, device=device, dtype=torch.long)
            pred_x0_uncond = self.denoiser(x_t, model_t, y_uncond, attention_mask)
            
            # Add guidance from every condition
            guidance_delta = torch.zeros_like(pred_x0_uncond)
            for config in cond_configs:
                label_idx = config['label_idx']
                scale = config['scale']
                y_cond = torch.full((num_samples,), label_idx, device=device, dtype=torch.long)
                pred_x0_cond = self.denoiser(x_t, model_t, y_cond, attention_mask)
                guidance_delta += scale * (pred_x0_cond - pred_x0_uncond)
            
            # final prediction
            pred_x0 = guidance_delta + pred_x0_uncond
            
            if i == 0:
                x_t = pred_x0
                continue
                
            alpha_cumprod_t = self.extract(self.alphas_cumprod, t, x_t.shape)
            alpha_cumprod_prev_t = self.extract(self.alphas_cumprod_prev, t, x_t.shape)
            beta_t = self.extract(self.betas, t, x_t.shape)
            sqrt_alpha_t = self.extract(self.sqrt_alphas, t, x_t.shape)
            sqrt_alpha_cumprod_prev_t = self.extract(self.sqrt_alphas_cumprod_prev, t, x_t.shape)
            
            # miu = coef1 * pred_x0 + coef2 * x_t
            coef1 = sqrt_alpha_cumprod_prev_t * beta_t / (1. - alpha_cumprod_t)
            coef2 = sqrt_alpha_t * (1. - alpha_cumprod_prev_t) / (1. - alpha_cumprod_t)
            miu = coef1 * pred_x0 + coef2 * x_t
            
            sigma = (1. - alpha_cumprod_prev_t) / (1. - alpha_cumprod_t) * beta_t
            
            noise = torch.randn_like(x_t)
            log_sigma = torch.log(torch.clamp(sigma, min=1e-20))
            x_t = miu + torch.exp(0.5 * log_sigma) * noise
            
        return x_t, attention_mask