import os
import math
from pathlib import Path
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from dataset.dataloader import Train_Dataset, Test_Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import random
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from models.attend import Attend
from models.version import __version__
from models.stage1_net import Net
import lpips
import torchvision.models as models

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, layer_index=16): 
        super().__init__()
        vgg16_features = models.vgg16(pretrained=True).features

        self.block = vgg16_features[:layer_index]
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.block(x) 

def sobel_mag(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

def edge_aware_tv(x, guide, w=20., r=7):
    g = sobel_mag(guide.mean(dim=1, keepdim=True))
    w_edge = torch.exp(-w * g)
    
    diff_h = (x[..., :, 1:] - x[..., :, :-1]).abs()
    diff_v = (x[..., 1:, :] - x[..., :-1, :]).abs()
    
    tv_h = (w_edge[..., :, :-1] * diff_h).mean()
    tv_v = (w_edge[..., :-1, :] * diff_v).mean()
    
    return tv_h + tv_v

def data_transform(X):
    return 2 * X - 1.0

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class Block(Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

class Unet(Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            sinusoidal_pos_emb_theta=10000,
            attn_dim_head=32,
            attn_heads=4,
            full_attn=None,  # defaults to full attention only for inner most layer
            flash_attn=False
    ):
        super().__init__()

        self.channels = channels

        input_channels = channels * 2

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_cond):
        assert all([divisible_by(d, self.downsample_factor) 
                    for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        x = torch.cat((x_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
            self,
            model,
            *,
            timesteps=1000,
            sampling_timesteps=None,
            beta_schedule='sigmoid',
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.,
            auto_normalize=True,
            offset_noise_strength=0.,
            min_snr_loss_weight=True,
            min_snr_gamma=5,
            pseudo_t_min=0,
            pseudo_t_max=50,
            delta_t_max=950,
            recon_use_random_eps=True,
            pdc_weight=1.0,
            tv_weight=3.0,
            color_weight=0.5
    ):

        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels

        self.stage1_Net = Net()

        self.stage1_Net.eval()
        for para in self.stage1_Net.parameters():
            para.requires_grad = False

        self.pseudo_t_min = pseudo_t_min
        self.pseudo_t_max = pseudo_t_max
        self.delta_t_max = delta_t_max
        self.recon_use_random_eps = recon_use_random_eps

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.offset_noise_strength = offset_noise_strength
        
        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        register_buffer('loss_weight', maybe_clipped_snr)

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        
        self.pdc_weight = pdc_weight
        self.tv_weight = tv_weight
        self.color_weight = color_weight

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t, x_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, clip_denoised=True):
        preds = self.model_predictions(x, t, x_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def ddim_sample(self, x_cond, return_all_timesteps=False):

        bs, c, h, w = x_cond.shape

        batch, device, total_timesteps, sampling_timesteps, eta = \
            bs, self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        t_start = self.num_timesteps - 1
        img = torch.randn((bs, 3, h, w), device=device)

        times = torch.linspace(-1, t_start, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        imgs = [img]

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond,
                                                             clip_x_start=True, rederive_pred_noise=True)
            
            pred_noise = pred_noise

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, data_batch, return_all_timesteps=False):

        low_img = data_batch["low_img"]

        high_img_pseudo = self.stage1_Net(low_img)

        x_cond = self.normalize(high_img_pseudo)
        pred_img = self.ddim_sample(x_cond, return_all_timesteps=return_all_timesteps)

        return pred_img
            
    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
    @autocast(enabled=False)
    def q_sample_from_xt(self, x_t, t, t_next, noise=None):
        assert torch.all(t_next >= t), "t_next >= t"
        noise = default(noise, lambda: torch.randn_like(x_t))

        sqrt_ab_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_ab_tnext = extract(self.sqrt_alphas_cumprod, t_next, x_t.shape)

        ratio = sqrt_ab_tnext / sqrt_ab_t
        sigma = torch.sqrt(torch.clamp(1. - ratio ** 2, min=0.0))

        return ratio * x_t + sigma * noise
    
    def white_balance(self, input):
        r_channel, b_channel, g_channel = input[:,0:1], input[:,1:2], input[:,2:3]
        rb_mean = (r_channel + b_channel) / 2
        g_channel_new = torch.minimum(g_channel, 1.2 * rb_mean)
        out = torch.cat([r_channel, g_channel_new, b_channel], dim=1)
        return out

    def forward(self, data_batch, vgg=None, noise=None, offset_noise_strength=None):

        low_img = data_batch["low_img"]

        high_img_pseudo = self.stage1_Net(low_img)

        b, _, _, _ = high_img_pseudo.shape
        device = high_img_pseudo.device

        x_tstar = self.normalize(high_img_pseudo)
        x_cond  = x_tstar

        t_star = torch.randint(0, self.pseudo_t_max + 1, (b,), device=device).long()
        delta  = torch.randint(1, self.delta_t_max + 1,  (b,), device=device).long()
        t_in   = torch.clamp(t_star + delta, max=self.num_timesteps - 1)

        noise_in = torch.randn_like(x_tstar)
        x_in = self.q_sample_from_xt(x_tstar, t_star, t_in, noise_in)

        x0_pred = self.model(x_in, t_in, x_cond)
        
        x0_pred_unnorm = self.unnormalize(x0_pred).clamp(0, 1)

        sqrt_ab_tin = extract(self.sqrt_alphas_cumprod, t_in, x_in.shape)
        sqrt_1m_ab_tin = extract(self.sqrt_one_minus_alphas_cumprod, t_in,   x_in.shape)
        eps_implied = (x_in - sqrt_ab_tin * x0_pred) / sqrt_1m_ab_tin.clamp_min(1e-8)

        sqrt_ab_tstar = extract(self.sqrt_alphas_cumprod, t_star, x_tstar.shape)
        sqrt_1m_ab_tstar = extract(self.sqrt_one_minus_alphas_cumprod, t_star, x_tstar.shape)
        x_tstar_pred = sqrt_ab_tstar * x0_pred + sqrt_1m_ab_tstar * eps_implied

        diff_loss = F.mse_loss(x_tstar_pred, x_tstar)

        eps_recon = (torch.randn_like(x_tstar) if self.recon_use_random_eps else torch.zeros_like(x_tstar))
        x_recon_tstar = sqrt_ab_tstar * x0_pred + sqrt_1m_ab_tstar * eps_recon
        x_recon_tstar_unnorm = self.unnormalize(x_recon_tstar).clamp(0, 1)
        
        pdc_loss = self.pdc_weight * F.l1_loss(vgg(x_recon_tstar_unnorm), vgg(self.white_balance(high_img_pseudo)))
        
        aux_loss = self.tv_weight * edge_aware_tv(x0_pred_unnorm, guide=low_img) + \
            self.color_weight * self.gray_world_loss(x0_pred_unnorm)

        return diff_loss, pdc_loss, aux_loss


class Trainer:
    def __init__(
            self,
            diffusion_model,
            data_dir,
            train_dataset,
            val_dataset,
            *,
            train_batch_size=16,
            patch_size=[128, 128],
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            ckpt_path='ckpt/stage2',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            max_grad_norm=1.
    ):
        super().__init__()

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            kwargs_handlers=[ddp_kwargs]
        )

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        self.dir = data_dir
        self.train_dataset = train_dataset
        self.va_dataset = val_dataset
        self.patch_size = patch_size

        self.ckpt_path = ckpt_path

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        self.max_grad_norm = max_grad_norm

        self.train_ds = Train_Dataset(image_dir=self.dir,
                                      filelist='{}.txt'.format(self.train_dataset),
                                      patch_size=self.patch_size)
        self.val_ds = Test_Dataset(image_dir=self.dir,
                                   filelist='{}.txt'.format(self.va_dataset))

        train_dl = DataLoader(self.train_ds, batch_size=train_batch_size, shuffle=True,
                              pin_memory=True, num_workers=8)
        val_dl = DataLoader(self.val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=8)

        train_dl, val_dl = self.accelerator.prepare(train_dl, val_dl)

        self.train_dl = cycle(train_dl)
        self.val_dl = val_dl

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        
        self.loss_fn_alex = lpips.LPIPS(net='alex').eval().to(self.model.device)
        
        self.vgg = VGG16FeatureExtractor().eval().to(self.model.device)
        
    @property
    def device(self):
        return self.accelerator.device

    def save(self, save_path, save_name):

        os.makedirs(save_path, exist_ok=True)

        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, os.path.join(save_path, '{}.pt'.format(save_name)))

    def load(self, ckpt_path):
        device = self.accelerator.device
        data = torch.load(os.path.join(ckpt_path), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):

        accelerator = self.accelerator
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process,
                  ncols=100) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_dl)

                    with self.accelerator.autocast():
                        diff_loss, pdc_loss, aux_loss = self.model(data, vgg=self.vgg)
                        loss = diff_loss + pdc_loss + aux_loss
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description('diff:{:.4f} pdc:{:.4f} aux:{:.4f}'.format(diff_loss.item(), pdc_loss.item(), aux_loss.item()))

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.model.eval()
                    
                        self.save(self.ckpt_path, 'model_latest')

                        with torch.inference_mode():
                            ema_model = self.ema.ema_model
                            ema_model.eval()
                            for i, data_batch in enumerate(self.val_dl):

                                img_name = data_batch["img_name"][-1]

                                pred_img = ema_model.sample(data_batch, return_all_timesteps=False)
                                pred_img = torch.clip(pred_img * 255.0, 0, 255.0).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')

                                pred_img = Image.fromarray(pred_img)
                                pred_img.save(os.path.join(self.results_folder, img_name))

                pbar.update(1)

        accelerator.print('training complete')
