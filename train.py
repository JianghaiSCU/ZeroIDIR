import os
from models import Unet, GaussianDiffusion, Trainer

if __name__ == "__main__":
    model = Unet(
        channels=3,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    )

    diffusion = GaussianDiffusion(
        model,
        timesteps=1000,
        sampling_timesteps=20
    )

    trainer = Trainer(
        diffusion,
        data_dir='/data/datasets',
        train_dataset='train_data_under',
        val_dataset='LOL_val',
        results_folder='./valid/LOL',
        ckpt_path='ckpt/stage2',
        patch_size=[256, 256],
        train_batch_size=4,
        train_lr=8e-5,
        save_and_sample_every=2000,
        train_num_steps=1000000,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False
    )

    trainer.train()
    