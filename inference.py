import os
from models import Unet, GaussianDiffusion
from pathlib import Path
import torch
from dataset.dataloader import Test_Dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs

class inference:
    def __init__(
            self,
            diffusion_model,
            data_dir,
            val_dataset,
            *,
            batch_size=128,
            ema_update_every=10,
            ema_decay=0.995,
            results_folder='./results',
            ckpt_path='ckpt/stage2',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
    ):
        super().__init__()

        # accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            kwargs_handlers=[ddp_kwargs]
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.batch_size = batch_size
        self.dir = data_dir
        self.va_dataset = val_dataset

        self.ckpt_path = ckpt_path

        self.val_ds = Test_Dataset(image_dir=self.dir, 
                                   filelist='{}.txt'.format(self.va_dataset))

        val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                            pin_memory=True, num_workers=8)
        
        self.val_dl = self.accelerator.prepare(val_dl)
        
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)

    @property
    def device(self):
        return self.accelerator.device

    def load(self):

        device = self.accelerator.device
        data = torch.load(os.path.join(self.ckpt_path), map_location=device)

        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

    def infer(self):
        self.load()
        
        self.ema.ema_model.eval()
                
        with torch.inference_mode():
            for i, data_batch in enumerate(self.val_dl):
                
                img_name = data_batch["img_name"][-1] 
                
                pred_img = self.ema.ema_model.sample(data_batch, return_all_timesteps=False)
    
                pred_img = torch.clip(pred_img * 255.0, 0, 255.0).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')
                pred_img = Image.fromarray(pred_img)
                pred_img.save(os.path.join(self.results_folder, img_name))
                

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
    
    dataset = 'LOL'

    inference = inference(
        diffusion,
        data_dir='/data/datasets',
        val_dataset='{}_val'.format(dataset),
        results_folder='./infer_{}'.format(dataset),
        ckpt_path='ckpt/stage2/test_model_under.pt',
        batch_size=1
    )
    
    inference.infer()