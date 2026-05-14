import os
import torch
import numpy as np
from PIL import Image as Image
import random
from torch.utils.data import Dataset

class Train_Dataset(Dataset):
    def __init__(self, image_dir, filelist, patch_size=(1024, 1024)):

        self.image_dir = image_dir

        self.file_list = os.path.join(self.image_dir, filelist)
        with open(self.file_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.patch_size = patch_size

    def get_images(self, index):
        low_img_name = self.input_names[index].replace('\n', '')
        
        low_img = Image.open(low_img_name)
                                        
        low_img = self.random_crop(np.asarray(low_img))
        
        data = {"low_img": low_img}

        return data

    def random_crop(self, img, start=None, patch_size=None):

        height, width = img.shape[:2]

        if patch_size is None:
            patch_size_h, patch_size_w = self.patch_size
        else:
            patch_size_h, patch_size_w = patch_size

        if start is None:
            x = random.randint(0, width - patch_size_w - 1)
            y = random.randint(0, height - patch_size_h - 1)
        else:
            x, y = start
        img_patch = img[y: y + patch_size_h, x: x + patch_size_w, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            img_patch = np.flip(img_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            img_patch = np.flip(img_patch, axis=0)
            
        img_patch = torch.tensor(img_patch.copy() / 255.0).permute(2, 0, 1).float()

        return img_patch

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


class Test_Dataset(Dataset):
    def __init__(self, image_dir, filelist):
        self.image_dir = image_dir

        self.file_list = os.path.join(self.image_dir, filelist)
        with open(self.file_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names

    def get_images(self, index):
        low_img_name = self.input_names[index].replace('\n', '')

        img_name = low_img_name.split('/')[-1]
        
        low_img = np.asarray(Image.open(low_img_name))

        low_img = torch.tensor(low_img / 255).permute(2, 0, 1).float()
            
        data = {"low_img": low_img, "img_name": img_name}

        return data

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)