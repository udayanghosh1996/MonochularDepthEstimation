import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
import numpy as np


class NYUDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset("sayakpaul/nyu_depth_v2", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]

        # rgb_image = np.array(example["image"]).transpose(1, 2, 0).astype('uint8')
        # depth_map = np.array(example["depth_map"]).squeeze()

        rgb_image = Image.fromarray((np.array(example["image"]) * 255).astype('uint8'))

        depth_map = Image.fromarray((np.array(example["depth_map"]) * 255).astype('uint8'), mode="L")




        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_map = self.transform(depth_map)

        return rgb_image, depth_map
