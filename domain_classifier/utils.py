import argparse
import os
from glob import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    """An simple image dataset for calculating inception score and FID."""

    def __init__(self, root, subset=None, exts=['png', 'jpg', 'JPEG'], transform=None, num_images=None):
        """Construct an image dataset.

        Args:
            root: Path to the image directory. This directory will be
                  recursively searched.
            subset: String name of subset samples directory. 
                    This directory will be recursively searched.
            exts: List of extensions to search for.
            transform: A torchvision transform to apply to the images. If
                       None, the images will be converted to tensors.
            num_images: The number of images to load. If None, all images
                        will be loaded.
        """
        self.paths = []
        self.transform = transform
        if subset is not None:
            root = os.path.join(root, subset)
        for ext in exts:
            self.paths.extend(
                list(glob(
                    os.path.join(root, '**/*.%s' % ext), recursive=True)))
        self.paths = self.paths[:num_images]

    def __len__(self):              # noqa
        return len(self.paths)

    def __getitem__(self, idx):     # noqa
        image = Image.open(self.paths[idx])
        image = image.convert('RGB')        # fix ImageNet grayscale images
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor(image)
        return image