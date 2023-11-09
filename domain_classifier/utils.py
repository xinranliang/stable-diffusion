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

# 44 occupations
social_job_list = ["administrative assistant", "electrician", "author", "optician", "announcer", "chemist", "butcher", "building inspector", "bartender", "childcare worker", "chef", "CEO", "biologist", "bus driver", "crane operator", "drafter", "construction worker", "doctor", "custodian", "cook", "nurse practitioner", "mail carrier", "lab tech", "pharmacist", "librarian", "nurse", "housekeeper", "pilot", "roofer", "police officer", "public relations person", "customer service representative", "software developer", "special ed teacher", "receptionist", "plumber", "security guard", "technical writer", "telemarketer", "veterinarian"]

def get_job_prompt(job_name, prompt_date, gender_label=None):
    if prompt_date == "2023-10-12" or prompt_date == "2023-10-15":
        extend_description = [" single", " in the center"]
    elif prompt_date == "2023-10-26" or prompt_date == "2023-10-29":
        extend_description = [" single", ""]
    elif prompt_date == "2023-11-03" or prompt_date == "2023-11-04":
        extend_description = ["", " in the center"]
    elif prompt_date == "2023-11-05" or prompt_date == "2023-11-06":
        extend_description = ["", ""]
    else:
        raise ValueError("invalid experiments date")

    if gender_label is None:
        return "A photo of a{} {}{}.".format(extend_description[0], job_name.lower(), extend_description[1])
    else:
        assert gender_label == "male" or gender_label == "female", "unspecified gender label"
        return "A photo of a{}{} {}{}.".format(extend_description[0], f" {gender_label}", job_name.lower(), extend_description[1])

def get_generic_prompt(prompt_date, gender_label=None):
    if prompt_date == "2023-10-30":
        extend_description = ["", " in the center"]
    elif prompt_date == "2023-10-31":
        extend_description = ["", ""]
    elif prompt_date == "2023-11-01":
        extend_description = [" single", " in the center"]
    elif prompt_date == "2023-11-02":
        extend_description = [" single", ""]
    else:
        raise ValueError("invalid experiments date")

    if gender_label is None:
        return "A photo of a{} person{}.".format(extend_description[0], extend_description[1])
    else:
        assert gender_label == "male" or gender_label == "female", "unspecified gender label"
        return "A photo of a{}{} person{}.".format(extend_description[0], f" {gender_label}", extend_description[1])


class SimpleDataset(Dataset):
    """An simple image dataset for calculating inception score and FID."""

    def __init__(self, root, subset=None, exp_date="none", domain=None, exts=['png', 'jpg', 'JPEG'], transform=None, num_images=None):
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
            subset_dir = get_job_prompt(subset, exp_date, domain).lower().replace(" ", "_")
            root = os.path.join(root, subset_dir)
        if exp_date in ["2023-10-30", "2023-10-31", "2023-11-01", "2023-11-02"]:
            subset_dir = get_generic_prompt(exp_date, domain).lower().replace(" ", "_")
            root = os.path.join(root, subset_dir)
        for ext in exts:
            if domain is None:
                self.paths.extend(
                    list(glob(
                        os.path.join(root, '**/*.%s' % ext), recursive=True)))
            elif domain == "female":
                all_files = list(glob(os.path.join(root, '**/*.%s' % ext), recursive=True))
                filter_files = [each_file for each_file in all_files if "female" in each_file]
                self.paths.extend(filter_files)
            elif domain == "male":
                all_files = list(glob(os.path.join(root, '**/*.%s' % ext), recursive=True))
                filter_files = []
                for each_file in all_files:
                    if "male" in each_file and "female" not in each_file:
                        filter_files.append(each_file)
                self.paths.extend(filter_files)
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