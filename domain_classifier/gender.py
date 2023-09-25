import argparse
import os
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import clip 

DOMAINS = (
    "male",
    "female"
)

root_path = "/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples"


class TxtImg_Gender(datasets.ImageFolder):
    def __init__(
        self, 
        root,
        transform, 
        target_transform,
        cfg_w
    ):
        root = os.path.join(root, f"guide_w{cfg_w}")
        super().__init__(root, transform, target_transform)


def clip_predict(batch_size, cfg_w):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # prepare data
    txtimg_gen_dataset = TxtImg_Gender(root = root_path, transform = preprocess, target_transform = None, cfg_w = cfg_w)
    txtimg_gen_dataloader = DataLoader(txtimg_gen_dataset, batch_size = batch_size, shuffle=False, num_workers=4)
    domain_text = torch.cat([clip.tokenize(f"a photo of a {domain} person") for domain in DOMAINS]) # num_labels x token_length

    male_probs, female_probs = [], []
    for image, class_label in iter(txtimg_gen_dataloader):
        # Calculate features
        with torch.no_grad():
            # discrete domain classification
            logits_per_image, logits_per_text = model(image.to(device), domain_text.to(device))
            probs = logits_per_image.softmax(dim=-1).cpu().numpy() # batch_size x num_domains
            male_probs.append(probs[:, 0])
            female_probs.append(probs[:, 1])

            # similarity distance metric
            # image_features = model.encode_image(image.to(device))
            # text_features = model.encode_text(domain_text.to(device))
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            # similarity = (image_features @ text_features.T)
    
    male_probs = np.concatenate(male_probs)
    female_probs = np.concatenate(female_probs)


if __name__ == "__main__":
    clip_predict(128, 0.0)
