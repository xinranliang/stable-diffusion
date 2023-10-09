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
    "woman",
    "man"
)

root_path = "/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples"

w_lst = [0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]


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
    domain_text = torch.cat([clip.tokenize(f"a photo of a {domain}") for domain in DOMAINS]) # num_labels x token_length

    pred_labels = []
    for image, class_label in iter(txtimg_gen_dataloader):
        # Calculate features
        with torch.no_grad():
            # discrete domain classification
            logits_per_image, logits_per_text = model(image.to(device), domain_text.to(device))
            probs = logits_per_image.softmax(dim=-1).cpu().numpy() # batch_size x num_domains
            syn_pred_value = np.argmax(probs, axis=-1)
            pred_labels.append(syn_pred_value)

            # similarity distance metric
            # image_features = model.encode_image(image.to(device))
            # text_features = model.encode_text(domain_text.to(device))
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            # similarity = (image_features @ text_features.T)
    
    pred_labels = np.concatenate(pred_labels)
    num_female = sum(pred_labels == 0)
    num_male = sum(pred_labels == 1)

    return {"num_male": num_male, "num_female": num_female}


def main():
    for cfg_w in w_lst:
        return_dict = clip_predict(batch_size=128, cfg_w=cfg_w)
        print(f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}")
        print("portion of predicted female: {:03f}".format(return_dict["num_female"] / 1000))


if __name__ == "__main__":
    main()
