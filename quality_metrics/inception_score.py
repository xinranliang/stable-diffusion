import numpy as np
from glob import glob
from PIL import Image
import argparse
import os 
import subprocess

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models.inception import inception_v3
from torchvision import datasets, transforms

import torchmetrics
from torchmetrics.image.inception import InceptionScore

from utils import SimpleDataset, social_job_list, get_job_prompt

# guidance values
w_lst = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def run_script(folder, prompt_label, gender_label):
    if not gender_label:
        if prompt_label:
            for job_name in social_job_list:
                job_dir = get_job_prompt(job_name).lower().replace(" ", "_")
                subprocess.run(["echo", f"input text prompt: {job_dir}"])
                # print(f"input text prompt: {job_dir}")
                for cfg_w in w_lst:
                    subprocess.run(["echo", f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}"])
                    # print(f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}")
                    sys_comm = [
                        "fidelity", "--gpu", "0", "--isc", "--samples-find-deep", "--silent",
                        "--input1", f"{folder}/guide_w{cfg_w}/{job_dir}",
                        "--isc-splits", "5", "-b", "100"
                    ]
                    subprocess.run(sys_comm)
        
        else:
            for cfg_w in w_lst:
                subprocess.run(["echo", f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}"])
                # print(f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}")
                sys_comm = [
                    "fidelity", "--gpu", "0", "--isc", "--samples-find-deep", "--silent",
                    "--input1", f"{folder}/guide_w{cfg_w}",
                    "--isc-splits", "5"
                ]
                subprocess.run(sys_comm)
    
    else:
        if prompt_label:
            for job_name in social_job_list:
                for gender_name in ["male", "female"]:
                    job_dir = get_job_prompt(job_name, gender_name).lower().replace(" ", "_")
                    subprocess.run(["echo", f"input text prompt: {job_dir}"])
                    # print(f"input text prompt: {job_dir}")
                    for cfg_w in w_lst:
                        subprocess.run(["echo", f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}"])
                        # print(f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}")
                        sys_comm = [
                            "fidelity", "--gpu", "0", "--isc", "--samples-find-deep", "--silent",
                            "--input1", f"{folder}/guide_w{cfg_w}/{job_dir}",
                            "--isc-splits", "5", "-b", "100"
                        ]
                        subprocess.run(sys_comm)
        
        else:
            for cfg_w in w_lst:
                subprocess.run(["echo", f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}"])
                # print(f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}")
                sys_comm = [
                    "fidelity", "--gpu", "0", "--isc", "--samples-find-deep", "--silent",
                    "--input1", f"{folder}/guide_w{cfg_w}",
                    "--isc-splits", "5"
                ]
                subprocess.run(sys_comm)


def compute_is_metrics(folder, prompt_label, gender_label):
    inception_metric = InceptionScore(normalize=True, splits=1)

    if not gender_label:
        if prompt_label:
            for job_name in social_job_list:
                print(f"input text prompt: {job_dir}")
                score_list = []

                for cfg_w in w_lst:
                    img_dataset = SimpleDataset(root=f"{folder}/guide_w{cfg_w}", subset=job_name)
                    img_dataload = DataLoader(img_dataset, batch_size=len(img_dataset)//10, shuffle=True, num_workers=4)
                    for img_tensor, img_path in iter(img_dataload):
                        print(img_tensor.shape)
                        inception_metric.update(img_tensor)
                        avg, std = inception_metric.compute()
                        score_list.append(avg)
                score_list = np.array(score_list, dtype=np.float64)
                score_list_avg, score_list_std = np.mean(score_list), np.std(score_list)
                print("IS mean: {} and std: {}".format(score_list_avg, score_list_std))
        
        else:
            score_list = []
            for cfg_w in w_lst:
                img_dataset = SimpleDataset(root=f"{folder}/guide_w{cfg_w}")
                img_dataload = DataLoader(img_dataset, batch_size=len(img_dataset)//10, shuffle=True, num_workers=4)
                for img_tensor, img_path in iter(img_dataload):
                    print(img_tensor.shape)
                    inception_metric.update(img_tensor)
                    avg, std = inception_metric.compute()
                    score_list.append(avg)
            score_list = np.array(score_list, dtype=np.float64)
            score_list_avg, score_list_std = np.mean(score_list), np.std(score_list)
            print("IS mean: {} and std: {}".format(score_list_avg, score_list_std))
    
    else:
        if prompt_label:
            for job_name in social_job_list:
                for gender_name in ["male", "female"]:
                    print(f"input text prompt: {job_dir}")
                    score_list = []
                    for cfg_w in w_lst:
                        img_dataset = SimpleDataset(root=f"{folder}/guide_w{cfg_w}", subset=job_name)
                        img_dataload = DataLoader(img_dataset, batch_size=len(img_dataset)//10, shuffle=True, num_workers=4)
                        for img_tensor, img_path in iter(img_dataload):
                            print(img_tensor.shape)
                            inception_metric.update(img_tensor)
                            avg, std = inception_metric.compute()
                            score_list.append(avg)
                    score_list = np.array(score_list, dtype=np.float64)
                    score_list_avg, score_list_std = np.mean(score_list), np.std(score_list)
                    print("IS mean: {} and std: {}".format(score_list_avg, score_list_std))
        
        else:
            score_list = []
            for cfg_w in w_lst:
                print(f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}")
                img_dataset = SimpleDataset(root=f"{folder}/guide_w{cfg_w}")
                img_dataload = DataLoader(img_dataset, batch_size=len(img_dataset), shuffle=True, num_workers=4)
                img_tensor = next(iter(img_dataload))
                for img_tensor, img_path in iter(img_dataload):
                    print(img_tensor.shape)
                    inception_metric.update(img_tensor)
                    avg, std = inception_metric.compute()
                    score_list.append(avg)
                score_list = np.array(score_list, dtype=np.float64)
                score_list_avg, score_list_std = np.mean(score_list), np.std(score_list)
                print("IS mean: {} and std: {}".format(score_list_avg, score_list_std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--master-folder", type=str, help="path to master folder, not including cfg_w")
    parser.add_argument("--per-job", action="store_true", help="whether to evaluate entire sample sets or decompose by jobs")
    parser.add_argument("--extend-gender", action="store_true", help="whether to evaluate on sub-group level with extended prompt")

    opt = parser.parse_args()
    
    # run_script(opt.master_folder, opt.per_job, opt.extend_gender)
    compute_is_metrics(opt.master_folder, opt.per_job, opt.extend_gender)