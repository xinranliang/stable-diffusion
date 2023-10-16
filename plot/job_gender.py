import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_job_gender_special(correct):
    ws = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    job_names = ["author", "lab tech", "pharmacist",  "public relations person", "veterinarian"]
    colors = ["blue", "orange", "green", "purple", "brown"]

    cfgw_counts = [
        np.array([0.377, 0.396, 0.403, 0.409, 0.413, 0.416], dtype=np.float64), # overall average
        np.array([0.564, 0.602, 0.62, 0.618, 0.62, 0.648], dtype=np.float64),
        np.array([0.566, 0.674, 0.724, 0.804, 0.766, 0.814], dtype=np.float64),
        np.array([0.576, 0.66, 0.72, 0.756, 0.774, 0.808], dtype=np.float64),
        np.array([0.564, 0.666, 0.682, 0.738, 0.752, 0.768], dtype=np.float64),
        np.array([0.508, 0.606, 0.694, 0.724, 0.742, 0.742], dtype=np.float64),
    ]
    base_counts = np.array([0.376, 0.392, 0.33, 0.482, 0.452, 0.456], dtype=np.float64) # cfg_w = 0.0
    if correct:
        coeff_lower = np.array([0.389, 0.672, 0.955, 0.711, 0.692, 0.674], dtype=np.float64)
        coeff_upper = np.array([1.056, 1.096, 1.136, 1.079, 1.116, 1.152], dtype=np.float64)
    
    cfgw_counts_lower, cfgw_counts_upper = [], []
    for idx in range(len(cfgw_counts)):
        cfgw_counts_lower.append(cfgw_counts[idx] * coeff_lower)
        cfgw_counts_upper.append(cfgw_counts[idx] * coeff_upper)

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        plt.plot(ws, cfgw_counts[0], linestyle="-", marker='o', label="overall average", color="black")
        plt.fill_between(ws, cfgw_counts_upper[0], cfgw_counts_lower[0], alpha=0.2, color="black")
        plt.axhline(base_counts[0], linestyle="--", color="black")
        for idx in range(len(job_names)):
            plt.plot(ws, cfgw_counts[idx + 1], linestyle="-", marker='o', label=job_names[idx], color=colors[idx])
            plt.fill_between(ws, cfgw_counts_upper[idx + 1], cfgw_counts_lower[idx + 1], alpha=0.1, color=colors[idx])
            plt.axhline(base_counts[idx + 1], linestyle="--", color=colors[idx])
        plt.xticks(ws, ws)
        plt.ylim(0.1, 1.01)
        plt.yticks(np.arange(0.1, 1.01, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Portion classified as Female samples")
        plt.title("Sampling distribution w.r.t Guidance from stable-diffusion-v2")
        plt.legend(ncols=3, loc="lower center")
    
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12/figures/sb-v2-job-gender.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12/figures/sb-v2-job-gender.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_job_gender_common(correct):
    ws = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    job_names = ["childcare worker", "nurse", "CSR", "doctor", "technical writer", "software developer"]
    colors = ["blue", "orange", "green", "purple", "brown", "red"]

    cfgw_counts = [
        np.array([0.922, 0.964, 0.974, 0.974, 0.972, 0.99], dtype=np.float64),
        np.array([0.892, 0.948, 0.97, 0.978, 0.982, 0.982], dtype=np.float64),
        np.array([0.792, 0.894, 0.94, 0.95, 0.976, 0.976], dtype=np.float64),
        np.array([0.288, 0.232, 0.236, 0.242, 0.262, 0.264], dtype=np.float64),
        np.array([0.35, 0.348, 0.282, 0.274, 0.292, 0.298], dtype=np.float64),
        np.array([0.07, 0.03, 0.022, 0.006, 0.012, 0.014], dtype=np.float64),
    ]
    base_counts = np.array([0.716, 0.672, 0.528, 0.346, 0.38, 0.218], dtype=np.float64) # cfg_w = 0.0
    if correct:
        coeff_lower = np.array([0.389, 0.672, 0.955, 0.711, 0.692, 0.674], dtype=np.float64)
        coeff_upper = np.array([1.056, 1.096, 1.136, 1.079, 1.116, 1.152], dtype=np.float64)
    
    cfgw_counts_lower, cfgw_counts_upper = [], []
    for idx in range(len(cfgw_counts)):
        cfgw_counts_lower.append(cfgw_counts[idx] * coeff_lower)
        cfgw_counts_upper.append(cfgw_counts[idx] * coeff_upper)

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(job_names)):
            plt.plot(ws, cfgw_counts[idx], linestyle="-", marker='o', label=job_names[idx], color=colors[idx])
            plt.fill_between(ws, cfgw_counts_upper[idx], cfgw_counts_lower[idx], alpha=0.2, color=colors[idx])
            plt.axhline(base_counts[idx], linestyle="--", color=colors[idx])
        plt.xticks(ws, ws)
        plt.ylim(0.0, 1.21)
        plt.yticks(np.arange(0.0, 1.21, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Portion classified as Female samples")
        plt.title("Sampling distribution w.r.t Guidance from stable-diffusion-v2")
        plt.legend(ncols=3)
    
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12/figures/sb-v2-job-gender-amplify.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12/figures/sb-v2-job-gender-amplify.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12/figures", exist_ok=True)
    # plot_job_gender_special(correct=True)
    plot_job_gender_common(correct=True)