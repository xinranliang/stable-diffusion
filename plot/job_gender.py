import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_job_gender(correct):
    ws = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], dtype=np.float64)

    counts = np.array([0.354, 0.358, 0.366, 0.381, 0.372, 0.369, 0.381, 0.384, 0.399], dtype=np.float64)
    if correct:
        coeff_lower = np.array([0.825, 0.611 + (0.825 - 0.611) / 4 * 3, 0.611 + (0.825 - 0.611) / 4 * 2, 0.611 + (0.825 - 0.611) / 4 * 1, 0.611, 0.611 + (0.826 - 0.611) / 4 * 1, 0.611 + (0.826 - 0.611) / 4 * 2, 0.611 + (0.826 - 0.611) / 4 * 3, 0.826], dtype=np.float64)
        coeff_upper = np.array([0.975, 0.833 + (0.975 - 0.833) / 4 * 3, 0.833 + (0.975 - 0.833) / 4 * 2, 0.833 + (0.975 - 0.833) / 4 * 1, 0.833, 0.833 + (1.000 - 0.833) / 4 * 1, 0.833 + (1.000 - 0.833) / 4 * 2, 0.833 + (1.000 - 0.833) / 4 * 3, 1.000], dtype=np.float64)
    
    counts_lower, counts_upper = counts * coeff_lower, counts * coeff_upper

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        plt.plot(ws, counts, linestyle="-", marker='o')
        plt.fill_between(ws, counts_upper, counts_lower, alpha=0.25)
        plt.xticks(ws, ws)
        plt.ylim(0.2, 0.61)
        plt.yticks(np.arange(0.2, 0.61, step=0.05))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Portion classified as Female samples")
        plt.title("Samples classified as Female w.r.t Guidance from stable-diffusion-v2")
    
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/figures/sb-v2-job-gender.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/figures/sb-v2-job-gender.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/figures", exist_ok=True)
    plot_job_gender(correct=True)