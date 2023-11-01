import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_repr_special(correct):
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
        true_repr = np.array([25, 29, 46, 36, 38, 41], dtype=np.float64) / 100
        pred_repr = np.array([36, 36, 46, 38, 40, 48], dtype=np.float64) / 100
        unsure_repr = np.array([23, 14, 7, 13, 10, 17], dtype=np.float64) / 100

        coeff_lower = pred_repr - (true_repr - unsure_repr / 2)
        coeff_upper = (true_repr + unsure_repr / 2) - pred_repr
    
    cfgw_counts_lower, cfgw_counts_upper = [], []
    for idx in range(len(cfgw_counts)):
        cfgw_counts_lower.append(cfgw_counts[idx] - coeff_lower)
        cfgw_counts_upper.append(cfgw_counts[idx] + coeff_upper)

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        plt.plot(ws, cfgw_counts[0], linestyle="-", marker='o', label="overall average", color="black")
        plt.fill_between(ws, cfgw_counts_upper[0], cfgw_counts_lower[0], alpha=0.2, color="black")
        plt.axhline(base_counts[0], linestyle="--", color="black")
        for idx in range(len(job_names)):
            plt.plot(ws, cfgw_counts[idx + 1], linestyle="-", marker='o', label=job_names[idx], color=colors[idx])
            plt.fill_between(ws, cfgw_counts_upper[idx + 1], cfgw_counts_lower[idx + 1], alpha=0.2, color=colors[idx])
            plt.axhline(base_counts[idx + 1], linestyle="--", color=colors[idx])
        plt.xticks(ws, ws)
        plt.ylim(0.1, 1.01)
        plt.yticks(np.arange(0.1, 1.01, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Portion classified as Female samples")
        plt.title("Sampling distribution w.r.t Guidance from stable-diffusion-v2")
        plt.legend(ncols=3)
    
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12/figures/sb-v2-job-gender.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12/figures/sb-v2-job-gender.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_repr_common(correct):
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
        true_repr = np.array([25, 29, 46, 36, 38, 41], dtype=np.float64) / 100
        pred_repr = np.array([36, 36, 46, 38, 40, 48], dtype=np.float64) / 100
        unsure_repr = np.array([23, 14, 7, 13, 10, 17], dtype=np.float64) / 100

        coeff_lower = pred_repr - (true_repr - unsure_repr / 2)
        coeff_upper = (true_repr + unsure_repr / 2) - pred_repr
    
    cfgw_counts_lower, cfgw_counts_upper = [], []
    for idx in range(len(cfgw_counts)):
        cfgw_counts_lower.append(cfgw_counts[idx] - coeff_lower)
        cfgw_counts_upper.append(cfgw_counts[idx] + coeff_upper)

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


def acc_repr_level():
    ws = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    true_repr = np.array([25, 29, 46, 36, 38, 41], dtype=np.float64) / 100
    pred_repr = np.array([36, 36, 46, 38, 40, 48], dtype=np.float64) / 100
    unsure_repr = np.array([23, 14, 7, 13, 10, 17], dtype=np.float64) / 100

    plt.figure(figsize=(8, 6))
    with plt.style.context('ggplot'):
        plt.plot(ws, pred_repr, linestyle="-", color="red", label="CLIP auto pred")
        plt.plot(ws, true_repr + unsure_repr / 2, linestyle="--", color="red", label="human anno lower/upper")
        plt.plot(ws, true_repr - unsure_repr / 2, linestyle="--", color="red")
        plt.fill_between(ws, true_repr + unsure_repr / 2, true_repr - unsure_repr / 2, alpha=0.2)

        plt.xticks(ws, ws)
        plt.ylim(0, 1.01)
        plt.yticks(np.arange(0, 1.01, step=0.05))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Empirical predicted v.s. annotated Female representation")
        plt.title("Automatic CLIP classifier w.r.t Guidance")
        plt.legend()
    
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12/figures/acc_repr_level.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12/figures/acc_repr_level.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_is(job_name=None):
    ws = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)

    if job_name is None:
        ws = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
        is_mean = np.array([12.558, 15.170, 15.829, 15.806, 15.777, 15.122, 15.262], dtype=np.float64)
        is_std = np.array([0.135, 0.261, 0.361, 0.300, 0.195, 0.254, 0.221], dtype=np.float64)
        is_mean_ext = np.array([11.857, 14.597, 14.748, 14.376, 13.784, 13.282, 12.956], dtype=np.float64)
        is_std_ext = np.array([0.101, 0.115, 0.210, 0.111, 0.095, 0.093, 0.204], dtype=np.float64)

        plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            plt.plot(ws, is_mean, linestyle="-", marker='o', label="IS w/o extended prompt", color="red")
            plt.fill_between(ws, is_mean + is_std, is_mean - is_std, alpha=0.2, color="red")
            plt.plot(ws, is_mean_ext, linestyle="-", marker='o', label="IS w/ extended prompt", color="blue")
            plt.fill_between(ws, is_mean_ext + is_std_ext, is_mean_ext - is_std_ext, alpha=0.2, color="blue")

            plt.xticks(ws, ws)
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylabel(r"IS value $\uparrow$")
            plt.title(f"Inception Score (IS) w.r.t Guidance")
            plt.legend()
        
        plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15/figures/is_overall.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15/figures/is_overall.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        return

    elif job_name == "author":
        is_male_mean = np.array([4.690, 4.094, 4.104, 4.032, 3.980, 4.078], dtype=np.float64)
        is_male_std = np.array([0.326, 0.435, 0.300, 0.252, 0.259, 0.281], dtype=np.float64)
        is_female_mean = np.array([5.242, 4.724, 4.471, 4.628, 4.470, 4.211], dtype=np.float64)
        is_female_std = np.array([0.482, 0.229, 0.287, 0.314, 0.262, 0.166], dtype=np.float64)
        female_repr = 0.392
        male_repr = 1 - female_repr
    elif job_name == "lab tech":
        is_male_mean = np.array([2.813, 1.849, 1.614, 1.620, 1.552, 1.565], dtype=np.float64)
        is_male_std = np.array([0.202, 0.126, 0.083, 0.140, 0.126, 0.113], dtype=np.float64)
        is_female_mean = np.array([2.308, 1.810, 1.648, 1.461, 1.515, 1.468], dtype=np.float64)
        is_female_std = np.array([0.248, 0.131, 0.175, 0.173, 0.066, 0.030], dtype=np.float64)
        female_repr = 0.33
        male_repr = 1 - female_repr
    elif job_name == "pharmacist":
        is_male_mean = np.array([2.444, 2.204, 2.054, 1.975, 1.947, 1.933], dtype=np.float64)
        is_male_std = np.array([0.055, 0.091, 0.110, 0.072, 0.117, 0.140], dtype=np.float64)
        is_female_mean = np.array([2.359, 2.367, 2.116, 1.916, 1.904, 1.815], dtype=np.float64)
        is_female_std = np.array([0.099, 0.074, 0.045, 0.078, 0.054, 0.068], dtype=np.float64)
        female_repr = 0.482
        male_repr = 1 - female_repr
    elif job_name == "veterinarian":
        is_male_mean = np.array([6.294, 5.561, 4.747, 4.378, 3.908, 4.042], dtype=np.float64)
        is_male_std = np.array([0.292, 0.129, 0.348, 0.351, 0.103, 0.315], dtype=np.float64)
        is_female_mean = np.array([7.071, 5.450, 4.544, 4.302, 3.935, 3.716], dtype=np.float64)
        is_female_std = np.array([0.419, 0.584, 0.230, 0.100, 0.227, 0.352], dtype=np.float64)
        female_repr = 0.456
        male_repr = 1 - female_repr
    elif job_name == "librarian":
        is_male_mean = np.array([6.294, 5.561, 4.747, 4.378, 3.908, 4.042], dtype=np.float64)
        is_male_std = np.array([0.292, 0.129, 0.348, 0.351, 0.103, 0.315], dtype=np.float64)
        is_female_mean = np.array([7.071, 5.450, 4.544, 4.302, 3.935, 3.716], dtype=np.float64)
        is_female_std = np.array([0.419, 0.584, 0.230, 0.100, 0.227, 0.352], dtype=np.float64)
        female_repr = 0.456
        male_repr = 1 - female_repr
    elif job_name == "software developer":
        is_male_mean = np.array([4.446, 3.382, 2.915, 2.832, 2.660, 2.564], dtype=np.float64)
        is_male_std = np.array([0.255, 0.391, 0.274, 0.213, 0.136, 0.098], dtype=np.float64)
        is_female_mean = np.array([3.648, 2.694, 2.456, 2.252, 2.220, 2.006], dtype=np.float64)
        is_female_std = np.array([0.149, 0.086, 0.074, 0.077, 0.064, 0.114], dtype=np.float64)
        female_repr = 0.218
        male_repr = 1 - female_repr
    elif job_name == "customer service representative":
        is_male_mean = np.array([5.473, 4.372, 4.138, 3.938, 4.041, 3.542], dtype=np.float64)
        is_male_std = np.array([0.558, 0.072, 0.215, 0.473, 0.313, 0.125], dtype=np.float64)
        is_female_mean = np.array([0.558, 0.072, 0.215, 0.473, 0.313, 0.125], dtype=np.float64)
        is_female_std = np.array([0.313, 0.385, 0.243, 0.354, 0.227, 0.220], dtype=np.float64)
        female_repr = 0.528
        male_repr = 1 - female_repr
    
    plt.figure(figsize=(8, 8/1.6))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    with plt.style.context('ggplot'):
        lns1 = ax1.plot(ws, is_male_mean, linestyle="-", marker='o', label="male IS", color="red")
        ax1.fill_between(ws, is_male_mean + is_male_std, is_male_mean - is_male_std, alpha=0.2, color="red")
        lns2 = ax1.plot(ws, is_female_mean, linestyle="-", marker='o', label="female IS", color="blue")
        ax1.fill_between(ws, is_female_mean + is_female_std, is_female_mean - is_female_std, alpha=0.2, color="blue")
        lns3 = ax2.axhline(male_repr, linestyle="--", label="male repr", color="red")
        lns4 = ax2.axhline(female_repr, linestyle="--", label="female repr", color="blue")

        ax1.set_xticks(ws, ws)
        ax1.set_xlabel("Scale of classifier-free guidance ($w$)")
        ax1.set_ylabel(r"IS value $\uparrow$")
        ax2.set_ylabel("percent")
        plt.title(f"Inception Score (IS) and Representation w.r.t Guidance for {job_name}")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
    
    job_name_title = job_name.replace(" ", "_")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15/figures/is_{job_name_title}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15/figures/is_{job_name_title}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_area(ratio, prompt_date, job_name=None):
    ws = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)

    if job_name is None:
        if prompt_date == "2023-10-12":
            box_area = np.array([48550.430, 68355.853, 76791.824, 81875.920, 84747.966, 86693.238, 89211.151], dtype=np.float64)
            mask_area = np.array([27448.790, 39572.997, 44905.137, 48086.105, 49862.348, 51317.725, 52755.480], dtype=np.float64)
            if ratio:
                box_area /= (512 * 512)
                mask_area /= (512 * 512)
            else:
                box_area /= 10**4
                mask_area /= 10**4
            
            plt.figure(figsize=(8, 8/1.6))
            with plt.style.context('ggplot'):
                plt.plot(ws, box_area, linestyle="-", marker='o', label="bounding box")
                plt.plot(ws, mask_area, linestyle="-", marker='o', label="segmentation mask")

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                if ratio:
                    plt.ylim(0.0, 0.41)
                    plt.yticks(np.arange(0.0, 0.41, step=0.05))
                    plt.ylabel(r"Percent of Detected Area on Image")
                else:
                    plt.ylim(2.0, 10.01)
                    plt.yticks(np.arange(2.0, 10.01, step=1.0))
                    plt.ylabel(r"Number of Pixels in Detected Area ($\times 10^4$)")
                plt.legend()
                plt.title(f"Average Area of the Predicted Person w.r.t Guidance on stable-diffusion-v2")

        elif prompt_date == "2023-10-26":
            box_area = np.array([82924.642, 121720.477, 133533.202, 139010.645, 142598.854, 144935.181, 147329.351], dtype=np.float64)
            mask_area = np.array([48482.189, 72501.553, 79607.583, 82540.191, 84850.324, 86134.319, 87486.960], dtype=np.float64)
            if ratio:
                box_area /= (512 * 512)
                mask_area /= (512 * 512)
            else:
                box_area /= 10**4
                mask_area /= 10**4
            
            plt.figure(figsize=(8, 8/1.6))
            with plt.style.context('ggplot'):
                plt.plot(ws, box_area, linestyle="-", marker='o', label="bounding box")
                plt.plot(ws, mask_area, linestyle="-", marker='o', label="segmentation mask")

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                if ratio:
                    plt.ylim(0.1, 0.71)
                    plt.yticks(np.arange(0.1, 0.71, step=0.05))
                    plt.ylabel(r"Percent of Detected Area on Image")
                else:
                    plt.ylim(4.0, 16.01)
                    plt.yticks(np.arange(4.0, 16.01, step=1.0))
                    plt.ylabel(r"Number of Pixels in Detected Area ($\times 10^4$)")
                plt.legend()
                plt.title(f"Average Area of the Predicted Person w.r.t Guidance on stable-diffusion-v2")

        if ratio:
            plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_area_ratio.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_area_ratio.pdf", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_area_value.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_area_value.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        return

def plot_area_bygender(ratio, prompt_date, job_name=None):
    ws = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)

    if job_name is None:
        if prompt_date == "2023-10-15":
            box_area_male = np.array([78676.913, 109148.644, 118823.930, 124649.410, 127712.918, 131233.132, 133176.112], dtype=np.float64)
            mask_area_male = np.array([45484.614, 64226.906, 70282.112, 73678.847, 75603.492, 77699.298, 78783.261], dtype=np.float64)
            box_area_female = np.array([79676.502, 110240.143, 119872.307, 124879.661, 129482.391, 132587.020, 134675.232], dtype=np.float64)
            mask_area_female = np.array([46215.723, 65724.784, 71985.728, 75178.224, 77971.478, 79801.550, 81236.795], dtype=np.float64)
            if ratio:
                box_area_male /= (512 * 512)
                mask_area_male /= (512 * 512)
                box_area_female /= (512 * 512)
                mask_area_female /= (512 * 512)
            else:
                box_area_male /= 10**4
                mask_area_male /= 10**4
                box_area_female /= 10**4
                mask_area_female /= 10**4
            
            plt.figure(figsize=(8, 8/1.6))
            with plt.style.context('ggplot'):
                plt.plot(ws, box_area_male, linestyle="-", marker='o', label="male")
                plt.plot(ws, box_area_female, linestyle="-", marker='o', label="female")

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                if ratio:
                    plt.ylim(0.2, 0.61)
                    plt.yticks(np.arange(0.2, 0.61, step=0.05))
                    plt.ylabel(r"Percent of Predicted Box Area on Image")
                else:
                    plt.ylim(7.0, 15.01)
                    plt.yticks(np.arange(7.0, 15.01, step=1.0))
                    plt.ylabel(r"Number of Pixels in Predicted Box Area ($\times 10^4$)")
                plt.legend()
                plt.title(f"Average Area of the Predicted Bounding Box Person w.r.t Guidance on stable-diffusion-v2")
                if ratio:
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_box_area_ratio_bygender.png", dpi=300, bbox_inches="tight")
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_box_area_ratio_bygender.pdf", dpi=300, bbox_inches="tight")
                else:
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_box_area_value_bygender.png", dpi=300, bbox_inches="tight")
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_box_area_value_bygender.pdf", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(8, 8/1.6))
            with plt.style.context('ggplot'):
                plt.plot(ws, mask_area_male, linestyle="-", marker='o', label="male")
                plt.plot(ws, mask_area_female, linestyle="-", marker='o', label="female")

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                if ratio:
                    plt.ylim(0.1, 0.41)
                    plt.yticks(np.arange(0.1, 0.41, step=0.05))
                    plt.ylabel(r"Percent of Predicted Mask Area on Image")
                else:
                    plt.ylim(4.0, 9.01)
                    plt.yticks(np.arange(4.0, 9.01, step=0.5))
                    plt.ylabel(r"Number of Pixels in Predicted Mask Area ($\times 10^4$)")
                plt.legend()
                plt.title(f"Average Area of the Predicted Segmentation Mask Person w.r.t Guidance on stable-diffusion-v2")
                if ratio:
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_mask_area_ratio_bygender.png", dpi=300, bbox_inches="tight")
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_mask_area_ratio_bygender.pdf", dpi=300, bbox_inches="tight")
                else:
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_mask_area_value_bygender.png", dpi=300, bbox_inches="tight")
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_mask_area_value_bygender.pdf", dpi=300, bbox_inches="tight")
            plt.close()

        elif prompt_date == "2023-10-29":
            box_area_male = np.array([78676.913, 109148.644, 118823.930, 124649.410, 127712.918, 131233.132, 133176.112], dtype=np.float64)
            mask_area_male = np.array([45484.614, 64226.906, 70282.112, 73678.847, 75603.492, 77699.298, 78783.261], dtype=np.float64)
            box_area_female = np.array([79676.502, 110240.143, 119872.307, 124879.661, 129482.391, 132587.020, 134675.232], dtype=np.float64)
            mask_area_female = np.array([46215.723, 65724.784, 71985.728, 75178.224, 77971.478, 79801.550, 81236.795], dtype=np.float64)
            if ratio:
                box_area_male /= (512 * 512)
                mask_area_male /= (512 * 512)
                box_area_female /= (512 * 512)
                mask_area_female /= (512 * 512)
            else:
                box_area_male /= 10**4
                mask_area_male /= 10**4
                box_area_female /= 10**4
                mask_area_female /= 10**4
            
            plt.figure(figsize=(8, 8/1.6))
            with plt.style.context('ggplot'):
                plt.plot(ws, box_area_male, linestyle="-", marker='o', label="male")
                plt.plot(ws, box_area_female, linestyle="-", marker='o', label="female")

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                if ratio:
                    plt.ylim(0.0, 0.41)
                    plt.yticks(np.arange(0.0, 0.41, step=0.05))
                    plt.ylabel(r"Percent of Predicted Box Area on Image")
                else:
                    plt.ylim(2.0, 10.01)
                    plt.yticks(np.arange(2.0, 10.01, step=1.0))
                    plt.ylabel(r"Number of Pixels in Predicted Box Area ($\times 10^4$)")
                plt.legend()
                plt.title(f"Average Area of the Predicted Bounding Box Person w.r.t Guidance on stable-diffusion-v2")
                if ratio:
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_box_area_ratio.png", dpi=300, bbox_inches="tight")
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_box_area_ratio.pdf", dpi=300, bbox_inches="tight")
                else:
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_box_area_value.png", dpi=300, bbox_inches="tight")
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_box_area_value.pdf", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(8, 8/1.6))
            with plt.style.context('ggplot'):
                plt.plot(ws, mask_area_male, linestyle="-", marker='o', label="male")
                plt.plot(ws, mask_area_female, linestyle="-", marker='o', label="female")

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                if ratio:
                    plt.ylim(0.0, 0.41)
                    plt.yticks(np.arange(0.0, 0.41, step=0.05))
                    plt.ylabel(r"Percent of Predicted Mask Area on Image")
                else:
                    plt.ylim(2.0, 10.01)
                    plt.yticks(np.arange(2.0, 10.01, step=1.0))
                    plt.ylabel(r"Number of Pixels in Predicted Mask Area ($\times 10^4$)")
                plt.legend()
                plt.title(f"Average Area of the Predicted Segmentation Mask Person w.r.t Guidance on stable-diffusion-v2")
                if ratio:
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_mask_area_ratio.png", dpi=300, bbox_inches="tight")
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_mask_area_ratio.pdf", dpi=300, bbox_inches="tight")
                else:
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_mask_area_value.png", dpi=300, bbox_inches="tight")
                    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/{prompt_date}/figures/sbv2_mask_area_value.pdf", dpi=300, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15/figures", exist_ok=True)
    # plot_repr_special(correct=True)
    # plot_repr_common(correct=True)
    # acc_repr_level()

    # plot_is()
    # special_jobs = ["author", "librarian", "pharmacist", "lab tech", "veterinarian", "librarian"]
    # for name in special_jobs:
        # plot_is(name)
    # common_jobs = ["software developer", "customer service representative"]
    # for name in common_jobs:
        # plot_is(name)
    plot_area(ratio=True, prompt_date="2023-10-12")
    plot_area(ratio=False, prompt_date="2023-10-12")
    plot_area(ratio=True, prompt_date="2023-10-26")
    plot_area(ratio=False, prompt_date="2023-10-26")
    plot_area_bygender(ratio=True, prompt_date="2023-10-15")
    plot_area_bygender(ratio=False, prompt_date="2023-10-15")