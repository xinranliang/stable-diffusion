import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

prompt_list = [
    "A photo of a person.",
    "A photo of a person in the center.",
    "A photo of a single person.",
    "A photo of a single person in the center.",
    "A photo of a [JOB].",
    "A photo of a [JOB] in the center.",
    "A photo of a single [JOB].",
    "A photo of a single [JOB] in the center.",
]

female_prompt_list = [
    "A photo of a female person.",
    "A photo of a female person in the center.",
    "A photo of a single female person.",
    "A photo of a single female person in the center.",
    # "A photo of a female [JOB].",
    # "A photo of a female [JOB] in the center.",
    "A photo of a single female [JOB].",
    "A photo of a single female [JOB] in the center.",
]

male_prompt_list = [
    "A photo of a male person.",
    "A photo of a male person in the center.",
    "A photo of a single male person.",
    "A photo of a single male person in the center.",
    # "A photo of a male [JOB].",
    # "A photo of a male [JOB] in the center.",
    "A photo of a single male [JOB].",
    "A photo of a single male [JOB] in the center.",
]

summary_list = prompt_list + female_prompt_list + male_prompt_list

def plot_area_pred(summary=False):
    ws = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

    mask_area = [
            np.array([56957.47, 96715.59, 104232.80, 109698.21, 117749.06, 114686.76, 116804.35], dtype=float),
            np.array([23536.56, 42170.06, 51330.23, 52729.38, 58165.26, 61825.21, 67268.87], dtype=float),
            np.array([39718.82, 67655.67, 74158.15, 79769.87, 87734.06, 86295.14, 92241.57], dtype=float),
            np.array([20237.66, 31157.26, 30735.35, 32539.24, 35458.70, 35406.51, 32522.75], dtype=float),
            np.array([56043.18, 82836.10, 89560.96, 93073.39, 95155.16, 96384.73, 98104.88], dtype=float),
            np.array([34274.00, 49059.66, 55873.75, 58754.94, 61276.79, 63271.45, 64332.35], dtype=float),
            np.array([48482.19, 72501.55, 79607.58, 82540.19, 84850.32, 86134.32, 87486.96], dtype=float),
            np.array([27448.79, 39573.00, 44905.14, 48086.11, 49862.35, 51317.73, 52755.48], dtype=float),
        ]
    auto_pred = [
            np.array([0.562, 0.666, 0.726, 0.744, 0.77, 0.766, 0.738], dtype=float),
            np.array([0.55, 0.468, 0.508, 0.53, 0.548, 0.61, 0.632], dtype=float),
            np.array([0.46, 0.422, 0.376, 0.318, 0.314, 0.294, 0.29], dtype=float),
            np.array([0.464, 0.454, 0.452, 0.424, 0.42, 0.432, 0.444], dtype=float),
            np.array([0.4064, 0.40575, 0.41815, 0.4319, 0.4396, 0.4396, 0.4483], dtype=float),
            np.array([0.39525, 0.40235, 0.4127, 0.4238, 0.4261, 0.42935, 0.43395], dtype=float),
            np.array([0.386, 0.378, 0.385, 0.394, 0.405, 0.407, 0.419], dtype=float),
            np.array([0.376, 0.377, 0.396, 0.403, 0.409, 0.413, 0.416], dtype=float),
        ]
    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(prompt_list)):
            plt.plot(ws, mask_area[idx] / (512 * 512), linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

            plt.xticks(ws, ws)
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylim(0.0, 0.51)
            plt.yticks(np.arange(0.0, 0.51, step=0.05))
            plt.ylabel("Percent of Predicted Mask Area on Image")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title("Mask Area of Figures w.r.t Guidance on SD-v2")

    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_agnostic.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_agnostic.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(prompt_list)):
            plt.plot(ws, auto_pred[idx], linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

            plt.xticks(ws, ws)
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylim(0.2, 0.81)
            plt.yticks(np.arange(0.2, 0.81, step=0.05))
            plt.ylabel("Value")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title("Percent of CLIP predicted females w.r.t Guidance on SD-v2")

    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_agnostic.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_agnostic.png", dpi=300, bbox_inches="tight")
    plt.close()

    if summary:
        return mask_area, auto_pred

def plot_area_pred_female(summary=False):
    ws = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

    mask_area = [
            np.array([81371.95, 138144.14, 145771.43, 150207.48, 147326.50, 148218.83, 151300.89], dtype=float),
            np.array([48186.75, 89702.80, 101462.66, 114453.11, 114788.87, 118145.55, 113460.94], dtype=float),
            np.array([82183.59, 129010.68, 136474.49, 140442.81, 136918.57, 139879.19, 138830.82], dtype=float),
            np.array([42782.90, 74791.99, 90200.11, 92344.72, 93039.28, 90864.63, 98256.14], dtype=float),
            np.array([66398.15, 91524.65, 97821.97, 100873.07, 102622.06, 103676.61, 104399.36], dtype=float),
            np.array([46215.72, 65724.78, 71985.73, 75178.22, 77971.48, 79801.55, 81236.80], dtype=float),
        ]
    auto_pred = [
            np.array([0.822, 0.99, 0.998, 0.998, 0.998, 0.998, 0.99], dtype=float),
            np.array([0.752, 0.958, 0.992, 0.992, 0.992, 0.994, 0.992], dtype=float),
            np.array([0.796, 0.986, 0.99, 0.994, 0.998, 0.99, 0.994], dtype=float),
            np.array([0.756, 0.916, 0.968, 0.98, 0.986, 0.988, 0.994], dtype=float),
            np.array([0.7384, 0.95, 0.9756, 0.98135, 0.98605, 0.98725, 0.98595], dtype=float),
            np.array([0.672, 0.921, 0.962, 0.974, 0.981, 0.982, 0.982], dtype=float),
        ]
    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(female_prompt_list)):
            plt.plot(ws, mask_area[idx] / (512 * 512), linestyle="-", marker='o', color=colors[idx], label=female_prompt_list[idx])

            plt.xticks(ws, ws)
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylim(0.0, 0.71)
            plt.yticks(np.arange(0.0, 0.71, step=0.05))
            plt.ylabel("Percent of Predicted Mask Area on Image")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title("Mask Area of Figures w.r.t Guidance on SD-v2")

    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_female.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_female.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(female_prompt_list)):
            plt.plot(ws, auto_pred[idx], linestyle="-", marker='o', color=colors[idx], label=female_prompt_list[idx])

            plt.xticks(ws, ws)
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylim(0.6, 1.01)
            plt.yticks(np.arange(0.6, 1.01, step=0.05))
            plt.ylabel("Value")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title("Percent of CLIP predicted females w.r.t Guidance on SD-v2")

    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_female.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_female.png", dpi=300, bbox_inches="tight")
    plt.close()

    if summary:
        return mask_area, auto_pred

def plot_area_pred_male(summary=False):
    ws = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

    mask_area = [
            np.array([99132.47, 145780.16, 153361.70, 150540.46, 149513.99, 153569.95, 154872.71], dtype=float),
            np.array([53184.56, 98260.19, 108101.99, 115468.20, 118709.24, 118575.14, 117397.69], dtype=float),
            np.array([84785.97, 134564.61, 139419.32, 137598.60, 138923.42, 143062.39, 142846.20], dtype=float),
            np.array([39490.71, 69067.09, 78622.94, 80917.86, 87854.57, 85607.50, 88610.76], dtype=float),
            np.array([66533.66, 91835.11, 96918.29, 100169.50, 101397.74, 102729.18, 103672.35], dtype=float),
            np.array([45484.61, 64226.91, 70282.11, 73678.85, 75603.49, 77699.30, 78783.26], dtype=float),
        ]
    auto_pred = [
            np.array([0.168, 0.012, 0.004, 0, 0, 0, 0], dtype=float),
            np.array([0.26, 0.03, 0.004, 0.004, 0.008, 0.004, 0.004], dtype=float),
            np.array([0.21, 0.024, 0.004, 0.002, 0.002, 0, 0.002], dtype=float),
            np.array([0.278, 0.048, 0.04, 0.022, 0.006, 0.014, 0.016], dtype=float),
            np.array([0.15585, 0.0201, 0.00865, 0.00695, 0.00565, 0.005, 0.00615], dtype=float),
            np.array([0.187, 0.039, 0.019, 0.013, 0.012, 0.009, 0.010], dtype=float),
        ]
    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(male_prompt_list)):
            plt.plot(ws, mask_area[idx] / (512 * 512), linestyle="-", marker='o', color=colors[idx], label=male_prompt_list[idx])

            plt.xticks(ws, ws)
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylim(0.0, 0.71)
            plt.yticks(np.arange(0.0, 0.71, step=0.05))
            plt.ylabel("Percent of Predicted Mask Area on Image")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title("Mask Area of Figures w.r.t Guidance on SD-v2")

    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_male.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_male.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(male_prompt_list)):
            plt.plot(ws, 1 - auto_pred[idx], linestyle="-", marker='o', color=colors[idx], label=male_prompt_list[idx])

            plt.xticks(ws, ws)
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylim(0.7, 1.01)
            plt.yticks(np.arange(0.7, 1.01, step=0.05))
            plt.ylabel("Value")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title("Percent of CLIP predicted females w.r.t Guidance on SD-v2")

    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_male.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_male.png", dpi=300, bbox_inches="tight")
    plt.close()

    if summary:
        return mask_area, auto_pred

if __name__ == "__main__":
    plot_area_pred()
    plot_area_pred_female()
    plot_area_pred_male()