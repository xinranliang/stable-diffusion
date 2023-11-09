import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

prompt_list = [
    "A photo of a single female administrative assistant.",
    "A photo of a single female announcer.",
    "A photo of a single female author.",
    "A photo of a single female bartender.",
    "A photo of a single female CEO.",
    "A photo of a single female chef.",
    "A photo of a single female chemist.",
    "A photo of a single female cook.",
    "A photo of a single female customer service representative.",
    "A photo of a single female doctor.",
    "a photo of a single female lab tech.",
    "A photo of a single female librarian.",
    "A photo of a single female nurse.",
    "A photo of a single female nurse practitioner.",
    "A photo of a single female optician.",
    "A photo of a single female pharmacist.",
    "A photo of a single female plumber.",
    "A photo of a single female public relations person.",
    "A photo of a single female receptionist.",
    "A photo of a single female software developer.",
    "A photo of a single female technical writer.",
    "A photo of a single female telemarketer.",
    # "A photo of a single female veterinarian."
]

mask_area = [
    np.array([84069.25, 101716.056, 108077.926, 107077.838, 107559.05, 105976.766, 106097.374], dtype=float),
    np.array([57990.704, 92419.12, 101056.206, 102216.4, 106186.034, 106065.336, 106993.538], dtype=float),
    np.array([88640.576, 126578.45, 134683.934, 140170.494, 141455.448, 143187.704, 145882.79], dtype=float),
    np.array([58107.884, 78551.788, 85554.386, 86557.708, 89929.436, 90904.814, 90004.248], dtype=float),
    np.array([66407.434, 96397.846, 103593.162, 107453.074, 108663.048, 106839.31, 110563.83], dtype=float),
    np.array([69989.118, 92940.21, 94429.89, 99907.856, 98252.998, 100005.9, 102262.632], dtype=float),
    np.array([65328.542, 89540.124, 96785.12, 102973.864, 106208.818, 103163.536, 107950.152], dtype=float),
    np.array([60237.02, 82377.388, 85905.098, 90519.114, 92506.688, 94156.67, 93602.336], dtype=float),
    np.array([99449.576, 120070.412, 126706.32, 122340.958, 124886.206, 127424.812, 125826.2], dtype=float),
    np.array([97749.032, 127225.242, 128634.092, 132626.464, 130630.492, 131614.046, 130977.752], dtype=float),
    np.array([74377.3, 99271.09, 111089.674, 113999.984, 115251.234, 116522.904, 118182.276], dtype=float),
    np.array([62529.604, 88295.014, 91489.488, 93517.78, 98130.984, 98964.078, 98232.072], dtype=float),
    np.array([91826.508, 116882.59, 121451.794, 124252.218, 123373.632, 123424.0, 125828.914], dtype=float),
    np.array([90873.868, 119269.282, 127300.124, 127753.876, 129835.69, 128362.622, 127449.168], dtype=float),
    np.array([70218.814, 104069.052, 110177.442, 112102.734, 115259.638, 120828.126, 117654.232], dtype=float),
    np.array([60416.3, 81829.73, 91332.142, 96031.514, 96582.054, 98552.706, 98775.244], dtype=float),
    np.array([63716.28, 91163.3, 96354.026, 99857.506, 105807.182, 105012.812, 105632.72], dtype=float),
    np.array([74870.016, 108999.696, 119654.736, 127326.21, 127052.068, 127132.222, 128284.042], dtype=float),
    np.array([66996.092, 92631.496, 99865.902, 100865.106, 101869.332, 105291.004, 107476.456], dtype=float),
    np.array([69693.522, 94833.166, 96060.872, 98088.14, 96212.0, 97077.524, 97707.666], dtype=float),
    np.array([62784.026, 84947.93, 90262.508, 93211.586, 93108.496, 92858.954, 93870.288], dtype=float),
    np.array([82073.056, 103749.966, 107709.366, 105477.242, 109423.836, 104333.124, 106683.8], dtype=float),
    # np.array([69034.672, 93826.778, 106163.288, 105471.734, 107632.866, 108516.248, 111823.216], dtype=float)
]

auto_pred = [
    np.array([0.918, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.728, 0.976, 0.996, 0.994, 0.996, 1.0, 0.996], dtype=float),
    np.array([0.904, 0.992, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.81, 0.992, 0.996, 0.996, 1.0, 0.996, 0.998], dtype=float),
    np.array([0.872, 1.0, 0.998, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.794, 0.992, 0.996, 0.998, 0.998, 1.0, 0.998], dtype=float),
    np.array([0.768, 0.98, 0.992, 0.994, 0.996, 0.996, 0.996], dtype=float),
    np.array([0.748, 0.974, 0.996, 0.996, 0.996, 0.998, 1.0], dtype=float),
    np.array([0.948, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.854, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.746, 0.982, 0.992, 0.988, 0.998, 0.996, 0.994], dtype=float),
    np.array([0.864, 0.986, 1.0, 0.998, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.818, 0.99, 0.998, 1.0, 0.998, 1.0, 1.0], dtype=float),
    np.array([0.842, 0.996, 1.0, 1.0, 0.998, 1.0, 1.0], dtype=float),
    np.array([0.806, 0.998, 0.998, 1.0, 0.998, 1.0, 1.0], dtype=float),
    np.array([0.832, 0.98, 0.996, 1.0, 0.998, 1.0, 1.0], dtype=float),
    np.array([0.682, 0.97, 0.99, 0.996, 0.992, 0.992, 0.998], dtype=float),
    np.array([0.874, 0.988, 0.998, 1.0, 0.998, 1.0, 0.998], dtype=float),
    np.array([0.88, 0.996, 0.998, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.816, 0.99, 1.0, 0.998, 1.0, 1.0, 0.998], dtype=float),
    np.array([0.792, 0.958, 0.984, 0.992, 0.998, 0.992, 0.996], dtype=float),
    np.array([0.894, 0.996, 1.0, 1.0, 1.0, 1.0, 0.998], dtype=float),
    # np.array([0.818, 0.954, 0.98, 0.956, 0.95, 0.966, 0.972], dtype=float)
]

def plot_area_pred(batch_val):
    ws = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan", "magenta"] * 2

    subset_index = range(len(prompt_list))[: len(prompt_list) // 2] if batch_val == 0 else range(len(prompt_list))[len(prompt_list) // 2 :]

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in subset_index:
            plt.plot(ws[1:], mask_area[idx][1:] / (512 * 512), linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

            plt.xticks(ws[1:], ws[1:])
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylim(0.2, 0.61)
            plt.yticks(np.arange(0.2, 0.61, step=0.05))
            plt.ylabel("Percent of Predicted Mask Area on Image")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title("Mask Area of Female Figures w.r.t Guidance on SD-v2")

    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures/sbv2_mask_ratio_female_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures/sbv2_mask_ratio_female_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in subset_index:
            plt.plot(ws[1:], auto_pred[idx][1:], linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

            plt.xticks(ws[1:], ws[1:])
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylim(0.95, 1.01)
            plt.yticks(np.arange(0.95, 1.01, step=0.01))
            plt.ylabel("Value")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title("Accuracy of CLIP predictions on Female samples w.r.t Guidance on SD-v2")

    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures/clip_auto_pred_female_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures/clip_auto_pred_female_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return

if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures", exist_ok=True)
    plot_area_pred(batch_val=0)
    plot_area_pred(batch_val=1)