import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

prompt_list = [
    "A photo of a single male administrative assistant.",
    "A photo of a single male announcer.",
    "A photo of a single male author.",
    "A photo of a single male bartender.",
    "A photo of a single male CEO.",
    "A photo of a single male chef.",
    "A photo of a single male chemist.",
    "A photo of a single male cook.",
    "A photo of a single male customer service representative.",
    "A photo of a single male doctor.",
    "a photo of a single male lab tech.",
    "A photo of a single male librarian.",
    "A photo of a single male nurse.",
    "A photo of a single male nurse practitioner.",
    "A photo of a single male optician.",
    "A photo of a single male pharmacist.",
    "A photo of a single male plumber.",
    "A photo of a single male public relations person.",
    "A photo of a single male receptionist.",
    "A photo of a single male software developer.",
    "A photo of a single male technical writer.",
    "A photo of a single male telemarketer.",
    # "A photo of a single male veterinarian."
]

mask_area = [
    np.array([92696.394, 116434.994, 118944.224, 119863.296, 122669.254, 119430.394, 119998.086], dtype=float),
    np.array([57643.716, 88006.314, 95814.612, 103125.822, 104395.346, 103522.954, 105628.824], dtype=float),
    np.array([73728.798, 116001.256, 121048.362, 124098.972, 124058.426, 126847.27, 127071.034], dtype=float),
    np.array([64998.466, 82015.17, 88222.654, 91990.686, 94352.266, 97887.16, 96614.258], dtype=float),
    np.array([67725.018, 97466.42, 103449.23, 106846.066, 107271.982, 109932.554, 111857.822], dtype=float),
    np.array([73432.498, 95295.842, 99273.832, 100339.048, 101473.52, 103581.828, 105441.044], dtype=float),
    np.array([59301.024, 81698.664, 86818.428, 93011.174, 94179.628, 98331.126, 98520.298], dtype=float),
    np.array([66941.108, 89317.314, 91772.196, 96628.052, 97842.462, 101063.07, 100379.324], dtype=float),
    np.array([92056.452, 112407.14, 115914.004, 116447.53, 117125.212, 115953.452, 117629.092], dtype=float),
    np.array([93301.806, 122900.148, 124412.162, 124715.784, 124434.12, 125999.958, 125906.458], dtype=float),
    np.array([66014.45, 99503.03, 102386.236, 107606.188, 110316.76, 111551.284, 115254.778], dtype=float),
    np.array([64165.876, 82692.296, 90860.258, 90147.25, 92394.752, 92959.554, 92915.756], dtype=float),
    np.array([91647.412, 117215.668, 119435.734, 121479.81, 121388.08, 121491.708, 120894.922], dtype=float),
    np.array([91872.414, 116807.072, 121129.704, 120203.078, 119705.816, 124271.46, 121069.668], dtype=float),
    np.array([65366.89, 94179.87, 97961.938, 104480.782, 104620.88, 106054.954, 105790.732], dtype=float),
    np.array([54864.82, 80008.596, 86681.96, 92686.104, 94708.332, 95670.862, 98671.102], dtype=float),
    np.array([73434.816, 94373.166, 103653.988, 106523.872, 107889.602, 109819.17, 108969.266], dtype=float),
    np.array([71250.932, 108200.162, 111887.03, 120403.042, 119877.582, 123876.904, 122256.902], dtype=float),
    np.array([70186.282, 94093.248, 98025.42, 103489.228, 105846.048, 104114.314, 104297.0], dtype=float),
    np.array([80449.046, 104468.51, 106510.856, 107925.524, 106679.166, 106229.228, 106191.866], dtype=float),
    np.array([68221.696, 91419.89, 95874.606, 96766.734, 99709.232, 98748.392, 96543.534], dtype=float),
    np.array([78418.918, 101384.58, 106030.524, 105154.112, 108337.188, 107054.88, 106306.276], dtype=float),
    # np.array([69034.672, 93826.778, 106163.288, 105471.734, 107632.866, 108516.248, 111823.216], dtype=float)
]

auto_pred = [
    np.array([0.1, 0.006, 0.0, 0.004, 0.002, 0.0, 0.004], dtype=float),
    np.array([0.188, 0.014, 0.008, 0.004, 0.006, 0.004, 0.006], dtype=float),
    np.array([0.222, 0.01, 0.004, 0.004, 0.002, 0.0, 0.0], dtype=float),
    np.array([0.136, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.112, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.146, 0.006, 0.002, 0.002, 0.002, 0.0, 0.002], dtype=float),
    np.array([0.186, 0.024, 0.008, 0.0, 0.004, 0.002, 0.002], dtype=float),
    np.array([0.206, 0.006, 0.0, 0.0, 0.002, 0.004, 0.004], dtype=float),
    np.array([0.082, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.138, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.15, 0.02, 0.01, 0.018, 0.004, 0.004, 0.0], dtype=float),
    np.array([0.15, 0.01, 0.0, 0.006, 0.0, 0.0, 0.002], dtype=float),
    np.array([0.14, 0.016, 0.006, 0.004, 0.002, 0.004, 0.0], dtype=float),
    np.array([0.16, 0.024, 0.016, 0.012, 0.012, 0.0, 0.004], dtype=float),
    np.array([0.19, 0.018, 0.002, 0.002, 0.0, 0.002, 0.0], dtype=float),
    np.array([0.186, 0.01, 0.004, 0.002, 0.002, 0.0, 0.0], dtype=float),
    np.array([0.078, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.242, 0.02, 0.016, 0.004, 0.0, 0.0, 0.004], dtype=float),
    np.array([0.162, 0.012, 0.004, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.08, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.166, 0.024, 0.01, 0.002, 0.0, 0.002, 0.006], dtype=float),
    np.array([0.158, 0.008, 0.002, 0.0, 0.0, 0.002, 0.0], dtype=float),
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
            plt.title("Mask Area of Male Figures w.r.t Guidance on SD-v2")

    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures/sbv2_mask_ratio_male_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures/sbv2_mask_ratio_male_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in subset_index:
            plt.plot(ws[1:], 1 - auto_pred[idx][1:], linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

            plt.xticks(ws[1:], ws[1:])
            plt.xlabel("Scale of classifier-free guidance ($w$)")
            plt.ylim(0.95, 1.01)
            plt.yticks(np.arange(0.95, 1.01, step=0.01))
            plt.ylabel("Value")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            plt.title("Accuracy of CLIP predictions on Male samples w.r.t Guidance on SD-v2")

    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures/clip_auto_pred_male_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures/clip_auto_pred_male_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return

if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/figures", exist_ok=True)
    plot_area_pred(batch_val=0)
    plot_area_pred(batch_val=1)