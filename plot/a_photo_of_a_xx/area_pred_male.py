import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

prompt_list = [
    "A photo of a male administrative assistant.",
    "A photo of a male announcer.",
    "A photo of a male author.",
    "A photo of a male bartender.",
    "A photo of a male biologist.",
    "A photo of a male CEO.",
    "A photo of a male chef.",
    "A photo of a male chemist.",
    "A photo of a male cook.",
    "A photo of a male custodian.",
    "A photo of a male customer service representative.",
    "A photo of a male doctor.",
    "A photo of a male housekeeper.",
    "a photo of a male lab tech.",
    "A photo of a male librarian.",
    "A photo of a male nurse.",
    "A photo of a male nurse practitioner.",
    "A photo of a male optician.",
    "A photo of a male pharmacist.",
    "A photo of a male public relations person.",
    "A photo of a male receptionist.",
    "A photo of a male software developer.",
    "A photo of a male special ed teacher.",
    "A photo of a male technical writer.",
    "A photo of a male telemarketer.",
    "A photo of a male veterinarian."
]

mask_area = [
    np.array([104964.416, 124867.342, 126271.11, 129132.176, 132276.836, 132927.218, 131778.46], dtype=float),
    np.array([58903.18, 90211.07, 99575.47, 105106.542, 103388.598, 106418.036, 104788.68], dtype=float),
    np.array([82722.386, 118703.808, 122935.29, 128696.51, 129213.028, 133278.064, 129701.794], dtype=float),
    np.array([63260.948, 85272.424, 93046.956, 93092.392, 94684.142, 97672.092, 99946.682], dtype=float),
    np.array([79030.414, 109066.68, 113925.922, 113988.338, 116610.552, 119837.244, 121205.828], dtype=float),
    np.array([85033.572, 111894.132, 118444.988, 118301.98, 118976.17, 119779.096, 122337.71], dtype=float),
    np.array([73496.19, 94424.826, 98596.082, 101747.816, 104384.418, 104834.372, 106248.53], dtype=float),
    np.array([63299.458, 92057.448, 101054.834, 102381.05, 104684.466, 106426.836, 105675.824], dtype=float),
    np.array([69583.664, 89805.892, 97126.376, 98300.56, 101769.022, 102910.538, 105565.27], dtype=float),
    np.array([59690.47, 71714.428, 79453.886, 80251.746, 79666.148, 82325.366, 85725.966], dtype=float),
    np.array([97911.908, 114149.694, 116649.556, 121763.026, 120768.976, 121232.334, 120758.176], dtype=float),
    np.array([6102888.016, 124037.628, 127637.43, 127799.84, 127287.67, 128120.518, 129752.656], dtype=float),
    np.array([59278.778, 81537.712, 88455.942, 88112.524, 92405.62, 91791.042, 95816.134], dtype=float),
    np.array([74402.096, 104260.516, 111223.77, 114956.818, 117067.424, 117968.714, 119920.716], dtype=float),
    np.array([70478.216, 89995.54, 93820.672, 97463.548, 97373.056, 99519.268, 99487.042], dtype=float),
    np.array([94224.692, 118311.59, 120072.912, 123243.0, 121214.488, 123202.996, 125923.71], dtype=float),
    np.array([96761.584, 121178.786, 125959.942, 123688.818, 124505.158, 126380.384, 126017.508], dtype=float),
    np.array([75989.046, 104704.528, 112042.302, 110398.682, 112785.002, 114057.516, 112464.244], dtype=float),
    np.array([61645.946, 83270.916, 90713.552, 93472.408, 98186.736, 96714.84, 101156.022], dtype=float),
    np.array([63131.154, 104154.842, 114539.476, 112299.598, 115229.324, 119273.916, 117665.252], dtype=float),
    np.array([72711.93, 98400.81, 103254.146, 106032.596, 103064.384, 106190.558, 106795.734], dtype=float),
    np.array([84881.922, 109988.72, 113836.632, 113490.14, 113800.814, 111834.202, 111678.116], dtype=float),
    np.array([76573.302, 110131.46, 114496.312, 116066.712, 115861.906, 118272.194, 118061.632], dtype=float),
    np.array([70559.218, 98572.46, 100845.422, 103047.476, 104034.558, 103812.548, 103783.432], dtype=float),
    np.array([89065.674, 109675.976, 110690.764, 113740.824, 112628.532, 113100.078, 113668.116], dtype=float),
    np.array([76437.628, 99949.548, 104113.272, 109202.878, 111557.208, 113169.89, 114832.916], dtype=float)
]

auto_pred = [
    np.array([0.092, 0.004, 0.0, 0.004, 0.0, 0.002, 0.0], dtype=float),
    np.array([0.16, 0.024, 0.01, 0.006, 0.006, 0.002, 0.004], dtype=float),
    np.array([0.176, 0.004, 0.002, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.124, 0.006, 0.002, 0.0, 0.0, 0.0, 0.002], dtype=float),
    np.array([0.122, 0.002, 0.002, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.092, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.128, 0.012, 0.0, 0.006, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.156, 0.012, 0.01, 0.002, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.222, 0.022, 0.006, 0.006, 0.002, 0.002, 0.002], dtype=float),
    np.array([0.142, 0.008, 0.004, 0.0, 0.006, 0.002, 0.002], dtype=float),
    np.array([0.102, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.116, 0.002, 0.002, 0.0, 0.002, 0.0, 0.0], dtype=float),
    np.array([0.272, 0.044, 0.036, 0.022, 0.012, 0.016, 0.016], dtype=float),
    np.array([0.13, 0.018, 0.004, 0.006, 0.0, 0.002, 0.002], dtype=float),
    np.array([0.124, 0.008, 0.0, 0.002, 0.0, 0.002, 0.002], dtype=float),
    np.array([0.186, 0.02, 0.004, 0.01, 0.0, 0.004, 0.0], dtype=float),
    np.array([0.18, 0.016, 0.01, 0.002, 0.008, 0.0, 0.008], dtype=float),
    np.array([0.146, 0.008, 0.002, 0.002, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.136, 0.02, 0.006, 0.002, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.19, 0.046, 0.026, 0.01, 0.006, 0.008, 0.006], dtype=float),
    np.array([0.128, 0.002, 0.002, 0.002, 0.0, 0.002, 0.0], dtype=float),
    np.array([0.1, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    np.array([0.188, 0.028, 0.012, 0.002, 0.0, 0.006, 0.0], dtype=float),
    np.array([0.162, 0.008, 0.0, 0.0, 0.002, 0.012, 0.002], dtype=float),
    np.array([0.144, 0.004, 0.004, 0.006, 0.002, 0.002, 0.002], dtype=float),
    np.array([0.214, 0.048, 0.02, 0.008, 0.016, 0.01, 0.01], dtype=float)
]

def plot_area_pred(batch_val):
    ws = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan", "magenta", "navy", "orchid"] * 2

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

    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures/sbv2_mask_ratio_male_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures/sbv2_mask_ratio_male_batch{batch_val}.png", dpi=300, bbox_inches="tight")
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

    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures/clip_auto_pred_male_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures/clip_auto_pred_male_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return

if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures", exist_ok=True)
    plot_area_pred(batch_val=0)
    plot_area_pred(batch_val=1)