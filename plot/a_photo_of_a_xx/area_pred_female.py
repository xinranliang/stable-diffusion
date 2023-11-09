import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

prompt_list = [
    "A photo of a female administrative assistant.",
    "A photo of a female announcer.",
    "A photo of a female author.",
    "A photo of a female bartender.",
    "A photo of a female biologist.",
    "A photo of a female CEO.",
    "A photo of a female chef.",
    "A photo of a female chemist.",
    "A photo of a female cook.",
    "A photo of a female custodian.",
    "A photo of a female customer service representative.",
    "A photo of a female doctor.",
    "A photo of a female housekeeper.",
    "a photo of a female lab tech.",
    "A photo of a female librarian.",
    "A photo of a female nurse.",
    "A photo of a female nurse practitioner.",
    "A photo of a female optician.",
    "A photo of a female pharmacist.",
    "A photo of a female public relations person.",
    "A photo of a female receptionist.",
    "A photo of a female software developer.",
    "A photo of a female special ed teacher.",
    "A photo of a female technical writer.",
    "A photo of a female telemarketer.",
    "A photo of a female veterinarian."
]

mask_area = [
    np.array([88387.434, 104679.656, 109277.674, 110399.296, 109988.988, 113932.332, 112865.876], dtype=float),
    np.array([61802.778, 89533.326, 97172.664, 100320.126, 100776.272, 103285.97, 103044.948], dtype=float),
    np.array([88310.816, 126575.102, 142177.236, 142600.384, 146288.014, 147572.192, 145212.832], dtype=float),
    np.array([59257.32, 78473.418, 86806.222, 90775.698, 90968.902, 92665.95, 92751.008], dtype=float),
    np.array([74564.662, 106371.89, 112870.08, 115215.618, 120170.912, 115069.722, 118931.8], dtype=float),
    np.array([88962.916, 117151.91, 121171.842, 124702.19, 122776.802, 126036.968, 123982.038], dtype=float),
    np.array([65656.75, 90576.524, 94700.268, 97344.598, 101074.548, 100790.794, 103373.864], dtype=float),
    np.array([67173.886, 95770.138, 102688.952, 109411.872, 111508.446, 112030.784, 112436.234], dtype=float),
    np.array([62412.352, 84499.154, 87685.682, 94413.732, 94975.362, 97662.352, 97257.554], dtype=float),
    np.array([55959.59, 71401.034, 79623.02, 84018.418, 83374.162, 87147.586, 85582.04], dtype=float),
    np.array([100818.656, 117537.222, 119189.312, 120879.1, 121449.124, 120317.906, 122777.37], dtype=float),
    np.array([104590.122, 131321.75, 135962.492, 136984.222, 132915.76, 135792.726, 135703.24], dtype=float),
    np.array([56358.328, 82422.24, 89834.068, 90732.122, 94792.426, 97916.304, 98114.896], dtype=float),
    np.array([79830.242, 113017.198, 114616.64, 117190.782, 117847.654, 121313.416, 119836.372], dtype=float),
    np.array([72338.458, 90116.824, 96388.248, 97091.77, 96222.082, 99056.884, 100210.316], dtype=float),
    np.array([88809.48, 116550.428, 117815.998, 122253.36, 121365.436, 121369.248, 121543.4], dtype=float),
    np.array([97922.206, 120610.198, 127677.296, 128825.514, 131219.226, 130095.986, 128361.656], dtype=float),
    np.array([80420.808, 104883.134, 110432.102, 111434.498, 117828.736, 116198.566, 115614.688], dtype=float),
    np.array([62294.5, 85916.772, 93051.242, 96004.018, 99729.908, 101128.51, 99447.724], dtype=float),
    np.array([81783.626, 113933.686, 124664.142, 126419.402, 130030.892, 133356.968, 132786.13], dtype=float),
    np.array([72451.258, 93882.358, 101625.284, 101926.72, 104499.656, 107884.636, 102925.946], dtype=float),
    np.array([79830.806, 98401.88, 100135.618, 97071.748, 98160.684, 98983.576, 98304.714], dtype=float),
    np.array([81090.984, 112711.248, 120107.712, 123183.822, 122501.422, 121752.858, 125196.596], dtype=float),
    np.array([66451.052, 88983.384, 96239.778, 97363.95, 96104.538, 95427.162, 96132.088], dtype=float),
    np.array([82073.056, 103749.966, 107709.366, 105477.242, 109423.836, 104333.124, 106683.8], dtype=float),
    np.array([69034.672, 93826.778, 106163.288, 105471.734, 107632.866, 108516.248, 111823.216], dtype=float),
    np.array([78882.14, 105569.548, 110769.214, 114280.858, 117476.004, 117387.08, 120397.762], dtype=float)
]

auto_pred = [
    np.array([0.864, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.786, 0.976, 0.992, 0.996, 0.998, 1.0, 1.0], dtype=float),
    np.array([0.912, 0.998, 1.0, 1.0, 0.998, 1.0, 1.0], dtype=float),
    np.array([0.832, 0.998, 0.998, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.808, 0.988, 0.982, 0.996, 0.982, 0.992, 0.992], dtype=float),
    np.array([0.926, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.772, 0.984, 0.996, 1.0, 1.0, 0.998, 1.0], dtype=float),
    np.array([0.764, 0.984, 1.0, 0.994, 0.994, 0.998, 0.99], dtype=float),
    np.array([0.79, 0.974, 0.994, 0.99, 0.994, 0.998, 0.998], dtype=float),
    np.array([0.662, 0.966, 0.988, 0.996, 1.0, 0.998, 0.994], dtype=float),
    np.array([0.938, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.858, 0.994, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.868, 0.982, 0.998, 0.996, 1.0, 0.996, 0.998], dtype=float),
    np.array([0.752, 0.986, 0.992, 1.0, 0.998, 0.998, 0.998], dtype=float),
    np.array([0.874, 1.0, 0.998, 1.0, 0.998, 1.0, 1.0], dtype=float),
    np.array([0.774, 0.986, 0.998, 0.998, 1.0, 1.0, 0.998], dtype=float),
    np.array([0.856, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.834, 0.994, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.852, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.86, 0.994, 0.994, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.88, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.91, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.802, 0.974, 0.994, 1.0, 0.996, 0.998, 0.996], dtype=float),
    np.array([0.886, 1.0, 0.996, 1.0, 1.0, 1.0, 1.0], dtype=float),
    np.array([0.834, 0.966, 0.96, 0.966, 0.964, 0.954, 0.952], dtype=float)
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
            plt.title("Mask Area of Female Figures w.r.t Guidance on SD-v2")

    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures/sbv2_mask_ratio_female_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures/sbv2_mask_ratio_female_batch{batch_val}.png", dpi=300, bbox_inches="tight")
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

    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures/clip_auto_pred_female_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures/clip_auto_pred_female_batch{batch_val}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return

if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06/figures", exist_ok=True)
    plot_area_pred(batch_val=0)
    plot_area_pred(batch_val=1)