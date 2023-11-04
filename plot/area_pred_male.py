import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_area_pred(extend_description):
    ws = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

    if extend_description is None:
        prompt_list = [
            "A photo of a male person.",
            "A photo of a single male doctor.",
            "A photo of a single male nurse practitioner.",
            "A photo of a single male lab tech.",
            "A photo of a single male pharmacist.",
            "A photo of a single male biologist.",
            "A photo of a single male chemist.",
            "A photo of a single male software developer.",
            "A photo of a single male author.",
        ]

        box_area = [
            np.array([145518.221, 212147.839, 223391.755, 220754.689, 226100.824, 228636.169, 231230.363], dtype=float),
            np.array([147052.134, 191883.343, 196178.894, 197742.653, 198823.216, 201636.570, 200556.623], dtype=float),
            np.array([148284.948, 189625.491, 193670.355, 196028.896, 195699.820, 201760.316, 199211.157], dtype=float),
            np.array([111956.926, 170812.673, 179583.362, 187759.283, 190845.983, 193868.426, 198694.011], dtype=float),
            np.array([91467.554, 131415.118, 142543.752, 151098.508, 154959.657, 156738.449, 160664.481], dtype=float),
            np.array([128941.421, 168822.989, 172727.413, 177081.358, 182222.225, 183358.827, 184623.297], dtype=float),
            np.array([99618.637, 136458.062, 144666.084, 154997.892, 155680.246, 162475.521, 164642.299], dtype=float),
            np.array([135607.687, 179117.612, 183769.052, 188498.903, 187965.829, 185981.068, 186515.333], dtype=float),
            np.array([119171.228, 186126.660, 194390.760, 200895.259, 201792.991, 205270.778, 205441.289], dtype=float),
        ]
        mask_area = [
            np.array([99132.468, 145780.158, 153361.704, 150540.456, 149513.992, 153569.954, 154872.712], dtype=float),
            np.array([93301.806, 122900.148, 124412.162, 124715.784, 124434.12, 125999.958, 125906.458], dtype=float),
            np.array([91872.414, 116807.072, 121129.704, 120203.078, 119705.816, 124271.46, 121069.668], dtype=float),
            np.array([66014.45, 99503.03, 102386.236, 107606.188, 110316.76, 111551.284, 115254.778], dtype=float),
            np.array([54864.82, 80008.596, 86681.96, 92686.104, 94708.332, 95670.862, 98671.102], dtype=float),
            np.array([76833.986, 102809.776, 103831.12, 108010.758, 109933.814, 110264.678, 111214.476], dtype=float),
            np.array([59301.024, 81698.664, 86818.428, 93011.174, 94179.628, 98331.126, 98520.298], dtype=float),
            np.array([80449.046, 104468.51, 106510.856, 107925.524, 106679.166, 106229.228, 106191.866], dtype=float),
            np.array([73728.798, 116001.256, 121048.362, 124098.972, 124058.426, 126847.27, 127071.034], dtype=float),
        ]
        auto_pred = [
            np.array([0.168, 0.012, 0.004, 0, 0, 0, 0], dtype=float),
            np.array([0.854, 0.996, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
            np.array([0.842, 0.996, 1.0, 1.0, 0.998, 1.0, 1.0], dtype=float),
            np.array([0.746, 0.982, 0.992, 0.988, 0.998, 0.996, 0.994], dtype=float),
            np.array([0.832, 0.98, 0.996, 1.0, 0.998, 1.0, 1.0], dtype=float),
            np.array([0.778, 0.968, 0.986, 0.984, 0.994, 0.988, 0.97], dtype=float),
            np.array([0.768, 0.98, 0.992, 0.994, 0.996, 0.996, 0.996], dtype=float),
            np.array([0.816, 0.99, 1.0, 0.998, 1.0, 1.0, 0.998], dtype=float),
            np.array([0.904, 0.992, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
        ]

        plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            for idx in range(len(prompt_list)):
                plt.plot(ws, box_area[idx] / (512 * 512), linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                plt.ylim(0.3, 0.91)
                plt.yticks(np.arange(0.3, 0.91, step=0.1))
                plt.ylabel("Percent of Predicted Box Area on Image")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
                plt.title("Box Area of Male Figures w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_box_ratio_male_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_box_ratio_male_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            for idx in range(len(prompt_list)):
                plt.plot(ws, mask_area[idx] / (512 * 512), linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                plt.ylim(0.2, 0.71)
                plt.yticks(np.arange(0.2, 0.71, step=0.1))
                plt.ylabel("Percent of Predicted Mask Area on Image")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
                plt.title("Mask Area of Male Figures w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_male_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_male_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.close()

        """plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            for idx in range(len(prompt_list)):
                plt.plot(ws, 1 - auto_pred[idx], linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                plt.ylim(0.7, 1.01)
                plt.yticks(np.arange(0.7, 1.01, step=0.05))
                plt.ylabel("Value")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
                plt.title("Accuracy of CLIP predictions on male samples w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_male_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_male_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.close()"""

    elif extend_description == "in the center":
        prompt_list = [
            "A photo of a male person in the center.",
            "A photo of a single male doctor in the center.",
            "A photo of a single male nurse practitioner in the center.",
            "A photo of a single male lab tech in the center.",
            "A photo of a single male pharmacist in the center.",
            "A photo of a single male biologist in the center.",
            "A photo of a single male chemist in the center.",
            "A photo of a single male software developer in the center.",
            "A photo of a single male author in the center.",
        ]
        box_area = [
            np.array([84883.693, 152552.997, 169600.208, 180410.091, 187729.451, 188520.965, 187156.943], dtype=float),
            np.array([108243.388, 152066.800, 160646.617, 168047.704, 169389.249, 173771.356, 175618.505], dtype=float),
            np.array([106665.196, 148065.186, 155787.649, 164397.046, 164668.033, 165610.066, 169993.650], dtype=float),
            np.array([95712.565, 128849.883, 146424.902, 148537.566, 158259.333, 162746.669, 162550.786], dtype=float),
            np.array([75909.425, 103500.544, 119343.737, 128073.161, 127053.279, 133947.018, 138346.609], dtype=float),
            np.array([103229.095, 136329.026, 144680.416, 148435.300, 147336.136, 155266.765, 146936.193], dtype=float),
            np.array([70934.939, 99822.772, 108636.692, 123009.414, 126199.836, 127887.594, 131037.824], dtype=float),
            np.array([102449.929, 140937.843, 153583.372, 160331.273, 161311.129, 163864.358, 163376.989], dtype=float),
            np.array([45325.202, 53757.118, 55311.266, 56032.287, 57617.730, 59967.470, 59733.415], dtype=float),
        ]
        mask_area = [
            np.array([53184.558, 98260.186, 108101.994, 115468.198, 118709.244, 118575.144, 117397.692], dtype=float),
            np.array([66205.248, 94507.192, 99737.3, 104686.566, 105148.374, 107987.86, 108548.348], dtype=float),
            np.array([63586.142, 89739.048, 95855.258, 100492.424, 100982.912, 102425.292, 103629.374], dtype=float),
            np.array([54850.764, 74243.696, 85776.758, 86374.074, 92458.15, 95663.172, 95013.708], dtype=float),
            np.array([44735.23, 64026.63, 74723.994, 80000.328, 80277.882, 84704.964, 87146.246], dtype=float),
            np.array([60649.162, 83373.778, 89115.194, 92586.936, 91402.304, 96421.126, 91114.344], dtype=float),
            np.array([40789.418, 60453.3, 66700.04, 75403.636, 79158.296, 79357.824, 81857.976], dtype=float),
            np.array([58152.35, 79049.822, 84945.006, 88393.242, 89148.574, 90167.124, 89005.96], dtype=float),
            np.array([25206.656, 32245.688, 33292.2, 34299.898, 34921.234, 36758.928, 36618.762], dtype=float),
        ]
        auto_pred = [
            np.array([0.26, 0.03, 0.004, 0.004, 0.008, 0.004, 0.004], dtype=float),
            np.array([0.154, 0.008, 0.0, 0.002, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.26, 0.06, 0.024, 0.022, 0.012, 0.012, 0.004], dtype=float),
            np.array([0.144, 0.026, 0.008, 0.0, 0.002, 0.004, 0.002], dtype=float),
            np.array([0.15, 0.014, 0.002, 0.004, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.17, 0.01, 0.004, 0.0, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.18, 0.05, 0.01, 0.008, 0.004, 0.002, 0.002], dtype=float),
            np.array([0.124, 0.002, 0.0, 0.0, 0.002, 0.0, 0.0], dtype=float),
            np.array([0.332, 0.162, 0.1, 0.076, 0.084, 0.06, 0.06], dtype=float),
        ]
    
        plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            for idx in range(len(prompt_list)):
                plt.plot(ws, box_area[idx] / (512 * 512), linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                plt.ylim(0.1, 0.81)
                plt.yticks(np.arange(0.1, 0.81, step=0.1))
                plt.ylabel("Percent of Predicted Box Area on Image")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
                plt.title("Box Area of Male Figures w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_box_ratio_male_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_box_ratio_male_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            for idx in range(len(prompt_list)):
                plt.plot(ws, mask_area[idx] / (512 * 512), linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                plt.ylim(0.0, 0.51)
                plt.yticks(np.arange(0.0, 0.51, step=0.1))
                plt.ylabel("Percent of Predicted Mask Area on Image")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
                plt.title("Mask Area of Male Figures w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_male_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_male_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            for idx in range(len(prompt_list)):
                plt.plot(ws, 1 - auto_pred[idx], linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                plt.ylim(0.6, 1.01)
                plt.yticks(np.arange(0.6, 1.01, step=0.05))
                plt.ylabel("Value")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
                plt.title("Accuracy of CLIP predictions on male samples w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_male_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_male_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.close()

    return

if __name__ == "__main__":
    plot_area_pred(extend_description=None)
    # plot_area_pred(extend_description="in the center")