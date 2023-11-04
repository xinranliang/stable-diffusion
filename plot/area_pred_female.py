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
            "A photo of a female person.",
            "A photo of a single female doctor.",
            "A photo of a single female nurse practitioner.",
            "A photo of a single female lab tech.",
            "A photo of a single female pharmacist.",
            "A photo of a single female biologist.",
            "A photo of a single female chemist.",
            "A photo of a single female software developer.",
            "A photo of a single female author.",
        ]

        box_area = [
            np.array([118209.112, 191874.162, 205435.169, 209874.653, 208413.185, 205952.420, 211865.394], dtype=float),
            np.array([147584.567, 192224.841, 195204.145, 202773.566, 200900.319, 204039.097, 202632.288], dtype=float),
            np.array([145920.490, 185169.867, 194163.279, 197082.921, 199129.147, 198028.038, 197317.019], dtype=float),
            np.array([123186.037, 168060.796, 183320.710, 188967.634, 191913.747, 192755.700, 196897.527], dtype=float),
            np.array([100516.248, 128006.209, 140468.686, 147701.274, 149342.196, 151540.749, 152527.920], dtype=float),
            np.array([120604.069, 155418.671, 164592.888, 171268.145, 167615.838, 172525.383, 170591.534], dtype=float),
            np.array([109675.563, 151310.569, 162658.465, 171365.604, 174726.846, 170133.478, 178536.254], dtype=float),
            np.array([121357.119, 159016.452, 163686.126, 168692.977, 166922.107, 168931.518, 167145.385], dtype=float),
            np.array([138903.458, 184713.659, 193864.159, 200405.707, 202629.630, 202422.372, 204864.056], dtype=float),
        ]
        mask_area = [
            np.array([81371.950, 138144.144, 145771.430, 150207.480, 147326.502, 148218.830, 151300.888], dtype=float),
            np.array([97749.032, 127225.242, 128634.092, 132626.464, 130630.492, 131614.046, 130977.752], dtype=float),
            np.array([90873.868, 119269.282, 127300.124, 127753.876, 129835.69, 128362.622, 127449.168], dtype=float),
            np.array([74377.3, 99271.09, 111089.674, 113999.984, 115251.234, 116522.904, 118182.276], dtype=float),
            np.array([60416.3, 81829.73, 91332.142, 96031.514, 96582.054, 98552.706, 98775.244], dtype=float),
            np.array([70319.442, 94719.206, 98822.144, 103518.856, 100174.282, 103256.772, 102340.268], dtype=float),
            np.array([65328.542, 89540.124, 96785.12, 102973.864, 106208.818, 103163.536, 107950.152], dtype=float),
            np.array([69693.522, 94833.166, 96060.872, 98088.14, 96212.0, 97077.524, 97707.666], dtype=float),
            np.array([88640.576, 126578.45, 134683.934, 140170.494, 141455.448, 143187.704, 145882.79], dtype=float),
        ]
        auto_pred = [
            np.array([0.822, 0.99, 0.998, 0.998, 0.998, 0.998, 0.99], dtype=float),
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
                plt.title("Box Area of Female Figures w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_box_ratio_female_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_box_ratio_female_extendFalse.png", dpi=300, bbox_inches="tight")
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
                plt.title("Mask Area of Female Figures w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_female_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_female_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            for idx in range(len(prompt_list)):
                plt.plot(ws, auto_pred[idx], linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                plt.ylim(0.6, 1.01)
                plt.yticks(np.arange(0.6, 1.01, step=0.05))
                plt.ylabel("Value")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
                plt.title("Accuracy of CLIP predictions on female samples w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_female_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_female_extendFalse.png", dpi=300, bbox_inches="tight")
        plt.close()

    elif extend_description == "in the center":
        prompt_list = [
            "A photo of a female person in the center.",
            "A photo of a single female doctor in the center.",
            "A photo of a single female nurse practitioner in the center.",
            "A photo of a single female lab tech in the center.",
            "A photo of a single female pharmacist in the center.",
            "A photo of a single female biologist in the center.",
            "A photo of a single female chemist in the center.",
            "A photo of a single female software developer in the center.",
            "A photo of a single female author in the center.",
        ]
        box_area = [
            np.array([77290.622, 133465.010, 151929.430, 168968.337, 168720.075, 172724.105, 167270.158], dtype=float),
            np.array([107013.157, 151777.953, 158366.066, 161175.897, 169789.670, 172576.281, 168361.111], dtype=float),
            np.array([114170.738, 146221.221, 159992.419, 161477.243, 166416.622, 172077.643, 174934.907], dtype=float),
            np.array([96360.385, 144383.374, 153962.099, 158532.169, 164969.941, 166913.602, 171454.149], dtype=float),
            np.array([81654.434, 113028.793, 123899.513, 127737.723, 139431.492, 137348.625, 144631.355], dtype=float),
            np.array([87130.500, 121339.482, 121010.474, 126625.960, 132861.134, 133265.370, 135560.859], dtype=float),
            np.array([84959.046, 120174.374, 128835.137, 139934.110, 145209.387, 151266.682, 152012.079], dtype=float),
            np.array([91512.452, 128573.334, 131933.446, 137953.298, 142804.610, 141943.485, 139769.335], dtype=float),
            np.array([43366.465, 56639.549, 54439.685, 56389.149, 58574.846, 59421.741, 59928.073], dtype=float),
        ]
        mask_area = [
            np.array([48186.748, 89702.796, 101462.660, 114453.114, 114788.866, 118145.546, 113460.940], dtype=float),
            np.array([67023.986, 96924.802, 103607.702, 104038.97, 109265.978, 110815.95, 108093.312], dtype=float),
            np.array([69389.734, 91904.494, 102056.592, 103462.3, 106323.0, 109456.846, 112193.746], dtype=float),
            np.array([56916.438, 85472.6, 91414.564, 95482.44, 99112.254, 100876.442, 103219.326], dtype=float),
            np.array([49567.124, 73565.64, 81152.184, 84199.492, 91954.124, 90401.996, 94757.962], dtype=float),
            np.array([49886.464, 72790.78, 74295.492, 77830.806, 80950.03, 82280.732, 83478.104], dtype=float),
            np.array([49680.676, 73303.46, 79723.562, 86567.14, 90077.762, 93010.43, 93893.556], dtype=float),
            np.array([52232.586, 73287.57, 74913.602, 78330.174, 80703.906, 78504.176, 77805.602], dtype=float),
            np.array([24309.964, 33770.932, 33914.632, 35358.296, 36742.416, 37625.902, 38270.58], dtype=float),
        ]
        auto_pred = [
            np.array([0.752, 0.958, 0.992, 0.992, 0.992, 0.994, 0.992], dtype=float),
            np.array([0.784, 0.994, 1.0, 0.996, 1.0, 1.0, 1.0], dtype=float),
            np.array([0.836, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
            np.array([0.748, 0.982, 1.0, 0.996, 0.996, 1.0, 1.0], dtype=float),
            np.array([0.782, 0.99, 0.994, 1.0, 0.996, 1.0, 1.0], dtype=float),
            np.array([0.76, 0.97, 0.986, 0.986, 0.974, 0.97, 0.968], dtype=float),
            np.array([0.71, 0.972, 0.988, 0.996, 0.988, 0.986, 0.994], dtype=float),
            np.array([0.76, 0.976, 0.994, 0.998, 0.998, 0.998, 1.0], dtype=float),
            np.array([0.714, 0.988, 0.996, 0.994, 0.998, 1.0, 0.996], dtype=float),
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
                plt.title("Box Area of Female Figures w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_box_ratio_female_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_box_ratio_female_extendTrue.png", dpi=300, bbox_inches="tight")
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
                plt.title("Mask Area of Female Figures w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_female_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/sbv2_mask_ratio_female_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            for idx in range(len(prompt_list)):
                plt.plot(ws, auto_pred[idx], linestyle="-", marker='o', color=colors[idx], label=prompt_list[idx])

                plt.xticks(ws, ws)
                plt.xlabel("Scale of classifier-free guidance ($w$)")
                plt.ylim(0.6, 1.01)
                plt.yticks(np.arange(0.6, 1.01, step=0.05))
                plt.ylabel("Value")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
                plt.title("Accuracy of CLIP predictions on female samples w.r.t Guidance on SD-v2")

        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_female_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.savefig("/n/fs/xl-diffbia/projects/stable-diffusion/logs/figures/clip_auto_pred_female_extendTrue.png", dpi=300, bbox_inches="tight")
        plt.close()

    return

if __name__ == "__main__":
    plot_area_pred(extend_description=None)
    plot_area_pred(extend_description="in the center")