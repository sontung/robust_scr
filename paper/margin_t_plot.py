import re
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.pylab import plt


def find_numbers(string_):
    pattern = r"[-+]?(?:\d*\.*\d+)"
    matches = re.findall(pattern, string_)
    if len(matches) == 0:
        return 0, 0
    numbers = list(map(float, matches))
    print(numbers[:3], numbers[3:])
    return sum(numbers[:3]) / 3, sum(numbers[3:]) / 3


def main():
    ds = {
        "0.2": "79.9 / 90.9 / 96.8	70.4 / 92.9 / 99.0",
        "0.3": "79.6 / 91.3 / 97.0	68.4 / 94.9 / 99.0",
        "0.4": "80.8 / 90.4 / 97.0	71.4 / 91.8 / 99.0",
        "0.5": "80.1 / 89.8 / 97.0	72.4 / 90.8 / 99.0",
        "0.6": "80.5 / 89.9 / 96.7	69.4 / 93.9 / 99.0",
        "0.7": "80.7 / 90.5 / 96.8	67.3 / 93.9 / 99.0",
        "0.8": "80.3 / 91.1 / 97.0	71.4 / 91.8 / 99.0",
        "0.9": "80.8 / 91.0 / 97.0	71.4 / 94.9 / 99.0",
    }
    day = []
    night = []
    for param_ in ds:
        res = ds[param_]
        d, n = find_numbers(res)
        day.append(d)
        night.append(n)

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 12,  # Set the global font size
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

    plt.figure(figsize=(5, 5))
    # plt.figure(figsize=(4, 2))
    plt.plot(np.arange(2, 10) / 10, day, label="Daytime images", marker="o")
    plt.plot(np.arange(2, 10) / 10, night, label="Nighttime images", marker="s")
    # plt.ylim(0, 100)
    plt.xticks(np.arange(2, 10) / 10)
    plt.xlabel(r"$\tau$")
    plt.ylabel("\% successfully localized images")
    # plt.ylabel("Acc.")
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Add vertical space

    plt.savefig(
        "/home/n11373598/work/phd_thesis/chapters/chapter5/figures/drawings/margin_tau.pdf", format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.1
    )


if __name__ == "__main__":
    main()
