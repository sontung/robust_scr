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
        "128": "80.1 / 89.8 / 97.0	72.4 / 90.8 / 99.0",
        "64": "80.7 / 90.7 / 97.0	70.4 / 91.8 / 99.0",
        "32": "79.6 / 90.2 / 96.1	70.4 / 87.8 / 94.9",
        "16": "	73.5 / 86.0 / 93.1	53.1 / 69.4 / 84.7",
        "8": "	56.3 / 67.1 / 82.8	25.5 / 36.7 / 45.9",
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
    x = [0, 1, 2, 3, 4]  # equally spaced
    labels = [128, 64, 32, 16, 8]
    plt.figure(figsize=(5, 5))
    plt.plot(x, day, label="Daytime images", marker="o")
    plt.plot(x, night, label="Nighttime images", marker="s")
    # plt.ylim(0, 100)
    plt.xticks(x, labels)
    plt.xlabel(r"PCA dimension")

    plt.ylabel("\% successfully localized images")
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Add vertical space

    plt.savefig(
        "/home/n11373598/work/phd_thesis/chapters/chapter5/figures/drawings/pca_dim.pdf", format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.1
    )


if __name__ == "__main__":
    main()
