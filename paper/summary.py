import matplotlib

matplotlib.use("Agg")
from matplotlib.pylab import plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 12,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

stats = {
    "ours": [53, 85.3, "circle"],
    "R-Score": [47, 83.0, "circle"],
    "HLoc": [7820, 92.1, "tri"],
    "GLACE": [27, 18.8, "circle"],
    "D2-Net": [2000, 80.27, "pentagon"],
    "D2-Net+SALAD": [2400, 83.72, "pentagon"],
}

shape_map = {"tri": "^", "circle": "o", "star": "*", "pentagon": "p"}

methods = list(stats.keys())
memory = [v[0] for v in stats.values()]
accuracy = [v[1] for v in stats.values()]
shapes = [v[2] for v in stats.values()]

# fig, ax = plt.subplots(figsize=(4.5, 5))  # Normal square plot
fig, ax = plt.subplots(figsize=(7, 5))  # Normal square plot

# Plot points
for m, mem, acc, shape in zip(methods, memory, accuracy, shapes):
    marker = shape_map.get(shape, "o")
    ax.scatter(
        mem, acc, marker=marker, s=100, label=m, edgecolors="black", linewidths=0.8
    )

# Axis config
ax.set_xscale("log")
ax.set_xticks([10, 100, 1000, 10000])
ax.set_xticklabels(["$10^1$", "$10^2$", "$10^3$", "$10^4$"])
ax.tick_params(axis="x", which="both", bottom=False, top=False)
ax.tick_params(axis="y", which="both", left=False, right=False)
ax.set_xlim(10, 10000)
ax.set_xlabel("Total memory (MB)", fontsize=12)
ax.set_ylabel("Accuracy (\%)", fontsize=12)
ax.set_title(r"Aachen night (0.25m, $2^\circ$)")

# Handle legends
handles, labels = ax.get_legend_handles_labels()
label_to_shape = {m: stats[m][2] for m in methods}
group1 = [
    (h, l) for h, l in zip(handles, labels) if label_to_shape[l] in ("star", "circle")
]
group2 = [(h, l) for h, l in zip(handles, labels) if label_to_shape[l] == "tri"]
group3 = [(h, l) for h, l in zip(handles, labels) if label_to_shape[l] == "pentagon"]

# Place legends *outside* the figure
legend1 = ax.legend(
    [h for h, _ in group1],
    [l for _, l in group1],
    title=r"Learning-based",
    frameon=False,
    fontsize=10,
    title_fontsize=10,
    loc="upper left",
    borderpad=0.2,
    labelspacing=0.5,
    handletextpad=0.5,
    borderaxespad=0.0,
    bbox_to_anchor=(1.02, 1.0),
)

legend2 = ax.legend(
    [h for h, _ in group2],
    [l for _, l in group2],
    title="Indirect matching",
    frameon=False,
    fontsize=10,
    title_fontsize=10,
    loc="upper left",
    borderpad=0.2,
    labelspacing=0.5,
    handletextpad=0.5,
    borderaxespad=0.0,
    bbox_to_anchor=(1.02, 0.65),  # manually adjusted to sit closer
)

legend3 = ax.legend(
    [h for h, _ in group3],
    [l for _, l in group3],
    title="Direct matching",
    frameon=False,
    fontsize=10,
    title_fontsize=10,
    loc="upper left",
    borderpad=0.2,
    labelspacing=0.5,
    handletextpad=0.5,
    borderaxespad=0.0,
    bbox_to_anchor=(1.02, 0.45),  # manually adjusted to sit closer
)

ax.add_artist(legend1)
ax.add_artist(legend2)
# ax.add_artist(legend3)

# for spine in ax.spines.values():
#     spine.set_visible(False)

# Fill background
ax.set_facecolor("lightblue")
ax.grid(True, which="major", linestyle=(0, (5, 5)), linewidth=1.0)
# fig.patch.set_facecolor("lightgray")

fig.savefig(
    "summ.png",
    format="png",
    dpi=600,
    bbox_inches="tight",  # expands canvas
)

plt.close()
