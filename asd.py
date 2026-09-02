standard_gru = [
    0.0014232673525515168,
    0.0003094059452045672,
    0.01163366356504952,
    0.16435643787136173,
    0.2841893623370935,
    0.3484220363125943,
]

egru = [
    0.032920792726523214,
    0.034839109453086804,
    0.03756188195530731,
    0.042728960975119384,
    0.04263613941055713,
    0.04480198116069383,
]

magru = [
    0.028650990650426634,
    0.020544554752362245,
    0.02862005006282194,
    0.07670173428878926,
    0.1422339131926546,
    0.19207921063545907,
]

MGU = [
    0.0017326732977560842,
    0.0004641089178068508,
    0.01577970326075902,
    0.08641708029968904,
    0.17583539904934345,
    0.2543935700513349,
]

sLSTM = [
    0.2008522776040164,
    0.0051136364610019054,
    0.461931821974841,
    0.5659091039137407,
    0.5991477370262146,
    0.6196022846482017,
]


import matplotlib.pyplot as plt
import numpy as np

# Data

buffer_sizes = ["50", "100", "250", "500", "750", "1000"]



# Plot setup

x = np.arange(len(buffer_sizes))
width = 0.15

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)


# Create bars

rects0 = ax.bar(
    x - 2 * width,
    sLSTM,
    width,
    label="sLSTM (vector gates)",
    color="#C44E52",
    edgecolor="black",
    linewidth=0.5,
)

rects1 = ax.bar(
    x - width,
    MGU,
    width,
    label="Minimal Gated Unit",
    color="#D071D3",
    edgecolor="black",
    linewidth=0.5,
)

rects2 = ax.bar(
    x,
    standard_gru,
    width,
    label="Standard GRU",
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.5,
)

rects3 = ax.bar(
    x + width,
    magru,
    width,
    label="maGRU - Our approach",
    color="#E7812E",
    edgecolor="black",
    linewidth=0.5,
)

rects4 = ax.bar(
    x + 2 * width,
    egru,
    width,
    label="eGRU - Our approach",
    color="#55A868",
    edgecolor="black",
    linewidth=0.5,
)


# Titles and labels

ax.set_title(
    "Difference on the noise filtering task (lower = better)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

ax.set_xlabel(
    "Buffer Size",
    fontsize=12,
    fontweight="medium",
    labelpad=8,
)

ax.set_ylabel(
    "Synthetic Copy Difference",
    fontsize=12,
    fontweight="medium",
)

ax.set_xticks(x)
ax.set_xticklabels(buffer_sizes, fontsize=11)


# Aesthetics

ax.legend(fontsize=11, frameon=True)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.6,
)

ax.set_axisbelow(True)

plt.tight_layout()
plt.show()