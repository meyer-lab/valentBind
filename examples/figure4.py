import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from matplotlib import gridspec, rcParams
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

from valentbind import polyc


def getSetup(figsize, gridd):
    """Establish figure set-up with subplots."""
    sns.set_theme(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    ax = list()
    for x in range(gridd[0] * gridd[1]):
        ax.append(f.add_subplot(gs1[x]))

    return (ax, f)


hfont = {"fontname": "Helvetica", "size": 22}
rcParams["pcolor.shading"] = "auto"


def heatmap(
    ax, L0, KxStar, Kav, Comp, Cplx, vrange=(-2, 4), title="", mode=0, overlay=True
):
    nAbdPts = 70
    abundRange = (1.5, 4.5)
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)

    norm = Normalize(
        vmin=vrange[0], vmax=vrange[1]
    )  # Use Normalize instead of plt.Normalize
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap="YlGnBu"), ax=ax)

    if mode == 0:
        title += " total ligand bound"
        cbar.set_label("Log Ligand Bound")
        func = np.vectorize(
            lambda abund1, abund2: np.sum(
                polyc(L0, KxStar, [abund1, abund2, 2e3], Cplx, Comp, Kav)[0]
            )
        )
    elif mode == 1:
        title += " $R_2$ bound"
        cbar.set_label("Log Receptor Bound")
        func = np.vectorize(
            lambda abund1, abund2: polyc(
                L0, KxStar, [abund1, abund2, 2e3], Cplx, Comp, Kav
            )[1][0, 1]
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))

    contours = ax.contour(
        X, Y, logZ, levels=np.arange(-20, 20, 0.5), colors="black", linewidths=0.5
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Expression of $R_1$")
    ax.set_ylabel("Expression of $R_2$")
    ax.set_title(title)
    plt.clabel(contours, inline=True, fontsize=6)
    ax.pcolor(X, Y, logZ, cmap="YlGnBu", vmin=vrange[0], vmax=vrange[1])

    if overlay:
        ax_new = ax.twinx().twiny()
        ax_new.set_xscale("linear")
        ax_new.set_yscale("linear")
        ax_new.set_xticks([])
        ax_new.set_yticks([])
        ax_new.set_xlim(abundRange)
        ax_new.set_ylim(abundRange)
        if mode == 0:
            item = [3.0, 3.9, 0.75, 0.25, 0]
            ax_new.add_patch(
                Ellipse(
                    xy=(item[0], item[1]),
                    width=item[2],
                    height=item[3],
                    angle=item[4],
                    facecolor="red",
                    fill=True,
                    alpha=0.8,
                    linewidth=1,
                )
            )
        if mode == 1:
            xs, ys = sample_cell(3, 4, size=1000)
            ax_new.plot(xs, ys, ".", color="red", markersize=1)


def sample_cell(x, y, size=100):
    xs = np.random.lognormal(np.log(x), sigma=0.09, size=size)
    ys = np.random.lognormal(np.log(y), sigma=0.04, size=size)
    return xs, ys


ax, f = getSetup((8, 4), (1, 2))


L0 = 1e-9
KxStar = 1e-12
Cplx = np.array([[2, 0]])
Kav = np.array([[1e8, 1e5, 6e5], [3e5, 1e7, 1e6]])

heatmap(ax[0], L0, KxStar, Kav, [1.0], Cplx, mode=0, title="[2,0]", vrange=(2, 8))
heatmap(ax[1], L0, KxStar, Kav, [1.0], Cplx, mode=1, title="[2,0]", vrange=(-4, 2))

ax[0].text(
    -0.2, 1.25, "a", transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top"
)
ax[1].text(
    -0.2, 1.25, "b", transform=ax[1].transAxes, fontsize=16, fontweight="bold", va="top"
)

f.savefig("figure4.pdf", dpi=f.dpi * 2)
