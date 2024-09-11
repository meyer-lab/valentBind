import numpy as np
import pandas as pd
from valentbind import polyc
from matplotlib import gridspec, rcParams, pyplot as plt
import seaborn as sns

rcParams["pcolor.shading"] = "auto"


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


def mixtureDF(L0, KxStar, Cplx, Kav, Rtot, Ctheta, N=2000, Lbound=True):
    """Default Cplx[0] and Cplx[1] mixture analysis"""

    C01tot = Ctheta[0] + Ctheta[1]
    r = np.arange(0, C01tot + 1e-6, step=1 / N)

    def mod(rr, single=None, Lbound=True, Rtotmul=1.0, Rtotdiv=1.0):
        if single == 0:
            Cth01 = np.array([rr, 0])
        elif single == 1:
            Cth01 = np.array([0, C01tot - rr])
        else:
            Cth01 = np.array([rr, C01tot - rr])
        Cth = np.hstack([Cth01, Ctheta[2:]])
        res = polyc(L0 * np.sum(Cth), KxStar, Rtot * Rtotmul / Rtotdiv, Cplx, Cth, Kav)
        return sum(res[0] if Lbound else res[1][:, 2])  # R3 bound

    def caseDF(single):
        return pd.DataFrame(
            {
                "x": r,
                "y": [mod(rr, single, Lbound) for rr in r],
                "ymin": [mod(rr, single, Lbound, Rtotmul=0.9) for rr in r],
                "ymax": [mod(rr, single, Lbound, Rtotdiv=0.9) for rr in r],
                "Ligand": [
                    ("Single " + str(Cplx[single, :]))
                    if single is not None
                    else "Mixture"
                ]
                * len(r),
            }
        )

    df = pd.concat([caseDF(None), caseDF(0), caseDF(1)])
    df = pd.melt(df, id_vars=["x", "Ligand"])
    return df


def mixtureFig(ax, Lbound=True):
    L0 = 1e-9
    KxStar = 1e-12
    Cplx = np.array([[2, 0], [1, 1], [1, 0]])
    Kav = np.array([[1e8, 1e5, 6e5], [3e5, 1e7, 1e6]])
    Rtot = np.array([2.5e4, 3e4, 2e3])
    Ctheta = [0.8, 0, 0.2]
    C01tot = Ctheta[0] + Ctheta[1]

    demodata = pd.DataFrame(
        {
            "x": np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            * C01tot,
            "y": np.array(
                [
                    3e3,
                    4.1e3,
                    4.4e3,
                    5.2e3,
                    5.7e3,
                    5.5e3,
                    3e3,
                    4.9e3,
                    5e3,
                    5.6e3,
                    5.8e3,
                    5.5e3,
                ]
            ),
            "Cases": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
        }
    )

    df = mixtureDF(L0, KxStar, Cplx, Kav, Rtot, Ctheta, Lbound=Lbound)

    sns.lineplot(data=df, x="x", y="value", hue="Ligand", ax=ax)

    ax.set_xlabel("Mixture Composition")
    if Lbound:
        ax.set_ylabel("Predicted ligand binding")
        ax.set_title("Predicted ligand binding")
        sns.scatterplot(
            data=demodata,
            x="x",
            y="y",
            hue="Cases",
            style="Cases",
            palette=sns.color_palette(["#9925be", "#be4d25"]),
            ax=ax,
        )
    else:
        ax.set_ylabel("Predicted $R_3$ binding")
        ax.set_title("Predicted $R_3$ binding")

    xs = np.arange(0, C01tot + 1e-5, C01tot / 4)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            "[2 0] {} nM\n[1 1] {} nM".format(
                np.round(n, decimals=2), np.round(C01tot - n, decimals=2)
            )
            for n in xs
        ]
    )
    ax.set_ylim(
        0,
    )


sns.set_theme(font_scale=2)
ax, f = getSetup((8, 4), (1, 2))
mixtureFig(ax[0])
h, l = ax[0].get_legend_handles_labels()
l1 = ax[0].legend(h[:3], l[:3], title="Ligand", loc="center right")
l2 = ax[0].legend(h[3:], l[3:], title="Cases", loc="lower center")

mixtureFig(ax[1], False)

f.savefig("figure3.pdf", dpi=f.dpi * 2)
