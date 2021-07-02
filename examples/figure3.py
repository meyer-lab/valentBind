import numpy as np
import pandas as pd
from valentbind import polyc, polyfc
from matplotlib import gridspec, rcParams, pyplot as plt
import seaborn as sns

rcParams['pcolor.shading'] = 'auto'

def getSetup(figsize, gridd):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    ax = list()
    for x in range(gridd[0] * gridd[1]):
        ax.append(f.add_subplot(gs1[x]))

    return (ax, f)

def mixtureFig(ax, Lbound=True):
    L0 = 1e-9
    KxStar = 1e-12
    Cplx = np.array([[2, 0], [1, 1]])
    Kav = np.array([[1e8, 1e5, 6e5], [3e5, 1e7, 1e6]])
    Rtot = np.array([2.5e4, 3e4, 2e3])

    N = 100
    r = np.arange(0, 1.001, step=1/N)

    def polymod1(r, Rtotx):
        arr = np.zeros_like(r)
        for i in range(len(arr)):
            if Lbound:
                arr[i] = sum(polyc(L0 * r[i], KxStar, Rtotx, Cplx[[0], :], [1.0], Kav)[0])
            else:
                arr[i] = sum(polyc(L0 * r[i], KxStar, Rtotx, Cplx[[0], :], [1.0], Kav)[1][:, 2])
        return arr

    def polymod2(r, Rtotx):
        arr = np.zeros_like(r)
        for i in range(len(arr)):
            if Lbound:
                arr[i] = sum(polyc(L0 * (1-r[i]), KxStar, Rtotx, Cplx[[1], :], [1.0], Kav)[0])
            else:
                arr[i] = sum(polyc(L0 * (1-r[i]), KxStar, Rtotx, Cplx[[1], :], [1.0], Kav)[1][:, 2])
        return arr

    def polymodm(r, Rtotx):
        arr = np.zeros_like(r)
        for i in range(len(arr)):
            if Lbound:
                arr[i] = sum(polyc(L0, KxStar, Rtotx, Cplx, [r[i], 1-r[i]], Kav)[0])
            else:
                arr[i] = sum(polyc(L0, KxStar, Rtotx, Cplx, [r[i], 1-r[i]], Kav)[1][:, 2])
        return arr

    singleOne = pd.DataFrame({
        "x": r,
        "y": polymod1(r, Rtot),
        "ymin": polymod1(r, Rtot * 0.9),
        "ymax": polymod1(r, Rtot / 0.9),
        "Ligand": ["Single [2, 0]"] * len(r)
    })


    singleTwo = pd.DataFrame({
        "x": r,
        "y": polymod2(r, Rtot),
        "ymin": polymod2(r, Rtot * 0.9),
        "ymax": polymod2(r, Rtot / 0.9),
        "Ligand": ["Single [1, 1]"] * len(r)
    })

    mixture = pd.DataFrame({
        "x": r,
        "y": polymodm(r, Rtot),
        "ymin": polymodm(r, Rtot * 0.9),
        "ymax": polymodm(r, Rtot / 0.9),
        "Ligand": ["Mixture"] * len(r)
    })

    demodata = pd.DataFrame({"x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                         "y": [3.18e3, 4.2e3, 5.2e3, 5.3e3, 6.1e3, 5.97e3, 3.18e3, 5.5e3, 5.5e3, 6.3e3, 6.4e3, 5.97e3],
                         "Cases": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"]})

    df = pd.concat([mixture, singleOne, singleTwo])
    df = pd.melt(df, id_vars=['x', 'Ligand'])

    sns.lineplot(data=df, x="x", y="value", hue="Ligand", ax=ax)

    ax.set_xlabel("Mixture Composition")
    if Lbound:
        ax.set_ylabel("Predicted ligand binding")
        ax.set_title("Predicted ligand binding")
        sns.scatterplot(data=demodata, x="x", y="y", hue="Cases", style="Cases", ax=ax)
    else:
        ax.set_ylabel("Predicted $R_3$ binding")
        ax.set_title("Predicted $R_3$ binding")

    xs = np.arange(0, 1.01, 0.25)
    ax.set_xticks(xs)
    ax.set_xticklabels(["[2 0] {}%\n[1 1] {}%".format(np.round(n*100), np.round(100-n*100)) for n in xs])


ax, f = getSetup((8, 4), (1, 2))
mixtureFig(ax[0])
mixtureFig(ax[1], False)

ax[0].text(-0.2, 1.25, "a", transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")
ax[1].text(-0.2, 1.25, "b", transform=ax[1].transAxes, fontsize=16, fontweight="bold", va="top")

f.savefig('figure3.pdf', dpi=f.dpi*2)
