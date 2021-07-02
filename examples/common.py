"""
Contains utilities and functions that are commonly used in the figure creation files.
"""
from string import ascii_lowercase
from matplotlib import gridspec, pyplot as plt
from matplotlib import rcParams
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


def subplotLabel(axs, indices=False):
    """ Place subplot labels on figure. """
    if not indices:
        for ii, ax in enumerate(axs):
            ax.text(-0.2, 1.25, ascii_lowercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    else:
        for jj, index in enumerate(indices):
            axs[index].text(-0.2, 1.25, ascii_lowercase[jj], transform=axs[index].transAxes, fontsize=16, fontweight="bold", va="top")
