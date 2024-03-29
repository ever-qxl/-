

# coding: utf-8

# NB

# get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import preprocessing
from django.views.generic import TemplateView
import os
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB



# code is borrowed from @kwinkunks
###
def compare_facies_plot(logs, compadre, facies_colors):
    # make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
        facies_colors[0:len(facies_colors)], 'indexed')

    ztop = logs.Depth.min()
    zbot = logs.Depth.max()

    cluster1 = np.repeat(np.expand_dims(logs['Facies'].values, 1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs[compadre].values, 1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(9, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im1 = ax[5].imshow(cluster1, interpolation='none', aspect='auto',
                       cmap=cmap_facies, vmin=1, vmax=9)
    im2 = ax[6].imshow(cluster2, interpolation='none', aspect='auto',
                       cmap=cmap_facies, vmin=1, vmax=9)

    divider = make_axes_locatable(ax[6])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label((17 * ' ').join([' SS ', 'CSiS', 'FSiS',
                                    'SiSh', ' MS ', ' WS ', ' D  ',
                                    ' PS ', ' BS ']))
    cbar.set_ticks(range(0, 1))
    cbar.set_ticklabels('')

    for i in range(len(ax) - 2):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)

    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(), logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(), logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(), logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(), logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(), logs.PE.max())
    ax[5].set_xlabel('Facies')
    ax[6].set_xlabel(compadre)

    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    ax[6].set_xticklabels([])
    f.suptitle('Well: %s' % logs.iloc[0]['Well Name'], fontsize=14, y=0.94)

###
