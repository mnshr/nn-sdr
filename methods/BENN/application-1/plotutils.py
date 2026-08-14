import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.ticker import FormatStrFormatter





## lazy to define a class to include two plot function
def prop_plot(eigenvalues, ax=None, figsize=(6.4, 4.8)):
    if ax is None:
        # get current axes if not provided
        ax=plt.figure(figsize=figsize).gca()
    colormap = mpl.colormaps['Set1']
    eigenvalues = np.sort(np.abs(np.real(eigenvalues)))[::-1]
    eigenvalues = eigenvalues[:np.sum(eigenvalues > 1e-6) + 1]
    prop = eigenvalues / np.sum(eigenvalues, axis=0)
    cums = np.cumsum(eigenvalues / np.sum(eigenvalues, axis=0))
    dim = eigenvalues.shape[0]

    ax.plot(np.arange(dim) + 1, prop)
    d1 = np.where(cums > 0.90)[0][0] + 1
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.hlines(y=prop[d1-1], xmin=xl[0], xmax=d1, color=colormap(0), linestyle='--')
    ax.vlines(x=d1, ymin=yl[0], ymax=prop[d1-1], color=colormap(0), linestyle='--')
    

    d2 = np.where(cums > 0.95)[0][0] + 1
    if d1 != d2:
        ax.hlines(y=prop[d2-1], xmin=xl[0], xmax=d2, color=colormap(1), linestyle='--')
        ax.set_xlim(xl)
        ax.vlines(x=d2, ymin=yl[0], ymax=prop[d2-1], color=colormap(1), linestyle='--')
        

    d3 = d2 + 1
    ax.hlines(y=prop[d3-1], xmin=xl[0], xmax=d3, color=colormap(2), linestyle='--')
    yt = ax.get_yticks()
    threshold_y = (yt[-1] - yt[0]) / len(yt) / 2
    yt = yt[abs(yt - prop[d1-1]) > threshold_y]
    yt = yt[abs(yt - prop[d2-1]) > threshold_y]
    yt = yt[abs(yt - prop[d3-1]) > threshold_y]
    _ = ax.set_yticks(np.sort(np.concatenate([yt, np.unique(np.array([prop[d1-1],prop[d2-1],prop[d3-1]]))])))
    ax.vlines(x=d3, ymin=yl[0], ymax=prop[d3-1], color=colormap(2), linestyle='--')
    xt = np.sort(np.concatenate([ax.get_xticks()[1:-1], np.array([d1, d2, d3])]))
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_xticks(np.unique(xt.astype(int)))
    ax.set_ylabel("Proportion of each eigenvalue")
    for text_ in ax.get_xticklabels():
        xpos = text_.get_position()[0]
        if xpos == d1:
            text_.set_color(colormap(0))
        elif xpos == d2:
            text_.set_color(colormap(1))
        elif xpos == d3:
            text_.set_color(colormap(2))

    
def cum_plot(eigenvalues, ax=None, figsize=(6.4, 4.8)):
    # ax axes returned by plt
    if ax is None:
        # get current axes if not provided
        ax=plt.figure(figsize=figsize).gca()
    eigenvalues = np.sort(np.abs(np.real(eigenvalues)))[::-1]
    eigenvalues = eigenvalues[:np.sum(eigenvalues > 1e-6) + 1]
    cums = np.cumsum(eigenvalues / np.sum(eigenvalues, axis=0))
    dim = eigenvalues.shape[0]
    colormap = mpl.colormaps['Set1']

    ax.plot(np.arange(dim) + 1, cums)
    d1 = np.where(cums > 0.90)[0][0] + 1
    ax.plot(np.arange(dim) + 1, cums)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.hlines(y=cums[d1-1], xmin=xl[0], xmax=d1, color=colormap(0), linestyle='--')
    ax.vlines(x=d1, ymin=yl[0], ymax=cums[d1-1], color=colormap(0), linestyle='--')

    d2 = np.where(cums > 0.95)[0][0] + 1
    if d1 != d2:
        ax.hlines(y=cums[d2-1], xmin=xl[0], xmax=d2, color=colormap(1), linestyle='--')
        ax.vlines(x=d2, ymin=yl[0], ymax=cums[d2-1], color=colormap(1), linestyle='--')
    
    yt = ax.get_yticks()[1:]
    threshold_y = (yt[-1] - yt[0]) / len(yt) / 2
    yt = yt[abs(yt - cums[d1-1]) > threshold_y]
    yt = yt[abs(yt - cums[d2-1]) > threshold_y]
    _ = ax.set_yticks(np.sort(np.concatenate([yt, np.unique(np.array([cums[d1-1],cums[d2-1]]))])))
    
    ax.set_xlim(xl)        
    ax.set_ylim(yl)

    xt = np.sort(np.concatenate([ax.get_xticks()[1:-1], np.array([d1, d2])]))
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_xticks(np.unique(xt.astype(int)))
    ax.set_ylabel("Cumulative proportion of eigenvalue")
    for text_ in ax.get_xticklabels():
        xpos = text_.get_position()[0]
        if xpos == d1:
            text_.set_color(colormap(0))
        elif xpos == d2:
            text_.set_color(colormap(1))



def visualization_2d(features, y, samples_size, save_file_name=None, figsize = (12, 10)):
   
    bestindx, bestacc = find_bestidx(features, y)

    cmap = mpl.cm.tab10
    if samples_size == y.shape[0]: # full sample visualization
        fig, axx = plt.subplots(1, 1, figsize=figsize, squeeze = False)
        ax = axx[0, 0]
        visualize_2d_(features[:, bestindx], y, ax, cmap)
        lgd = ax.legend(bbox_to_anchor =(1.3, 0.9))
    else:
        stratified_split = StratifiedShuffleSplit(n_splits=10, train_size=samples_size)
        fig, axx = plt.subplots(2, 2, figsize=figsize, squeeze = False)
        plt.subplots_adjust(wspace=0.5)
        for j in range(4):
            ax = axx[j//2, j%2]
            train_indices, _ = next(stratified_split.split(y, y))
            visualize_2d_(features[train_indices][:, bestindx], y[train_indices], ax, cmap)
            lgd = ax.legend(bbox_to_anchor =(1.1, 0.9))

    if save_file_name is not None:
        plt.savefig(save_file_name, bbox_extra_artists=(lgd, ), bbox_inches='tight')


def visualize_2d_(features, y, ax, cmap):
    # visualize the features wrt y. features is a n*2 matrix. 
    # print()
    for i in range(10):
        numberi_idx = np.where(y == i)
        smp = features[numberi_idx]
        sc = ax.scatter(smp[:, 0], smp[:, 1], s=50, color=cmap(i), label=f"{i}")


def find_bestidx(features, y):
    bestindx = [0, 1]
    bestacc = 0
    for i in range(features.shape[1]):
        for j in range(i + 1, features.shape[1]):
            lg = LogisticRegression(multi_class="multinomial")
            lg.fit(features[:, [i, j]], y)
            if accuracy_score(y, lg.predict(features[:, [i, j]])) > bestacc:
                bestindx = [i, j]
                bestacc = accuracy_score(y, lg.predict(features[:, [i, j]]))
    return bestindx, bestacc



def compare_plot(pred1, pred2, y, samples_size, save_file_name=None, subplot_num=1, figsize = (12, 10)):
    ## figsize is perfered as (12, 10 * subplot_num)
    plt.subplots_adjust(wspace=0.5)
    bestindx1, bestacc1 = find_bestidx(pred1, y)
    bestindx2, bestacc2 = find_bestidx(pred2, y)

    fig, axx = plt.subplots(subplot_num, 2, figsize=figsize, squeeze = False)
    cmap = mpl.cm.tab10
    for j in range(subplot_num):
        ax = axx[j, 0]
        ax2 = axx[j, 1]
        if samples_size == y.shape[0]:
            visualize_2d_(pred1[:, bestindx1], y, ax, cmap)
            visualize_2d_(pred2[:, bestindx2], y, ax2, cmap)
        else:
            stratified_split = StratifiedShuffleSplit(n_splits=max(10, subplot_num), train_size=samples_size)
            train_indices, _ = next(stratified_split.split(y, y))
            visualize_2d_(pred1[train_indices][:, bestindx1], y[train_indices], ax, cmap)
            visualize_2d_(pred2[train_indices][:, bestindx2], y[train_indices], ax2, cmap)



        lgd = ax.legend(bbox_to_anchor =(1.1, 0.9))
        lgd2 = ax2.legend(bbox_to_anchor =(1.1, 0.9))

    if save_file_name is not None:
        plt.savefig(save_file_name, bbox_extra_artists=(lgd,lgd2), bbox_inches='tight')

