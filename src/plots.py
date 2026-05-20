"""
    All plotting functions are found in this Script.
"""


import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D

import numpy as np

import os

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.utils import Number2LabelMapper

## Parameters

RANDOMS_SEED = 42


## Functions

def _plot_hexagonal_distance_map (som,
                                 X,
                                 y,
                                 label_dict,
                                 fig_title,
                                 items2plot = 1500,
                                 ax=None):
    """
        som: Minisom object
        X: training data
        y: label data
        label_dict: A dictionary containing the label information
        savefig_path: DIRECTORY to save the figure
        figname: Name for the figure.
        figtitle: Title for the figure
        items2plot: number of items to plot, default:1500. This is the fisrt 1500 items that are plotted.
        This is more suited for MNIST-like datasets, where the label is given in integers and it is easy to fetch the label data from the dataset.
    """
    xx, yy = som.get_euclidean_coordinates()
    umatrix = som.distance_map()
    weights = som.get_weights()

    if ax is None:
        f = plt.figure(figsize=(10,10))
        ax = f.add_subplot(111)

    ax.set_aspect('equal')

    # Get the hexagonal patch
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            hex = RegularPolygon((xx[(i, j)], wy),
                                 numVertices=6,
                                 radius=.95 / np.sqrt(3),
                                 facecolor=cm.Blues(umatrix[i, j]),
                                 alpha=.4,
                                 edgecolor='black')
            ax.add_patch(hex)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+']
    colors = [f"C{i}" for i in range(len(markers))]

    # Plot the labels

    for image, label in zip(X[:items2plot], y[:items2plot]):
        w = som.winner (image)                          # BMU
        wx, wy = som.convert_map_to_euclidean(w)
        wy *= np.sqrt(3) / 2
        ax.plot(wx, wy,
                 markers[int(label)],
                 markerfacecolor='None',
                 markeredgecolor=colors[int(label)],
                 markersize=8,
                 markeredgewidth=2)

    # Set x and y axis
    upper_x_lim = weights.shape[0] + 1
    upper_y_lim = weights.shape[1] - 4
    xrange = np.arange(-1, upper_x_lim)
    yrange = np.arange(-1, upper_y_lim)
    ax.set_xticks(xrange-.5, xrange)
    ax.set_yticks(yrange * np.sqrt(3) / 2, yrange)

    # Add colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right",size="5%", pad=0.05)
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues,
                                orientation='vertical', alpha=.4)
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                      rotation=270, fontsize=16)

    #Labels
    legend_elements = [ Line2D([0], [0], marker=marker, color=color, label=label,
                               markerfacecolor='w', markersize=8, linestyle='None', markeredgewidth=2)
                        for marker,color,label in zip(markers, colors, label_dict.values())]

    ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.15), loc='upper left',
          borderaxespad=0., ncol=3, fontsize=8)

    ax.set_title (fig_title)


def _plot_hexagonal_heatmap (som,
                            X,
                            figtitle,
                            ax=None)-> None:
    """
        som: Minisom object
        X: training data.
        savefig_path: DIRECTORY to save the figure
        figname: Name for the figure.
        figtitle: Title for the figure
    """
    xx, yy = som.get_euclidean_coordinates()
    weights = som.get_weights()
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,10))

    ax.set_aspect('equal')
    frequencies  = som.activation_response(X)
    # Get the hexagonal patch
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            hex = RegularPolygon((xx[(i, j)], wy),
                                 numVertices=6,
                                 radius=.95 / np.sqrt(3),
                                 facecolor=cm.Reds(frequencies[i, j]),
                                 alpha=.4,
                                 edgecolor='black')
            ax.add_patch(hex)


    # Set x and y axis
    upper_x_lim = weights.shape[0] + 1
    upper_y_lim = weights.shape[1] - 3
    xrange = np.arange(-1, upper_x_lim)
    yrange = np.arange(-1, upper_y_lim)
    ax.set_xticks(xrange-.5, xrange)
    ax.set_yticks(yrange * np.sqrt(3) / 2, yrange)

    # Add colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right",size="5%", pad=0.05)
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Reds,
                                orientation='vertical', alpha=.4)
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel('frequency',
                      rotation=270, fontsize=16)


    ax.set_title (figtitle)



def subplots_uheat_maps (som,
                         X,
                         y ,
                         label_dict,
                         savefig_path,
                         figname_umap,
                         figname_heatmap,
                         figname_all,
                         figtitle,
                         item2plot=1500):
    """
        Subplots both distance and heat map in [1,2]
    """

    fig, axis = plt.subplots(1,2, figsize=(22,22))
    _plot_hexagonal_distance_map(som, X, y, label_dict, figtitle, item2plot, ax=axis[0])
    _plot_hexagonal_heatmap(som,X, figtitle, ax=axis[1])
    path2fig = os.path.join(savefig_path, figname_all)
    plt.savefig (path2fig,format='png', dpi=300, bbox_inches='tight' )

    fignames = [figname_umap, figname_heatmap]
    for  ax, figname in zip(axis, fignames):
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        path2fig = os.path.join(savefig_path, figname)
        fig.savefig(path2fig,format='png', dpi=300, bbox_inches=extent.expanded(1.275, 1.4))

    plt.show()

def plot_confusion_matrix (y_true, y_pred, labels, savefig_path, figname) -> None:
    """
        Plots the confusion matrix given the true labels and the predicted labels.
    """
    temp_matrix = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(temp_matrix, display_labels=labels).plot()
    plt.xticks(rotation=90, ha="right")

    path2fig = os.path.join(savefig_path, figname)

    plt.savefig(path2fig, format='png', dpi=300, bbox_inches='tight')

def plot_some_wrongly_predicted_images (wrong_X,
                                        wrong_y,
                                        wrong_pred_y,
                                        save_fig_path,
                                        figname,
                                        images2plot = 25,
                                        dataset="pathmnist"):
    """
        wrong_X: Input array of wrongly predicted images. This can be the scaled array
        wrong_y: The TRUE label of those images.
        wrong_pred_y: The predicted label of those images.
        save_fig_path: Directory to save the figure.
        figname: Name for the figure.
        images2plot: Number of images to be plotted. It is set to 25 as default. Be CAREFUL when changing it, the subplots number should be changed.
        dataset: Name of the dataset, default is "pathmnist".
    """

    label_mapper = Number2LabelMapper(dataset)
    some_images = [ image.reshape(28, 28, 3) for image in wrong_X[:images2plot] ]
    some_labels = label_mapper.map_label(wrong_y[:25])
    some_false_labels = label_mapper.map_label(wrong_pred_y[:25])

    fig,axes = plt.subplots(5,5, figsize=(25,10))
    # Flatten axes for easy iteration
    axes = axes.flatten()

    for ax, img, pred_label, wrong_label in zip(axes, some_images, some_labels, some_false_labels):
        # Display image
        ax.imshow(img)
        ax.set_title(f"True label:{pred_label}\nPred label: {wrong_label}")

        # Remove axis ticks
        ax.axis("off")

    plt.tight_layout()
    path2fig = os.path.join(save_fig_path, figname)
    plt.savefig (path2fig,format='png', dpi=300, bbox_inches='tight')


def plot_some_images_with_label (true_X,
                                 true_y,
                                 save_fig_path,
                                 figname,
                                 images2plot=25,
                                 dataset="pathmnist"):
    """
        true_X: Input array of correctly predicted images. This can be the scaled array
        true_y: The TRUE label of those images.
        save_fig_path: Directory to save the figure.
        figname: Name for the figure.
        images2plot: Number of images to be plotted. It is set to 25 as default. Be CAREFUL when changing it, the subplots number should be changed.
        dataset: Name of the dataset, default is "pathmnist".
    """

    label_mapper = Number2LabelMapper(dataset)
    some_images = [image.reshape(28, 28, 3) for image in true_X[:images2plot]]
    some_labels = label_mapper.map_label(true_y[:25])

    fig, axes = plt.subplots(5, 5, figsize=(25, 10))
    # Flatten axes for easy iteration
    axes = axes.flatten()

    for ax, img, pred_label in zip(axes, some_images, some_labels):
        # Display image
        ax.imshow(img)
        ax.set_title(f"{pred_label}")

        # Remove axis ticks
        ax.axis("off")

    plt.tight_layout()
    path2fig = os.path.join(save_fig_path, figname)
    plt.savefig(path2fig, format='png', dpi=300, bbox_inches='tight')












