from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.morphology import dilation, disk

from skimage.transform import resize
from skimage.morphology import disk

from ipywidgets import interact, IntSlider


def plot_blobs(ax, img, blobs, colour='red'):

    ax.imshow(img)

    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=colour, linewidth=2, fill=False)
        ax.add_patch(c)


def visualise_rgb(mcherry, gfp, phase_contrast):

    r = np.clip(mcherry + phase_contrast, 0, 1)
    g = np.clip(gfp + phase_contrast, 0, 1)
    b = phase_contrast

    return np.dstack([r, g, b])


def plot_confusion_matrix(
    ax, matrix, labels, title='Confusion matrix', fontsize=9):

    ax.set_xticks([x for x in range(len(labels))])
    ax.set_xticklabels([])
    ax.set_yticks([y for y in range(len(labels))])
    ax.set_yticklabels([])

    # Place labels on minor ticks
    ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
    ax.set_xticklabels(labels, rotation='90', fontsize=fontsize, minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(labels))], minor=True)
    ax.set_yticklabels(labels[::-1], fontsize=fontsize, minor=True)

    # Hide major tick labels
    ax.tick_params(which='major', labelbottom='off', labelleft='off')

    # Finally, hide minor tick marks
    ax.tick_params(which='minor', width=0)

    # Plot heat map
    proportions = [1. * row / sum(row) for row in matrix]
    ax.pcolor(np.array(proportions[::-1]), cmap='bwr')

    # Plot counts as text
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            confusion = matrix[::-1][row][col]
            if confusion != 0:
                ax.text(col + 0.5, row + 0.5, confusion, fontsize=fontsize,
                    horizontalalignment='center',
                    verticalalignment='center')

    # Add finishing touches
    ax.grid(True, linestyle=':')
#     ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('prediction', fontsize=fontsize)
    ax.set_ylabel('actual', fontsize=fontsize)


def visualise_random_crops(crops, crops_3d, class_labels, bbs, pad, num_crops=10):

    fig = plt.figure(figsize=(10, 4))

    N = crops.shape[0]

    for i in range(num_crops):
        neg_idx = np.argwhere(class_labels == 1)[:, 0]
        idx = np.random.choice(neg_idx)

        ax = fig.add_subplot(4, num_crops, i + 1)
        h, w = bbs[idx]

        rect = plt.Rectangle((pad - pad * w, pad - pad * h),
                             2 * pad * w, 2 * pad * h, linewidth=1,
                             edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.imshow(crops_3d[idx])
        ax.axis('off')

        ax = fig.add_subplot(4, num_crops, num_crops + i + 1)
        ax.imshow(crops[idx], cmap='Greys_r')
        ax.axis('off')

        pos_idx = np.argwhere(class_labels == 2)[:, 0]
        idx = np.random.choice(pos_idx)

        ax = fig.add_subplot(4, num_crops, 2 * num_crops + i + 1)
        h, w = bbs[idx]
        
        rect = plt.Rectangle((pad - pad * w, pad - pad * h),
                             2 * pad * w, 2 * pad * h, linewidth=1,
                             edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.imshow(crops_3d[idx])
        ax.axis('off')

        ax = fig.add_subplot(4, num_crops, 3 * num_crops + i + 1)
        ax.imshow(crops[idx], cmap='Greys_r')
        ax.axis('off')

    plt.show()


def create_mosaic_plot(crops,
                       class_labels,
                       class_names,
                       nb_channels=1,
                       nb_rows=10,
                       nb_cols=10):

    def create_mosaic(crops):

        dim = crops[0].shape[0]
        mosaic = np.zeros((nb_rows * dim, nb_cols * dim, nb_channels))

        for i in range(nb_rows):

            for j in range(nb_cols):

                top = dim * i
                bottom = dim * (i + 1)
                left = dim * j
                right = dim * (j + 1)

                mosaic[top:bottom, left:right] = crops[nb_cols * i + j]

        return mosaic

    fig = plt.figure(figsize=(15, 6))

    for i, class_name in enumerate(class_names):
        class_crops = crops[class_labels == i]

        mosaic = create_mosaic(class_crops)

        ax = fig.add_subplot(1, 3, i + 1)

        if nb_channels == 1:
            ax.imshow(mosaic[:, :, 0], cmap='Greys_r')
        else:
            ax.imshow(mosaic, cmap='Greys_r')
        
        ax.set_title('%s samples' % class_name)
        ax.axis('off')

    plt.show()


def plot_embeddings(ax, embedding, labels):

    ax.scatter(embedding[(labels==0), 0],
               embedding[(labels==0), 1],
               color='red', alpha=0.5, edgecolor='black')
    ax.scatter(embedding[(labels==1), 0],
               embedding[(labels==1), 1],
               color='blue', alpha=0.5, edgecolor='black')
    ax.scatter(embedding[(labels==2), 0],
               embedding[(labels==2), 1],
               color='green', alpha=0.5, edgecolor='black')
    ax.scatter(embedding[(labels==3), 0],
               embedding[(labels==3), 1],
               color='yellow', alpha=0.5, edgecolor='black')
