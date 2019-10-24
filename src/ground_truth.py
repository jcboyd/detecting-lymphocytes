import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from src.vis_utils import visualise_rgb


def extract_background_crops(pl, nb_crops, pad, sep=5):

    background = np.ones_like(pl['channels'][0])
    regions = pl['features'][0]

    for props in regions:
        y, x = list(map(int, props.centroid))
        background[y, x] = 0

    dists = distance_transform_edt(background) #, metric='manhattan')

    # threshold distance map
    ys, xs = np.where((dists >= 1 * sep) * (dists <= 3 * sep)) # 0.5 * pad)
    idx = np.random.randint(xs.shape[0], size=nb_crops)

    # pad image
    padded_pc = np.pad(pl['channels'][2], pad, 'reflect')  # 2d pad

    frame_img = visualise_rgb(
        pl['channels'][0],
        pl['channels'][1],
        pl['channels'][2])

    padding = ((pad, pad), (pad, pad), (0, 0))
    padded_rgb = np.pad(frame_img, padding, 'reflect')  # 3d pad

    # adjust coordinates for padding
    y_centers, x_centers = ys[idx] + pad, xs[idx] + pad
    coords = list(zip(y_centers, x_centers))

    crops = np.stack([padded_pc[y-pad:y+pad, x-pad:x+pad] for y, x in coords])
    crops_rgb = np.stack([padded_rgb[y-pad:y+pad, x-pad:x+pad] for y, x in coords])

    return crops, crops_rgb


def crop_image(img, x, y, pad):
    return img[y-pad:y+pad, x-pad:x+pad]


def export_crops(pls, pad=7, dummy_class=2):

    nb_crops = sum([len(pl['features'][0]) for pl in pls])

    crops = np.empty((nb_crops, pad + pad, pad + pad))
    crops_rgb = np.empty((nb_crops, pad + pad, pad + pad, 3))
    class_labels = np.empty(nb_crops)
    bbs = np.empty((nb_crops, 2))

    all_props = []

    idx = 0

    for pl in pls:  # iterate over pipelines

        frame_pc = pl['channels'][-1]

        frame_img = visualise_rgb(
            pl['channels'][0],
            pl['channels'][1],
            pl['channels'][2])

        padded_pc = np.pad(frame_pc, pad, 'reflect')  # 2d pad
        padding = ((pad, pad), (pad, pad), (0, 0))
        padded_rgb = np.pad(frame_img, padding, 'reflect')  # 3d pad

        frame_regions = pl['features'][0]
        pl_props = []

        # extract object crops
        for i, props in enumerate(frame_regions):

            y, x = list(map(int, props.centroid))

            ymin, xmin, ymax, xmax = props.bbox
            w = xmax - xmin
            h = ymax - ymin
            bbs[idx] = [(h + 2) / (2 * pad), (w + 2) / (2 * pad)]

            crops[idx] = crop_image(padded_pc, x+pad, y+pad, pad)
            crops_rgb[idx] = crop_image(padded_rgb, x+pad, y+pad, pad)

            class_labels[idx] = pl['classes']['class'][i]

            if class_labels[idx] != dummy_class:
                pl_props.append(props.centroid)

            idx += 1

        all_props.append(pl_props)

    return crops, crops_rgb, class_labels, bbs, all_props


def visualise_random_crops(crops, crops_3d, class_labels, nb_crops, nb_classes):

    N = crops.shape[0]
    nb_rows = nb_classes * 2

    fig = plt.figure(figsize=(10, nb_rows))

    for i in range(nb_crops):
        b_cell_idx = np.argwhere(class_labels == 0)[:, 0]
        idx = np.random.choice(b_cell_idx)

        ax = fig.add_subplot(nb_rows, nb_crops, i + 1)
        ax.imshow(crops_3d[idx])
        ax.axis('off')

        ax = fig.add_subplot(nb_rows, nb_crops, nb_crops + i + 1)
        ax.imshow(crops[idx], cmap='Greys_r')
        ax.axis('off')

        d_cell_idx = np.argwhere(class_labels == 1)[:, 0]
        idx = np.random.choice(d_cell_idx)

        ax = fig.add_subplot(nb_rows, nb_crops, 2 * nb_crops + i + 1)
        ax.imshow(crops_3d[idx])
        ax.axis('off')

        ax = fig.add_subplot(nb_rows, nb_crops, 3 * nb_crops + i + 1)
        ax.imshow(crops[idx], cmap='Greys_r')
        ax.axis('off')

        if nb_classes == 3:

            t_cell_idx = np.argwhere(class_labels == 2)[:, 0]
            idx = np.random.choice(t_cell_idx)

            ax = fig.add_subplot(nb_rows, nb_crops, 4 * nb_crops + i + 1)
            ax.imshow(crops_3d[idx])
            ax.axis('off')

            ax = fig.add_subplot(nb_rows, nb_crops, 5 * nb_crops + i + 1)
            ax.imshow(crops[idx], cmap='Greys_r')
            ax.axis('off')

    plt.show()
