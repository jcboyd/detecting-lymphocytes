from __future__ import print_function
from __future__ import division

import os
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.filters import gaussian, threshold_otsu
from skimage.filters.rank import mean
from skimage.morphology import opening, disk, reconstruction
from skimage.measure import label, regionprops
from scipy.ndimage import convolve


def normalise(img, p=99):

    """Applies min-max normalisation where max is set to a given percentile.

    Parameters
    -----------
        img : numpy.ndarray
        p : int

    Returns
    --------
        numpy.ndarray
    """

    # extract statistics
    percentile = np.percentile(img, p)

    minimum = np.min(img)
    # scale intensities
    img = (img - minimum) / float(percentile - minimum)
    # clip values
    img[img < 0] = 0
    img[img > 1] = 1

    return img


def separate_channels(img):

    """Separates the three channels of an RGB image 

    Parameters
    -----------
        img : numpy.ndarray of size H x W x 3

    Returns
    --------
        Tuple of numpy.ndarray
    """

    mcherry = img[:, :, 0]
    gfp = img[:, :, 1]
    phase_contrast = img[:, :, 2]

    return mcherry, gfp, phase_contrast


def prefilter(channel, sigma=1):

    """Applies a Gaussian filter to each input.

    Parameters
    -----------
        phase_contrast : numpy.ndarray
        red_recon : numpy.ndarray
        green_recon : numpy.ndarray
        sigma : float, variance of the gaussian

    Returns
    --------
        Tuple of numpy.ndarray
    """

    return gaussian(channel, sigma)


def background_subtraction(phase_contrast, diameter=19):

    """Local mean subtraction with clipping.

    Parameters
    -----------
        phase_contrast : numpy.ndarray

    Returns
    --------
        Tuple of numpy.ndarray
    """

    # compute local average image

    selem = disk(diameter)
    background = mean((phase_contrast * 255).astype('uint8'), selem) / 255.

    # build difference and clip
    segmentation = np.maximum(background - phase_contrast, 0)

    return background, segmentation


def fill_holes(segmentation, scale_factor=1):

    """Fills holes in image by erosion of an encompassing seed.

    Parameters
    -----------
        segmentation : numpy.ndarray
        scale_factor : float, scales Otsu threshold after reconstruction

    Returns
    --------
        Tuple of numpy.ndarray
    """

    seed = segmentation.max() * np.ones_like(segmentation)

    # pad with zeros
    fill = reconstruction(np.pad(seed, mode='constant', pad_width=1),
                          np.pad(segmentation, mode='constant', pad_width=1),
                          'erosion', selem=disk(1))

    # slice away zeros after reconstruction
    fill = fill[1:-1, 1:-1]

    # Perform Otsu thresholding
    t = threshold_otsu(fill)

    mask = fill >= scale_factor * t
#     opened = opening(mask, disk(1))
    opened = mask

    return fill, mask, opened


def label_connected_components(opened):

    """Labels connected components.

    Parameters
    -----------
        opened : numpy.ndarray

    Returns
    --------
        numpy.ndarray
    """

    labels = label(opened > 0)
    return labels


def extract_features(mcherry, gfp, phase_contrast, labels):

    """Extracts set of features from each connected component.

    Parameters
    -----------
        phase_contrast : numpy.ndarray
        red_recon : numpy.ndarray
        green_recon : numpy.ndarray
        labels : numpy.ndarray

    Returns
    --------
        Tuple of numpy.ndarray
    """

    regions = regionprops(labels, coordinates='rc')
    coords = np.array([list(props.centroid) for props in regions])

    # take averages over connected components
    agg_mean = lambda channel, props : np.mean(channel[props.coords[:, 0],
                                                       props.coords[:, 1]])

    features = np.array([[
        agg_mean(mcherry, props),
        agg_mean(gfp, props),
        agg_mean(phase_contrast, props),
        props.eccentricity,
        props.area,
        props.bbox[0],
        props.bbox[1],
        props.bbox[2],
        props.bbox[3]] for props in regions])

    columns = ['mean_red', 'mean_green', 'mean_pc', 'eccentricity', 'area',
               'xmin', 'ymin', 'xmax', 'ymax']

    df_features = pd.DataFrame(data=features, columns=columns)
    df_features['y'] = coords[:, 0]
    df_features['x'] = coords[:, 1]

    pad = 12

    h, w = labels.shape[0:2]

    in_y = (pad < df_features['y']) & (df_features['y'] < (h - pad))
    in_x = (pad < df_features['x']) & (df_features['x'] < (w - pad))

    df_features['interior'] = in_y & in_x

    return regions, df_features


def assign_classes(df_features, mode, gfp_delta):

    """Assigns classes to connected components by thresholding features

    Parameters
    -----------
        phase_contrast: numpy.ndarray
        mode: string, 'two-class' or 'three-class'
        delta: float, margin of error discarded around threshold

    Returns
    --------
        List of integers
    """

    t_reds = 0.2 # threshold_otsu(df_features.mean_red)
    t_greens = threshold_otsu(df_features.mean_green)

    # increasing delta should increase confidence in class assignment

    # lo_red = df_features.mean_red < t_reds
    hi_red = df_features.mean_red > t_reds

    lo_green = df_features.mean_green < (t_greens - gfp_delta)
    hi_green = df_features.mean_green > (t_greens + gfp_delta)# + gfp_delta)

    small_dead = np.percentile(df_features.area, q=35)
    small_raji = np.percentile(df_features.area, q=55)
    perc95 = np.percentile(df_features.area, q=100)

    # eliminate large OBJECTS BY BOUNDING BOX!

    not_small_dead = df_features.area > small_dead
    not_small_raji = df_features.area > small_raji

    not_large_object = df_features.area < perc95

    size_range_dead = not_small_dead & not_large_object
    size_range_raji = not_small_raji & not_large_object
    # small_object = df_features.area < np.mean(df_features.area)

    # 2-class assignment
    if mode == 'two-class':

        df_features['class'] = 3  # default--dummy class

        b_cell_idx = lo_green & size_range_raji & df_features.interior  # B cell indices
        d_cell_idx = hi_green & size_range_dead & df_features.interior + hi_red  # dead B cell indices

        df_features.loc[b_cell_idx, 'class'] = 1
        df_features.loc[d_cell_idx, 'class'] = 2

    # 3-class assignment
    elif mode == 'three-class':

        df_features['class'] = 4

        b_cell_idx = hi_red & lo_green & size_range  # B cell indices
        d_cell_idx = lo_red & hi_green & size_range  # dead B cell indices
        t_cell_idx = lo_red & lo_green & size_range  # T cell indices

        df_features.loc[b_cell_idx, 'class'] = 1
        df_features.loc[d_cell_idx, 'class'] = 2
        df_features.loc[t_cell_idx, 'class'] = 3

    return df_features


def class_instance_masks(df_features, labels):

    unique_labels = np.unique(labels)

    class_masks = []
    instance_masks = []

    for class_number in [1, 2]:

        instance_labels = np.where(df_features['class'] == class_number)[0]
        class_instances = unique_labels[instance_labels]
        class_mask = np.isin(labels, class_instances)

        class_mask = opening(class_mask, disk(2))

        class_masks.append(class_mask)
        instance_masks.append(label(class_mask))

    return class_masks, instance_masks


def run_pipeline(img, mode, gfp_delta):

    mcherry, gfp, phase_contrast = separate_channels(img)

    yield 'channels', (mcherry, gfp, phase_contrast)

    mcherry, gfp, phase_contrast = map(prefilter, (mcherry, gfp, phase_contrast))

    yield 'prefilter', (mcherry, gfp, phase_contrast)

    background, segmentation = background_subtraction(phase_contrast)

    yield 'background', (background, segmentation)

    fill, mask, opened = fill_holes(segmentation)

    yield 'fill holes', (fill, mask, opened)

    labels = label_connected_components(opened)

    yield 'labels', labels

    regions, df_features = extract_features(mcherry, gfp, phase_contrast, labels)

    yield 'features', (regions, df_features)

    classes = assign_classes(df_features, mode, gfp_delta)

    yield 'classes', classes

    class_masks, instance_masks = class_instance_masks(df_features, labels)

    yield 'class_masks', (class_masks, instance_masks)
