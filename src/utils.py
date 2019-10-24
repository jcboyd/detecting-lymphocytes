import os
import re
import numpy as np

from skimage.transform import rotate, rescale
from skimage.io import imread
from skimage.measure import regionprops
from skimage.morphology import dilation, disk
from skimage.feature import blob_log

from scipy.ndimage import convolve
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from keras.utils import to_categorical
import torch

from bs4 import BeautifulSoup


experiments = {
    'raji_target' : {
        'train_wells' : ['VID193_A1', 'VID193_A2', 'VID193_A5', 'VID193_A6_1',
                         'VID193_A6_2'],
        'val_wells' : ['VID193_A6_3'],
        'test_wells' : ['VID193_A6_4']
    },

    'CAR_June' : {
        'train_wells' : ['VID193_B1', 'VID193_B2', 'VID193_B5', 'VID193_B6_1',
                         'VID193_B6_2'],
        'val_wells' : ['VID193_B6_3'],
        'test_wells' : ['VID193_B6_4']
    }
}


def import_annotations(annotations_dir, plate):
    
    file_names = sorted(list(filter(lambda x : not x.startswith('.'), 
                    os.listdir(os.path.join(annotations_dir, plate)))))

    arr_num_dead, arr_num_raji = [], []
    gt_b_masks, gt_d_masks = [], []

    for file_name in file_names:

        file_path = os.path.join(annotations_dir, plate, file_name)

        b_mask = np.zeros((256, 256))
        d_mask = np.zeros((256, 256))

        with open(file_path) as fp:
            soup = BeautifulSoup(fp)

        num_dead, num_raji = 0, 0

        for obj in soup.findAll('object'):

            num_dead += obj.find('name').text == 'Dead'
            num_raji += obj.find('name').text == 'RAJI'

            xmin = int(obj.bndbox.xmin.string)
            xmax = int(obj.bndbox.xmax.string)
            ymin = int(obj.bndbox.ymin.string)
            ymax = int(obj.bndbox.ymax.string)

            center_x = int((xmax + xmin) / float(2))
            center_y = int((ymax + ymin) / float(2))

            if obj.find('name').text == 'RAJI':
                b_mask[center_y, center_x] = 1
            else:
                d_mask[center_y, center_x] = 1

            gt_b_masks.append(b_mask)
            gt_d_masks.append(d_mask)

        arr_num_dead.append(num_dead)
        arr_num_raji.append(num_raji)

        # print(('No. dead: %d, No. RAJI: %d') % (num_dead, num_raji))
    
    return arr_num_dead, arr_num_raji, gt_b_masks, gt_d_masks



def bb_intersection_over_union(boxA, boxB):

    """
    Taken from:

    https://github.com/ForeverStrongCheng/intersection_over_union-python/blob/master/intersection_over_union.py
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou


def convert_to_bounding_box(y, x, h, w):

    y_min = np.clip(4 * y - 12 * h, 0, 256)
    y_max = np.clip(4 * y + 12 * h, 0, 256)

    x_min = np.clip(4 * x - 12 * w, 0, 256)
    x_max = np.clip(4 * x + 12 * w, 0, 256)
    
    return y_min, x_min, y_max, x_max


def nms(probs, bbs, prob_threshold, iou_threshold):

    list_coords = list(map(list, np.nonzero(probs[0] > prob_threshold)))
    list_coords = list(zip(*list_coords))

    coords, bounding_boxes = [], []
    
    for y, x, _ in list_coords:
        eps = 5e-2
        h, w = bbs[0, y, x]

        if h < eps or w < eps:
            probs[0, y, x, :] = 0
            continue

        coords.append([y, x])
        bounding_boxes.append((convert_to_bounding_box(y, x, h, w)))

    coords = np.array(coords)
    bounding_boxes = np.array(bounding_boxes)

    detections = []

    while np.any(probs > prob_threshold):

        # find bounding box of highest remaining probability
        _, y, x, cls = np.unravel_index(probs.argmax(), probs.shape)

        h, w = bbs[0, y, x, :]

        bb = convert_to_bounding_box(y, x, h, w)

        # compute iou over remaining bounding boxes
        ious = np.array(list(map(lambda x : bb_intersection_over_union(bb, x),
            bounding_boxes)))

        bb += (cls,)

        if h > 0.2 and w > 0.2:  # discard small bounding boxes
            detections.append(bb)

        # for all bounding boxes concerned, zero-out probs
        for i, iou in enumerate(ious):
            if iou >= iou_threshold:
                y, x = coords[i]
                probs[0, y, x, :] = 0

        # ...and remove from bounding_boxes list
        bounding_boxes = bounding_boxes[ious < iou_threshold]
        coords = coords[ious < iou_threshold]

    return detections


def smooth_probabilties(probs, alpha=0.33):
    weights = np.array([(1-alpha)/2, alpha, (1-alpha)/2])
    return np.sum(probs * weights[:, np.newaxis, np.newaxis], axis=0)


def get_files(input_dir, experiment, days=10):

    train_wells = experiments[experiment]['train_wells']
    val_wells = experiments[experiment]['val_wells']
    test_wells = experiments[experiment]['test_wells']

    # collect all files
    img_names = os.listdir(os.path.join(input_dir, 'phase_contrast'))

    # filter hidden files
    img_names = list(filter(lambda x : not x.startswith('.'), img_names))

    # take first N days
    f_filter = lambda x : int(re.search('(.2?)d', x).group(1)) < days
    img_names = list(filter(f_filter, img_names))

    train_imgs = sorted(list(filter(lambda x : any([x.startswith(well)
                      for well in train_wells]), img_names)))
    val_imgs = sorted(list(filter(lambda x : any([x.startswith(well)
                    for well in val_wells]), img_names)))
    test_imgs = sorted(list(filter(lambda x : any([x.startswith(well)
                     for well in test_wells]), img_names)))
    # terminal_frames = []

    print('No. train images: %d' % len(train_imgs))
    print('No. val images: %d' % len(val_imgs))
    print('No. test images: %d' % len(test_imgs))

    return train_imgs, val_imgs, test_imgs


def get_data(input_dir, file_names, mode, pretrained, normalise):

    # hard-coded means used in pretrained PyTorch models
    torch_mean = [0.485, 0.456, 0.406]
    torch_std = [0.229, 0.224, 0.225]

    pc_dir = os.path.join(input_dir, 'phase_contrast')
    gfp_dir = os.path.join(input_dir, 'GFP_calibrated')
    mcherry_dir = os.path.join(input_dir, 'mCherry_calibrated')

    pc = np.stack([imread(os.path.join(pc_dir, img_name))[:960, :1280]
                   for img_name in sorted(file_names)])

    gfp = np.stack([imread(os.path.join(gfp_dir, img_name))[:960, :1280]
                    for img_name in sorted(file_names)])

    mcherry = np.stack([imread(os.path.join(mcherry_dir, img_name))[:960, :1280]
                        for img_name in sorted(file_names)])

    if normalise:

        pc = pc / float(2 ** 8)  # phase contrast already limited to 8-bit integer

        if pretrained:
            pc = np.stack([pc, pc, pc], axis=-1)
            pc = (pc - torch_mean) / torch_std

        else:
            pc = pc[..., None]
            pc = (pc - torch_mean[0]) / torch_std[0]

        gfp = np.clip(gfp, 0, 2 ** 10)
        gfp = gfp / float(2 ** 10)

        mcherry = np.clip(mcherry, 0, 2 ** 7)
        mcherry = mcherry / float(2 ** 7)

    if mode == 'gfp':
        return pc, gfp[..., None]

    elif mode == 'mcherry':
        return pc, mcherry[..., None]

    elif mode == 'stacked':
        return pc, np.stack([gfp, mcherry], axis=-1)

    else:
        raise(Exception('Mode \'%s\' not valid' % mode))


def detect_objects(probabilities, threshold=0.9, bb_size=10, iou=0.3):

    b_probs = probabilities[0, :, :, 1]
    d_probs = probabilities[0, :, :, 2]

    b_centers, d_centers = nms(b_probs, d_probs, threshold, bb_size, iou)

    # [2 * np.array(nms(p, threshold, bb_size, iou)) 
    #                         for p in [b_probs, d_probs]]

    return b_centers, d_centers


def pad_input(phase_contrast, pad_width):
    padded_pc = np.pad(phase_contrast, pad_width, mode='reflect')
    return padded_pc[np.newaxis, :, :, np.newaxis]


def create_masks(height, width, b_centers, d_centers):

    raji_mask = np.zeros((height, width))
    if len(b_centers) > 0:
        raji_mask[b_centers[:, 0], b_centers[:, 1]] = 1

    dead_mask = np.zeros((height, width))
    if len(d_centers) > 0:
        dead_mask[d_centers[:, 0], d_centers[:, 1]] = 1

    return raji_mask, dead_mask


def mean_precision(confusion_matrix):
    tp = np.diag(confusion_matrix)
    tp_and_fp = np.sum(confusion_matrix, axis=0)  # column sum
    precision = tp.astype(np.float) / tp_and_fp
    return np.mean(precision)


def mean_recall(confusion_matrix):
    tp = np.diag(confusion_matrix)
    tp_and_fn = np.sum(confusion_matrix, axis=1)  # row sum
    recall = tp.astype(np.float) / tp_and_fn
    return np.mean(recall)


def data_augmentation(x_batch):

    """Randomly modifies data batch.

        Parameters
        -----------
        x_batch : 2-D ndarray

        Returns
        --------
        x_batch : 2-D ndarray
    """

    batch_size = x_batch.shape[0]

    # random left-right reflection
    flip_idx = np.random.binomial(n=1, p=0.5, size=batch_size)
    x_batch = np.array([np.fliplr(x_batch[i]) if flip_idx[i] else x_batch[i]
                        for i in range(batch_size)])

    # random up-down reflection
    flip_idx = np.random.binomial(n=1, p=0.5, size=batch_size)
    x_batch = np.array([np.flipud(x_batch[i]) if flip_idx[i] else x_batch[i]
                        for i in range(batch_size)])

    return x_batch


def gen_balanced(x_train, y_train, batch_size=64, augment=False):

    """Generator of balanced SGD batches from potentially unbalanced data.

        Parameters
        -----------
        x_train : 2-D ndarray
        y_train : 1-D ndarray
        batch_size : int, optional

        Yields
        -------
        (x_batch, y_batch) : tuple of 2-D ndarray and 1-D ndarray
    """

    assert(x_train.shape[0] == y_train.shape[0])

    labels = np.argmax(y_train, axis=1)
    nb_classes = np.max(labels) + 1
    class_labels = [np.where(labels == i)[0] for i in range(nb_classes)]

    while True:

        size = batch_size // nb_classes
        batch_idx = np.ravel([np.random.choice(class_labels[class_idx], size) 
                              for class_idx in range(nb_classes)])

        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]

        if augment:

            x_batch = data_augmentation(x_batch)

        yield x_batch, y_batch[:, np.newaxis, np.newaxis, :]


def gen_balanced_multi(x_train, y_train, batch_size=64, augment=False):

    """Generator of balanced SGD batches from potentially unbalanced data.

        Parameters
        -----------
        x_train : 2-D ndarray
        y_train : 1-D ndarray
        batch_size : int, optional

        Yields
        -------
        (x_batch, y_batch) : tuple of 2-D ndarray and 1-D ndarray
    """

    assert(x_train.shape[0] == y_train.shape[0])

    labels = np.argmax(y_train[:, :3], axis=1)
    nb_classes = np.max(labels) + 1
    class_labels = [np.where(labels == i)[0] for i in range(nb_classes)]

    while True:

        sizes = [0.5 * batch_size, 0.25 * batch_size, 0.25 * batch_size]
        batch_idx = np.hstack([np.random.choice(class_labels[class_idx], int(sizes[class_idx]))
                              for class_idx in range(nb_classes)])

        x_batch = x_train[batch_idx]
        y_batch_obj = to_categorical(y_train[batch_idx, 0] == 0)
        y_batch_cls = y_train[batch_idx][:, 1:3]
        y_batch_bbs = y_train[batch_idx][:, 3:]

        w_batch = y_batch_obj[:, 1][:, np.newaxis]
        w_batch = np.hstack([w_batch, w_batch])
        w_batch = w_batch[:, np.newaxis, np.newaxis, :]

        if augment:

            x_batch = data_augmentation(x_batch)

        yield [x_batch, w_batch], [y_batch_obj[:, np.newaxis, np.newaxis, :],
                                   y_batch_cls[:, np.newaxis, np.newaxis, :],
                                   y_batch_bbs[:, np.newaxis, np.newaxis, :]]


def downsample_balanced(x_train, y_train):

    """Downsamples categorical data to the size of the least represented class.

        Parameters
        -----------
        x_train : 2-D array
        y_train : 1-D array

        Returns
        --------
        x_train_balanced : 2-D array
        y_train_balanced : 1-D array
    """

    nb_classes = len(np.unique(y_train))

    class_min = np.min(np.bincount(y_train))

    class_labels = [np.where(y_train == i)[0] for i in range(nb_classes)]

    batch_idx = np.ravel([np.random.choice(class_labels[class_idx], class_min) 
                          for class_idx in range(nb_classes)])

    x_train_balanced = x_train[batch_idx]
    y_train_balanced = y_train[batch_idx]

    return x_train_balanced, y_train_balanced


def calculate_precision(tps, fps):
    return tps / float(tps + fps)


def calculate_recall(tps, fns):
    return tps / float(tps + fns)


def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def evaluate(mask0, mask1, max_dist=7):

    labelled0 = label(mask0)
    labelled1 = label(mask1)

    regions0 = regionprops(labelled0)
    regions1 = regionprops(labelled1)

    tps, fps, fns = (0, 0, 0)

    for props in regions0:
        center_y, center_x = props.centroid

        for dist in range(max_dist):

            left = np.clip(center_x - dist, 0, mask0.shape[1]).astype('int')
            right = np.clip(center_x + dist + 1, 0, mask0.shape[1]).astype('int')
            top = np.clip(center_y - dist, 0, mask0.shape[0]).astype('int')
            bottom = np.clip(center_y + dist + 1, 0, mask0.shape[0]).astype('int')

            # find objects in produced mask within bounding box
            objects = list(filter(lambda x: x != 0,
                np.unique(labelled1[top:bottom, left:right])))

            if len(objects) > 0:
                # get props for first object in bounding box
                object_id = objects[0]
                props1 = list(filter(
                    lambda props: props.label == object_id, regions1))[0]
                # erase object in ground truth
                coords0 = props.coords
                labelled0[(coords0[:, 0], coords0[:, 1])] = 0
                # erase object in mask
                coords1 = props1.coords
                labelled1[(coords1[:, 0], coords1[:, 1])] = 0
                tps += 1
                break
    # false negatives - missed examples in ground truth
    fns = len(regionprops(label(labelled0 > 0)))
    # false positives - missed examples in prediction
    fps = len(regionprops(label(labelled1 > 0)))
    return tps, fps, fns
