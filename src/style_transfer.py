import numpy as np
from skimage.morphology import disk, diamond, erosion, dilation

import torch
import torch.nn as nn
from torch import optim
from torchvision.models import vgg19
import torchvision.transforms as transforms

mu_0, sigma_0, min_0, max_0 = 4.84, 2 * 0.94, 3, 10
mu_1, sigma_1, min_1, max_1 = 3.46, 2 * 0.58, 2, 7

p_dead = 0.2
delta = 2  # compensate for cell perimiter lost during segmentation

def draw_content(style_img, nb_cells, selem='disk', mode='no-overlap'):

    canvas = np.zeros_like(style_img)
    instance_mask = np.zeros_like(canvas)
    bboxes = []

    clearance_map = np.zeros_like(canvas)

    for i in range(nb_cells):

        mask_img = np.zeros_like(canvas)

        # Choose cell radius from bimodal distribution
        if np.random.rand() > p_dead:
            r = int(np.clip(delta + mu_0 + sigma_0 * np.random.randn(), min_0, max_0))
        else:
            r = int(np.clip(delta + mu_1 + sigma_1 * np.random.randn(), min_1, max_1))

        ys, xs = np.nonzero(1 - clearance_map)

        coords = list(zip(ys, xs))
        idx = np.random.randint(len(coords))
        y, x = coords[idx]

        mask = disk(r) if selem == 'disk' else diamond(r)

        # determine bounding box
        top = np.maximum(y - r, 0)  # N.B. top < bottom
        bottom = np.minimum(y + r + 1, canvas.shape[0])
        left = np.maximum(x - r, 0)
        right = np.minimum(x + r + 1, canvas.shape[1])

        # adjust at borders
        adjust_top = top - (y - r)
        adjust_bottom = 2 * r - ((y + r + 1) - bottom) + 1
        adjust_left = left - (x - r)
        adjust_right = 2 * r - ((x + r + 1) - right) + 1

        mask_img[top:bottom, left:right] = mask[adjust_top:adjust_bottom,
                                                adjust_left:adjust_right]

        instance_mask[mask_img > 0] = (i + 1) * mask_img[mask_img > 0]

        # mark clearance
        dilated_mask_img = dilation(mask_img, disk(3))
        clearance_map[dilated_mask_img > 0] = dilated_mask_img[dilated_mask_img > 0]

        # insert mask according to mode
        if mode == 'no-overlap':

            canvas[mask_img > 0] = mask_img[mask_img > 0]
            eroded_mask = erosion(mask_img, disk(1))
            canvas -= 0.5 * eroded_mask

        elif mode == 'overlap':

            eroded_mask = erosion(mask_img, disk(1))
            mask_img -= eroded_mask
            canvas[mask_img > 0] = mask_img[mask_img > 0]

        elif mode == 'mask':

            canvas[mask_img > 0] = mask_img[mask_img > 0]

        bboxes.append([left, top, right, bottom])

    return canvas.astype('float32'), instance_mask.astype('uint8'), bboxes, clearance_map


deep_layer_dict = {'conv1_1' : 2,
                   'conv2_1' : 7,
                   'conv3_1' : 12,
                   'conv4_1' : 21,
                   'conv5_1' : 30}

layer_dict = {'conv1_1' : 2,
              'conv2_1' : 4,
              'conv3_1' : 7,
              'conv4_1' : 9,
              'conv5_1' : 12}


class VGGWrapper(nn.Module):

    def __init__(self, layers):

        super(VGGWrapper, self).__init__()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layer_dict = layers

        self.vgg = vgg19(pretrained=True).features.to(device).eval()
        # convenient to normalise within neural net
        self.means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def forward(self, x):

        outputs = []

        x = (x - self.means) / self.stds
        prev_layer_idx = 0

        for layer in self.layer_dict:

            layer_idx = self.layer_dict[layer]
            x = self.vgg[prev_layer_idx:layer_idx](x)
            outputs.append(x)
            prev_layer_idx = layer_idx

        return outputs


def gram_matrix(x):

    _, c, h, w = x.size()
    x = x.view(c, h * w)
    G = torch.mm(x, x.t())

    return G / (c * h * w)  # normalised by number of neurons in layer


def style_transfer(content_img,
                   style_img,
                   max_iters=300,
                   style_weight=1e7,
                   content_weight=1,
                   layers='shallow',
                   verbose=True):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)

    scaled_size = 128

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(scaled_size),
        transforms.ToTensor(),
    ])

    content_img = preprocess(content_img)[None, ...]
    style_img = preprocess(style_img)[None, ...]

    if layers == 'shallow':
        vgg_wrapper = VGGWrapper(layer_dict)
    else:
        vgg_wrapper = VGGWrapper(deep_layer_dict) 

    content_layer = 4

    content_targets = vgg_wrapper(content_img)[content_layer].detach()
    style_targets = [gram_matrix(layer).detach() for layer in vgg_wrapper(style_img)]

    x = content_img.clone()
    optimizer = optim.LBFGS([x.requires_grad_()])

    iters = [0]

    while iters[0] <= max_iters:

        def closure():

            x.data.clamp_(0, 1)

            optimizer.zero_grad()

            # forward pass
            layers = vgg_wrapper(x)
            content_scores = layers[content_layer]
            style_scores = [gram_matrix(layer) for layer in layers]

            # calculate losses
            style_loss = style_weight * sum([nn.MSELoss()(s, t) for s, t in 
                              list(zip(style_scores, style_targets))])

            content_loss = content_weight * nn.MSELoss()(content_scores, 
                                                         content_targets)

            iters[0] += 1

            if iters[0] % 50 == 0 and verbose:
                print('[%d]\tContent loss: %.04f\tStyle loss: %.04f' % (
                    iters[0], content_loss.item(), style_loss.item()))

            loss = style_loss + content_loss

            # backpropagate
            loss.backward()

            return loss

        optimizer.step(closure)  # note LBFGS calls the closure several times

    return x.data.clamp_(0, 1)
