import numpy as np
from skimage.morphology import disk, diamond, erosion

import torch
import torch.nn as nn
from torch import optim
from torchvision.models import vgg19
import torchvision.transforms as transforms


def draw_content(style_img, nb_cells, selem='disk', min_r=4, max_r=8):

    mean_intensity = np.mean(style_img / 255.)
    canvas = mean_intensity * np.ones_like(style_img)
    contour_intensity = 0.8

    for _ in range(nb_cells):

        mask_img = np.zeros_like(canvas)

        r = np.random.randint(min_r, max_r)
        y = np.random.randint(r, mask_img.shape[0] - r)
        x = np.random.randint(r, mask_img.shape[1] - r)

        if selem == 'disk':
            mask = contour_intensity * disk(r) 
        else:
            mask = contour_intensity * diamond(r) 

        mask_img[y-r:y+r+1, x-r:x+r+1] = mask

        canvas[mask_img > 0] = mask_img[mask_img > 0]    
        eroded_mask = erosion(mask_img, disk(1))
        canvas -= eroded_mask
        canvas += mean_intensity * eroded_mask / contour_intensity

    return canvas.astype('float32')


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
