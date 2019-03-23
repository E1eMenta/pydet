import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from ..utils.torch import SSDDecodeConf, SSDDecodeBoxes

def s_(k, smin, smax, m):
    '''
    Area of box at k-th feature map from SSD paper
    :param k: Index of feature map [0, m)
    :param smin: Minimum area
    :param smax: Maximum area
    :param m: Number of feature maps
    :return: Area of box at k-th feature map
    '''
    s_k = smin + (smax - smin) / (m - 1) * k
    return s_k


class AnchorCellCreator:
    def __init__(self, aspect_ratios, smin=0.2, smax=0.95):
        self.aspect_ratios = aspect_ratios
        self.smin = smin
        self.smax = smax

        self.anchors_per_cell = [len(ars) + 1 for ars in aspect_ratios]

    def __call__(self, k, H, W, h_i, w_i):
        s_k = s_(k, self.smin, self.smax, len(self.aspect_ratios))
        s_kplus = s_(k, self.smin, self.smax, len(self.aspect_ratios))
        s_prime = np.sqrt(s_k * s_kplus)

        cell_h = 1.0 / H
        cell_w = 1.0 / W
        cell_cx = (w_i + 0.5) * cell_w
        cell_cy = (h_i + 0.5) * cell_h

        anchors = []
        for ar in self.aspect_ratios[k]:
            w = s_k * np.sqrt(ar)
            h = s_k / np.sqrt(ar)
            anchors.append(np.array([cell_cx, cell_cy, w, h]))
        # Extra box
        w = s_prime * np.sqrt(1.0)
        h = s_prime / np.sqrt(1.0)
        anchors.append(np.array([cell_cx, cell_cy, w, h]))

        return np.stack(anchors)



def create_anchors(sizes, anchor_creator, clamp=True):
    anchors = []
    anchors_per_cell = [0] * len(sizes)
    for k, (H, W) in enumerate(sizes):
        cell_h = 1.0 / H
        cell_w = 1.0 / W
        for h_i in range(H):
            for w_i in range(W):
                cell_cx = (w_i + 0.5) * cell_w
                cell_cy = (h_i + 0.5) * cell_h

                boxes = anchor_creator(k, H, W, h_i, w_i)
                anchors_per_cell[k] = len(boxes)
                anchors.append(boxes)

    anchors = np.concatenate(anchors, axis=0).astype(np.float32)
    anchors = torch.from_numpy(anchors)
    if clamp:
        anchors = torch.clamp(anchors, 0.0, 1.0)
    return anchors, anchors_per_cell

class SSDHead(nn.Module):
    def __init__(self, n_classes, in_channels, fmap_sizes, anchor_creator):
        super().__init__()

        self.n_classes      = n_classes
        self.c_             = n_classes + 1 # Background class
        self.in_channels    = in_channels

        self.channels_per_anchor = 4 + self.c_

        anchors, anchors_per_cell = create_anchors(fmap_sizes, anchor_creator)

        self.register_buffer('anchors', anchors)

        self.detection_layers = []
        for in_channel, anchors_num in zip(self.in_channels, anchors_per_cell):
            layer = nn.Conv2d(in_channel,  anchors_num * self.channels_per_anchor, 3, padding=1)
            self.detection_layers.append(layer)

        self.detection_layers = nn.ModuleList(self.detection_layers)



    def forward(self, backbone_outs):
        device = backbone_outs[0].device
        batch_size = backbone_outs[0].shape[0]

        outs = []
        for backbone_out, layer in zip(backbone_outs, self.detection_layers):
            out = layer(backbone_out).permute((0, 2, 3, 1)).contiguous()
            out = out.view((batch_size, -1, self.channels_per_anchor))
            outs.append(out)
        output = torch.cat(outs, dim=1)

        conf = output[:, :, :self.c_]
        loc = output[:, :, self.c_:]

        return conf, loc, self.anchors

    def minimal_forward(self, backbone_outs):
        outs = []
        for backbone_out, layer in zip(backbone_outs, self.detection_layers):
            out = layer(backbone_out)
            outs.append(out)

        outs = outs + [self.anchors]

        return tuple(outs)

def SSDPostprocess(output, variances=(0.1, 0.2)):
    conf_batch, loc_batch, anchors = output
    conf_batch = F.softmax(conf_batch, dim=-1)

    # conf_batch = conf_batch.cpu()
    # loc_batch = loc_batch.cpu()
    # anchors = anchors.cpu()


    labels_batch, scores_batch = SSDDecodeConf(conf_batch)
    bboxes_batch = SSDDecodeBoxes(loc_batch, anchors, variances=variances)

    return bboxes_batch, labels_batch, scores_batch
