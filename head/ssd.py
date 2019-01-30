import torch
from torch import nn

import numpy as np

from utils.numpy import SSDDecode, NMS

default_aspect_ratios = [
    [2, 1, 1 / 2],
    [2, 1 / 2, 1, 3, 1 / 3],
    [2, 1 / 2, 1, 3, 1 / 3],
    [2, 1 / 2, 1, 3, 1 / 3],
    [2, 1, 1 / 2],
    [2, 1, 1 / 2]
]

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
    def __init__(self, aspect_ratios, smin=0.2, smax=0.9):
        '''

        :param aspect_ratios:
        :param smin:
        :param smax:
        '''
        self.aspect_ratios = aspect_ratios
        self.smin = smin
        self.smax = smax

        self.anchors_per_cell = [len(ars) + 1 for ars in aspect_ratios]

    def __call__(self, k, box):
        s_k = s_(k, self.smin, self.smax, len(self.aspect_ratios))
        s_kplus = s_(k, self.smin, self.smax, len(self.aspect_ratios))
        s_prime = np.sqrt(s_k * s_kplus)

        cell_cx, cell_cy, cell_w, cell_h = box

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

class DefaultAnchorCreator(AnchorCellCreator):
    def __init__(self):
        super().__init__(default_aspect_ratios)

def create_anchors(sizes, anchor_creator):
    anchors = []
    for k, (H, W) in enumerate(sizes):
        cell_h = 1.0 / H
        cell_w = 1.0 / W
        for h_i in range(H):
            for w_i in range(W):
                cell_cx = (w_i + 0.5) * cell_w
                cell_cy = (h_i + 0.5) * cell_h

                boxes = anchor_creator(k, [cell_cx, cell_cy, cell_w, cell_h])
                anchors.append(boxes)

    anchors = np.concatenate(anchors, axis=0)
    return torch.from_numpy(anchors)

class SSDHead(nn.Module):
    def __init__(self, n_classes, in_channels, anchor_creator=DefaultAnchorCreator()):
        super().__init__()

        self.n_classes      = n_classes
        self.c_             = n_classes + 1 # Background class
        self.in_channels    = in_channels
        self.anchor_creator = anchor_creator


        self.channels_per_anchor = 4 + self.c_

        self.detection_layers = []
        for in_channel, anchors_num in zip(self.in_channels, anchor_creator.anchors_per_cell):
            layer = nn.Conv2d(in_channel,  anchors_num * self.channels_per_anchor, 3, padding=1)
            self.detection_layers.append(layer)

        self.detection_layers = nn.ModuleList(self.detection_layers)

        self.last_H0 = None
        self.last_W0 = None
        self.anchors = None


    def forward(self, backbone_outs):
        device = backbone_outs[0].device
        batch_size = backbone_outs[0].shape[0]

        sizes = []
        outs = []
        for backbone_out, layer in zip(backbone_outs, self.detection_layers):
            out = layer(backbone_out).permute((0, 2, 3, 1)).contiguous()
            H, W = out.shape[1], out.shape[2]
            sizes.append((H, W))
            out = out.view((batch_size, -1, self.channels_per_anchor))
            outs.append(out)
        output = torch.cat(outs, dim=1)

        conf = output[:, :, :self.c_]
        loc = output[:, :, self.c_:]

        H0, W0 = sizes[0]
        if H0 != self.last_H0 or W0 != self.last_W0:
            self.anchors = create_anchors(sizes, self.anchor_creator)
            self.anchors = self.anchors.to(device)

        return conf, loc, self.anchors

def SSDPostprocess(conf_batch, loc_batch, anchors, score_thresh=0.01, nms_threshold=0.5, variances=(0.1, 0.2)):
    conf_batch_np = conf_batch.cpu().numpy()
    loc_batch_np = loc_batch.cpu().numpy()
    anchors_np = anchors.cpu().numpy()

    bboxes_batch, labels_batch, scores_batch = SSDDecode(
        conf_batch_np,
        loc_batch_np,
        anchors_np,
        variances
    )

    chosen_bboxes = []
    chosen_labels = []
    chosen_scores  = []
    for bboxes, labels, scores in zip(bboxes_batch, labels_batch, scores_batch):
        bboxes = bboxes[scores > score_thresh]
        labels = labels[scores > score_thresh]
        scores  = scores[scores > score_thresh]

        selected_indices = NMS(bboxes, scores, labels=labels, threshold=nms_threshold)

        bboxes = bboxes[selected_indices]
        labels = labels[selected_indices]
        scores = scores[selected_indices]

        chosen_bboxes.append(bboxes)
        chosen_labels.append(labels)
        chosen_scores.append(scores)

    return chosen_bboxes, chosen_labels, chosen_scores