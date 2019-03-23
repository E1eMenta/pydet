import torch
from torch import nn
import numpy as np

from .ssd import create_anchors, s_

class AnchorCellCreator:
    def __init__(self, aspect_ratios, scales, layers_num, smin=0.05, smax=0.87):
        self.aspect_ratios = aspect_ratios
        self.smin = smin
        self.smax = smax
        self.scales = scales
        self.layers_num = layers_num

        self.anchors_per_cell = len(aspect_ratios) * len(scales)

    def __call__(self, k, H, W, h_i, w_i):
        s_k = s_(k, self.smin, self.smax, self.layers_num)

        cell_h = 1.0 / H
        cell_w = 1.0 / W
        cell_cx = (w_i + 0.5) * cell_w
        cell_cy = (h_i + 0.5) * cell_h

        anchors = []
        for scale in self.scales:
            for ar in self.aspect_ratios:
                w = s_k * scale * np.sqrt(ar)
                h = s_k * scale / np.sqrt(ar)
                anchors.append(np.array([cell_cx, cell_cy, w, h]))

        return np.stack(anchors)


class BoxSubnet(nn.Module):
    def __init__(self, num_features_in, num_anchors, feature_size=None):
        super().__init__()
        feature_size = num_features_in if feature_size is None else feature_size

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        return out

class ClassSubnet(nn.Module):
    def __init__(self, num_features_in, num_anchors, num_classes, feature_size=None):
        super().__init__()
        feature_size = num_features_in if feature_size is None else feature_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)
        return out

class RetainaNetHead(nn.Module):
    def __init__(self, n_classes, in_channels, sizes, anchor_creator):
        super().__init__()

        self.c_             = n_classes + 1 # Background class
        self.in_channels    = in_channels
        self.sizes          = sizes

        anchors, anchors_per_cell = create_anchors(sizes, anchor_creator)
        self.register_buffer('anchors', anchors)
        anchors_per_cell = anchors_per_cell[0]

        self.class_subnet = ClassSubnet(in_channels,
                                        num_anchors=anchors_per_cell,
                                        num_classes=self.c_
                                        )
        self.box_subnet = BoxSubnet(in_channels,
                                    num_anchors=anchors_per_cell,
                                    )


    def forward(self, backbone_outs):
        device = backbone_outs[0].device
        batch_size = backbone_outs[0].shape[0]


        class_outs = []
        box_outs = []
        for backbone_out in backbone_outs:
            class_out = self.class_subnet(backbone_out).permute((0, 2, 3, 1)).contiguous()
            box_out = self.box_subnet(backbone_out).permute((0, 2, 3, 1)).contiguous()
            class_out = class_out.view((batch_size, -1, self.c_))
            box_out = box_out.view((batch_size, -1, 4))

            class_outs.append(class_out)
            box_outs.append(box_out)
        class_outs = torch.cat(class_outs, dim=1)
        box_outs = torch.cat(box_outs, dim=1)

        return class_outs, box_outs, self.anchors

    def minimal_forward(self, backbone_outs):
        class_box_outs = []
        for backbone_out in backbone_outs:
            class_out = self.class_subnet(backbone_out)
            box_out = self.box_subnet(backbone_out)

            class_box_outs += [class_out, box_out]

        class_box_outs = class_box_outs + [self.anchors]

        return tuple(outs)