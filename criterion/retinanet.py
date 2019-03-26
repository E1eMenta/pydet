import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssd import match_boxes

class FocalLoss(nn.Module):
    def __init__(self, iou_threshold=0.5, alfa=0.25, gamma=2.0, variances=(0.1, 0.2)):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super().__init__()
        self.iou_threshold = iou_threshold
        self.alfa = alfa
        self.gamma = gamma
        self.variances = variances

    def onehot_encoding(self, labels, num_classes):
        batch_size, anchors_num = labels.shape

        labels_onehot = torch.zeros((batch_size * anchors_num, num_classes)).to(labels.device)
        labels_reshaped = labels.view(-1)
        labels_onehot[np.arange(batch_size * anchors_num), labels_reshaped] = 1
        labels_onehot = labels_onehot.view(batch_size, anchors_num, num_classes)
        return labels_onehot

    def forward(self, model_out, target):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        confidence, predicted_locations, priors = model_out
        target_boxes, target_labels = target

        device = confidence.device
        labels = []
        gt_locations = []

        for boxes_i, labels_i in zip(target_boxes, target_labels):
            true_locations_i, true_labels_i = match_boxes(boxes_i, labels_i, priors.cpu(), self.variances, self.iou_threshold)
            gt_locations.append(true_locations_i)
            labels.append(true_labels_i)


        gt_locations = torch.stack(gt_locations)
        labels = torch.stack(labels)
        gt_locations = gt_locations.to(device)
        labels = labels.to(device)
        num_pos = torch.sum(labels > 0)

        num_classes = confidence.shape[2]
        labels_onehot = self.onehot_encoding(labels, num_classes)

        prob = F.softmax(confidence, dim=2)
        classification_loss = torch.pow((1 - prob), self.gamma) * (-1) * torch.log(prob)
        classification_loss[:, :, 0] *= self.alfa
        classification_loss[:, :, 1:] *= (1 - self.alfa)
        classification_loss = torch.sum(classification_loss * labels_onehot)


        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)

        smooth_l1_loss = smooth_l1_loss/num_pos
        classification_loss = classification_loss / num_pos
        loss = smooth_l1_loss + classification_loss

        return {"loss": loss, "class": classification_loss, "loc": smooth_l1_loss}
