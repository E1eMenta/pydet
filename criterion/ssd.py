import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.numpy import SSDEncode

class SSDLoss(nn.Module):
    def __init__(self, match_thresh=0.5, neg_pos=3,  variance=(0.1, 0.2)):
        super().__init__()
        self.match_thresh = match_thresh
        self.neg_pos = neg_pos
        self.variance = variance

    def forward(self, model_output, targets):
        pred_conf, pred_loc, anchors = model_output
        batch_size, anchors_num, n_classes = pred_conf.shape

        anchors = anchors.cpu().numpy()
        targets = [(target[0].cpu().numpy(), target[1].cpu().numpy()) for target in targets]

        target_conf, target_loc = SSDEncode(
            targets, anchors,
            variances=self.variance,
            threshold=self.match_thresh
        )

        target_conf = torch.from_numpy(target_conf).to(pred_conf.device).long()
        target_loc = torch.from_numpy(target_loc).to(pred_conf.device)

        positive_position = target_conf > 0
        N = torch.sum(positive_position).float()

        # Localization Loss (Smooth L1)
        loc_p = pred_loc[positive_position]
        loc_t = target_loc[positive_position]
        localization_loss = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Classification loss
        class_loss = 0
        for batch_i in range(batch_size):
            target_conf_i, target_loc_i = target_conf[batch_i], target_loc[batch_i]
            pred_conf_i, pred_loc_i = pred_conf[batch_i], pred_loc[batch_i]

            positive_num = torch.sum(target_conf_i > 0)
            negative_num = max(10, min(positive_num * self.neg_pos, anchors_num - positive_num))
            class_loss_i = F.cross_entropy(pred_conf_i, target_conf_i, reduction='none')

            class_loss_positives = class_loss_i[target_conf_i > 0]
            class_loss_negatives = class_loss_i[target_conf_i == 0]

            _, loss_idx = class_loss_negatives.sort(0, descending=True)
            class_loss_negatives = class_loss_negatives[loss_idx[:negative_num]]

            class_loss += torch.sum(class_loss_positives) + torch.sum(class_loss_negatives)

        class_loss = class_loss / N
        localization_loss = localization_loss / N
        total_loss = class_loss + localization_loss

        return total_loss