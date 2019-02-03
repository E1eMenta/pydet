import torch
def iou_v2v(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [xmin, ymin, xmax, ymax].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''

    N = box1.shape[0]
    M = box2.shape[0]
    lt = torch.max(
        torch.unsqueeze(box1[:, :2], dim=1),    # [N,2] -> [N,1,2]
        torch.unsqueeze(box2[:, :2], dim=0)     # [M,2] -> [1,M,2]
    )                                           # -> [N,M,2]
    rb = torch.min(
        torch.unsqueeze(box1[:, 2:], dim=1),    # [N,2] -> [N,1,2] -> [N,M,2]
        torch.unsqueeze(box2[:, 2:], dim=0)     # [M,2] -> [1,M,2] -> [N,M,2]
    )                                           # -> [N,M,2]

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = torch.unsqueeze(area1, dim=1)   # [N,] -> [N,1]
    area2 = torch.unsqueeze(area2, dim=0)   # [M,] -> [1,M]

    iou = inter / (area1 + area2 - inter)   # -> [N,M]
    return iou

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def SSDEncode(targets, anchors, variances=(0.1, 0.2), threshold=0.5):
    anchors_num = anchors.shape[0]
    conf_batch = []
    loc_batch = []
    anchors_point = point_form(anchors)
    for batch_idx, target in enumerate(targets):
        conf = torch.zeros((anchors_num), dtype=torch.int64).to(anchors.device)
        loc = torch.ones((anchors_num, 4), dtype=torch.float32).to(anchors.device)

        boxes = target[0]
        labels = target[1]

        if len(boxes) == 0:
            conf_batch.append(conf)
            loc_batch.append(loc)
            continue
        overlaps = iou_v2v(boxes, anchors_point)
        overlaps[overlaps < threshold] = 0

        max_iou, max_idx = torch.max(overlaps, dim=0)
        max_idx = max_idx[max_iou > 0]
        if len(max_idx) == 0:
            conf_batch.append(conf)
            loc_batch.append(loc)
            continue

        conf[max_iou > 0] = labels[max_idx] + 1

        matched_boxes = boxes[max_idx]
        matched_anchors = anchors[max_iou > 0]

        g_cxcy = ((matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2) - matched_anchors[:, :2]
        g_cxcy /= (variances[0] * matched_anchors[:, 2:])
        g_wh = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / matched_anchors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        matched_loc = torch.cat([g_cxcy, g_wh], dim=1)
        loc[max_iou > 0] = matched_loc

        conf_batch.append(conf)
        loc_batch.append(loc)

    conf_batch = torch.stack(conf_batch)
    loc_batch = torch.stack(loc_batch)

    return conf_batch, loc_batch