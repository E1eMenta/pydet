import numpy as np

def iou_b2b(bboxA, bboxB):
    '''
    Compute intersection over union of two bboxes
    :param bboxA: (ndarray) bounding bboxes, shape [4]. bbox format [xmin, ymin, xmax, ymax]
    :param bboxB: (ndarray) bounding bboxes, shape [4]. bbox format [xmin, ymin, xmax, ymax]
    :return: IoU
    '''
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    bboxAArea = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxBArea = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])

    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    return iou

def iou_b2v(bbox, bboxes):
    """
    Compute intersection over union of one bbox and set of bboxes
    :param bbox: (ndarray) bounding bboxes, shape [4]. bbox format [xmin, ymin, xmax, ymax]
    :param bboxes: (ndarray) bounding bboxes, shape [N,4]. bbox format [xmin, ymin, xmax, ymax]
    :return: IoU: Shape: [N]
    """

    A = np.maximum(bbox[:2], bboxes[:, :2])
    B = np.minimum(bbox[2:], bboxes[:, 2:])
    interArea = np.maximum(B[:, 0] - A[:, 0], 0) * np.maximum(B[:, 1] - A[:, 1], 0)
    bboxArea = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    bboxesArea = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    union = bboxArea + bboxesArea - interArea
    iou = interArea / union
    return iou

def iou_v2v(bboxesA, bboxesB):
    '''Compute intersection over union of two set of bboxes.
    :param bboxesA: (ndarray) bounding bboxes, shape [N,4]. bbox format [xmin, ymin, xmax, ymax]
    :param bboxesB: (ndarray) bounding bboxes, shape [M,4]. bbox format [xmin, ymin, xmax, ymax]
    :return: IoU: Shape: [N]
    '''

    lt = np.maximum(
        np.expand_dims(bboxesA[:, :2], axis=1),    # [N,2] -> [N,1,2]
        np.expand_dims(bboxesB[:, :2], axis=0)     # [M,2] -> [1,M,2]
    )                                             # -> [N,M,2]
    rb = np.minimum(
        np.expand_dims(bboxesA[:, 2:], axis=1),    # [N,2] -> [N,1,2] -> [N,M,2]
        np.expand_dims(bboxesB[:, 2:], axis=0)     # [M,2] -> [1,M,2] -> [N,M,2]
    )                                             # -> [N,M,2]

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (bboxesA[:, 2] - bboxesA[:, 0]) * (bboxesA[:, 3] - bboxesA[:, 1])  # [N,]
    area2 = (bboxesB[:, 2] - bboxesB[:, 0]) * (bboxesB[:, 3] - bboxesB[:, 1])  # [M,]
    area1 = np.expand_dims(area1, axis=1)   # [N,] -> [N,1]
    area2 = np.expand_dims(area2, axis=0)   # [M,] -> [1,M]

    iou = inter / (area1 + area2 - inter)   # -> [N,M]
    return iou

def point_form(bboxes):
    """ Convert bboxes from center form (cx, cy, w, h) to point form (xmin, ymin, xmax, ymax)
    :param bboxes: (ndarray) bounding bboxes, shape [N,4]. bbox format [cx, cy, w, h]
    :return: bboxes: (ndarray) bounding bboxes, shape [N,4]. bbox format [xmin, ymin, xmax, ymax]
    """
    return np.concatenate((bboxes[:, :2] - bboxes[:, 2:]/2,      # xmin, ymin
                           bboxes[:, :2] + bboxes[:, 2:]/2), 1)  # xmax, ymax

def softmax(x, axis=-1):
    exp = np.exp(x - np.max(x))
    sum_exp = np.sum(exp, axis=axis, keepdims=True)
    return exp / sum_exp

def SSDDecode(conf_batch, loc_batch, anchors, variances=(0.1, 0.2)):
    '''

    :param conf: (ndarray) SSD classification output, shape [batch size, anchors num, classes num + 1]. Zero index of
        axis=2 corresponds to background class
    :param loc: (ndarray) SSD localization output, shape [batch size, anchors num, 4]
    :param anchors: (ndarray) Anchors in center form, shape [anchors num, 4]
    :param variances: Magical numbers:)  Loss coeffs for center localization and size localization
    :return: bboxes_batch: List [batch size] of (ndarray) predicted bboxes in point (xmin, ymin, xmax, ymax) form, shape [N, 4]
             labels_batch: List [batch size] of (ndarray) predicted labels of boxes, shape [N]
             scores_batch: List [batch size] of (ndarray) predicted scores of boxes, shape [N]
    '''
    loc_batch = np.clip(loc_batch, a_min=-10e5, a_max=10)

    scores_wo_background = softmax(conf_batch, axis=-1)[:, :, 1:]
    scores_batch = np.max(scores_wo_background, axis=-1)
    labels_batch = np.argmax(scores_wo_background, axis=-1)

    anchors = anchors[np.newaxis]
    bboxes_batch = np.concatenate((
        anchors[:, :, :2] + loc_batch[:, :, :2] * variances[0] * anchors[:, :, 2:],
        anchors[:, :, 2:] * np.exp(loc_batch[:, :, 2:] * variances[1])), axis=2)


    return bboxes_batch, labels_batch, scores_batch

def SSDEncode(targets, anchors, variances=(0.1, 0.2), threshold=0.5):
    '''
    Encode bboxes to anchor format
    :param targets: List [batch size] of (bboxes, labels).
        bboxes (ndarray) bounding bboxes, shape [Q, 4] of point (xmin, ymin, xmax, ymax)
        labels (ndarray) labels, bounding bboxes of shape [Q]
    :param anchors: (ndarray) anchor bboxes, shape [anchors num, 4]
    :param variances: Magical numbers:)  Loss coeffs for center localization and size localization
    :param threshold: Minimum IoU between anchor bbox and gt bbox for anchor bbox to be matched
    :return:
        conf_batch: (ndarray) Encoded format of labels, shape [batch size, anchors num]. Zero index is background.
        loc_batch:  (ndarray) Encoded format of bbox coordinates, shape [batch size, anchors num, 4]
    '''
    anchors_num = anchors.shape[0]
    conf_batch = []
    loc_batch = []
    anchors_point = point_form(anchors)
    for batch_idx, target in enumerate(targets):
        conf = np.zeros((anchors_num), dtype=np.int32)
        loc = np.ones((anchors_num, 4), dtype=np.float32)

        bboxes = target[0]
        labels = target[1]

        if len(bboxes) == 0:
            conf_batch.append(conf)
            loc_batch.append(loc)
            continue
        overlaps = iou_v2v(bboxes, anchors_point)
        overlaps[overlaps < threshold] = 0

        max_iou = np.max(overlaps, axis=0)
        max_idx = np.argmax(overlaps, axis=0)
        max_idx = max_idx[max_iou > 0]

        conf[max_iou > 0] = labels[max_idx] + 1

        matched_boxes = bboxes[max_idx]
        matched_anchors = anchors[max_iou > 0]

        g_cxcy = ((matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2) - matched_anchors[:, :2]
        g_cxcy /= (variances[0] * matched_anchors[:, 2:])
        g_wh = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / matched_anchors[:, 2:]
        g_wh = np.log(g_wh) / variances[1]
        matched_loc = np.concatenate([g_cxcy, g_wh], axis=1)
        loc[max_iou > 0] = matched_loc

        conf_batch.append(conf)
        loc_batch.append(loc)

    conf_batch = np.stack(conf_batch)
    loc_batch = np.stack(loc_batch)

    return conf_batch, loc_batch

def NMS(bboxes, scores, labels=None, threshold=0.5, max_output_size=100 ):
    '''
    Non maximum suppression
    :param bboxes: (ndarray) Bounding bboxes, shape [Q, 4]. bboxes in point (xmin, ymin, xmax, ymax] form
    :param scores: (ndarray) Scores of bboxes, shape [Q].
    :param labels: If set (ndarray) Labels of bboxes, shape [Q]. Force IoU to 0 between two bboxes with different labels.
        In not set assume all labels have same label.
    :param threshold: (ndarray) Threshold for deciding whether bboxes overlap too much with respect to IOU.
    :param max_output_size: Maximum number of bboxes to be selected by non max suppression.
    :return: selected_indices: (ndarray) Indices of chosen bboxes.
    '''
    if len(scores) == 0:
        return []

    labels = labels if labels is not None else np.zeros(len(scores))

    idx_sort = np.argsort(scores)
    bboxes = bboxes[idx_sort]
    labels = labels[idx_sort]

    selected_indices = [idx_sort[0]]
    chosen_boxes = np.expand_dims(bboxes[0], axis=0)
    selected_labels = labels[0:1]

    for idx, (box, label) in enumerate(zip(bboxes[1:], labels[1:, np.newaxis])):
        real_idx = idx + 1

        IoUs = iou_b2v(box, chosen_boxes)
        IoUs[selected_labels != label] = 0

        if np.sum(IoUs > threshold) == 0:
            selected_indices.append(idx_sort[real_idx])
            chosen_boxes = np.concatenate([chosen_boxes, np.expand_dims(box, axis=0)], axis=0)
            selected_labels = np.concatenate([selected_labels, label])
            if len(chosen_boxes) >= max_output_size:
                break
    selected_indices = np.stack(selected_indices)

    return selected_indices