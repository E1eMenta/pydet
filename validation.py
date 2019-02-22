import torch
import numpy as np

from pydet.utils.numpy import iou_v2v, postprocess

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ImageParams:
    def __init__(
            self,
            predBoxes,
            predScores,
            predLabels,
            trueBoxes,
            trueLabels
    ):
        self.predBoxes = predBoxes
        self.predScores = predScores
        self.predLabels = predLabels
        self.trueBoxes = trueBoxes
        self.trueLabels = trueLabels

        self.IoU_table = None

    def match(
            self,
            match_threshold=0.5
              ):
        if self.IoU_table is None:
            idx_sort = np.argsort(-self.predScores)
            pred_boxes = self.predBoxes[idx_sort]
            pred_labels = self.predLabels[idx_sort]
            self.IoU_table = iou_v2v(pred_boxes, self.trueBoxes)
            wrong_label = np.expand_dims(pred_labels, axis=1) != np.expand_dims(self.trueLabels, axis=0)
            self.IoU_table[wrong_label] = 0

        pred_matched = np.zeros((len(self.predBoxes)), dtype=np.int32)
        true_matched = np.zeros((len(self.trueBoxes)), dtype=np.int32)
        if len(self.trueBoxes) == 0:
            return pred_matched, true_matched

        iou_table = np.copy(self.IoU_table)
        iou_table[iou_table < match_threshold] = 0

        for pred_idx, row in enumerate(iou_table):
            filtered_row = row * np.logical_not(true_matched)
            if np.amax(filtered_row) > 0:
                pred_matched[pred_idx] = 1
                true_matched[np.argmax(filtered_row)] = 1

        return pred_matched, true_matched

class DetectionMetrics:

    def __init__(self, n_classes):
        '''
        Class to calculate detection metrics
        :param n_classes: Number of classes WITHOUT background class
        '''
        self.n_classes = n_classes

        self.imageParams_list = []


    def add_batch(
            self,
            batch_pred_boxes,
            batch_pred_scores,
            batch_pred_labels,
            batch_true_boxes,
            batch_true_labels
    ):
        '''
        Add batch to calculate different metrics.
        :param batch_pred_boxes: list [N] of numpy arrays [Q, 4], where N - batch size, Q - quantity of detected boxes.
            Dimension of size 4 is decoded as a bbox in fractional left-top-right-bottom [xmin, ymin, xmax, ymax] format.
        :param batch_pred_scores:  list [N] of numpy arrays [Q]. Confidence of corresponding boxes.
        :param batch_pred_labels: list [N] of numpy arrays [Q]. Class id of corresponding boxes.

        :param batch_true_boxes: list [N] of numpy arrays [M, 4], where N - batch size, M - quantity of annotated boxes.
            Dimension of size 4 is decoded as a bbox in fractional left-top-right-bottom [xmin, ymin, xmax, ymax] format.
        :param batch_true_labels: list [N] of numpy arrays [Q]. Class id of corresponding ground truth boxes.
        '''


        batch_size = len(batch_pred_boxes)

        for idx in range(batch_size):
            params = ImageParams(
                batch_pred_boxes[idx],
                batch_pred_scores[idx],
                batch_pred_labels[idx],
                batch_true_boxes[idx],
                batch_true_labels[idx]
            )
            self.imageParams_list.append(params)

    def addAndMatch_batch(
            self,
            batch_pred_boxes,
            batch_pred_scores,
            batch_pred_labels,
            batch_true_boxes,
            batch_true_labels,
            match_threshold
    ):
        pred_matched_list, true_matched_list = [], []
        batch_size = len(batch_pred_boxes)

        for idx in range(batch_size):
            params = ImageParams(
                batch_pred_boxes[idx],
                batch_pred_scores[idx],
                batch_pred_labels[idx],
                batch_true_boxes[idx],
                batch_true_labels[idx]
            )
            self.imageParams_list.append(params)
            pred_matched, true_matched = params.match(match_threshold)

            pred_matched_list.append(pred_matched)
            true_matched_list.append(true_matched)

        return pred_matched_list, true_matched_list

    def clear(self):
        self.imageParams_list = []

    def match_all(self, match_threshold):
        '''
        Match all available boxes with ground truth bboxes. Results are concatenated over all images.
        :param match_threshold: Intersection over union threshold to match predicted and ground truth boxes.
        :return: pred_matched: numpy array [Q]. arr[i]==1 if ith predicted bbox is matched else arr[i]==0 - false alarm.
            Q - is number of all predicted boxes over all images.
                true_matched: numpy array [M]. arr[i]==1 if ith ground truth bbox is matched else arr[i]==0 - miss detection
            M - is number of all ground truth boxes over all images.
                scores: numpy array [Q]. Scores of corresponding predicted boxes.
                pred_labels: numpy array [Q]. Labels of corresponding predicted boxes.
                true_labels: numpy array [M]. Labels of corresponding ground truth boxes.


        '''
        pred_matched = np.zeros((0), dtype=np.int32)
        true_matched = np.zeros((0), dtype=np.int32)
        scores       = np.zeros((0), dtype=np.float32)
        pred_labels  = np.zeros((0), dtype=np.int32)
        true_labels  = np.zeros((0), dtype=np.int32)


        for params in self.imageParams_list:
            pred_match_i, true_match_i = params.match(match_threshold)

            pred_matched = np.concatenate([pred_matched, pred_match_i])
            true_matched = np.concatenate([true_matched, true_match_i])
            scores = np.concatenate([scores, params.predScores])
            pred_labels = np.concatenate([pred_labels, params.predLabels])
            true_labels = np.concatenate([true_labels, params.trueLabels])

        return pred_matched, true_matched, scores, pred_labels, true_labels


    def precision_recall(
            self,
            conf_threshold=0.5,
            match_threshold=0.5
    ):
        '''
        Calculation Precision and Recall over all available data
        :param conf_threshold: Minimum confidence threshold for bbox to be in output.
            If scalar - use same threshold for each class, if list[n_classes] - use per-class threshold
        :return:  Precision and Recall
        '''
        pred_matched, true_matched, scores, pred_labels, true_labels = self.match_all(match_threshold)

        prec = np.zeros((self.n_classes), dtype=np.float32)
        rec = np.zeros((self.n_classes), dtype=np.float32)
        for class_id in range(self.n_classes):
            if type(conf_threshold) == list:
                threshold = conf_threshold[class_id]
            elif type(conf_threshold) == float:
                threshold = conf_threshold

            pred_mask = np.logical_and(pred_labels == class_id, scores > threshold)
            Tp = np.sum(pred_matched[pred_mask])
            Fp = np.sum(1 - pred_matched[pred_mask])

            gt_num = len(true_matched[true_labels == class_id])

            prec[class_id] = Tp / (Tp + Fp + 1e-5)
            rec[class_id] = Tp / (gt_num + 1e-5)
        return prec, rec

    def AP(self, match_threshold):
        '''
        Calculation of per-class Average Precision and mean Average Precision
        :param match_threshold: Intersection over union threshold to match predicted and ground truth boxes.
        :return: AP: Average Precision
                 mAP: mean Average Precision
        '''
        return self.calculate_AP(match_threshold)

    def calculate_AP(self, match_threshold=0.5):
        '''
        Calculation of per-class Average Precision and mean Average Precision
        :param match_threshold: Threshold to match predicted and ground truth boxes.
        :return: AP: Average Precision
                 mAP: mean Average Precision
        '''
        pred_matched, true_matched, scores, pred_labels, true_labels = self.match_all(match_threshold)

        idx_sort = np.argsort(-scores)
        pred_matched = pred_matched[idx_sort]
        pred_labels = pred_labels[idx_sort]
        scores = scores[idx_sort]

        min_score = 0.01
        AP = np.zeros((self.n_classes), dtype=np.float32)
        for class_id in range(self.n_classes):
            positives = len(true_matched[true_labels == class_id])
            pred_mask = np.logical_and(pred_labels == class_id, scores > min_score)

            true_positive = pred_matched[pred_mask]
            false_negative = 1 - pred_matched[pred_mask]

            true_positive = np.cumsum(true_positive)
            false_negative = np.cumsum(false_negative)




            precision = true_positive / (true_positive + false_negative + 1e-5)
            recall = true_positive / (positives + 1e-5)
            AP[class_id] = np.sum((recall[1:] - recall[:-1]) * precision[1:])

        mAP = np.mean(AP)
        return AP, mAP

    def COCO_mAP(self):
        '''
        Calculation of COCO mean Average Precision(mAP@[.5:.95])
        :return: COCO mean Average Precision
        '''
        mAPs = 0
        IoUs = np.arange(0.5, 1.0, 0.05)
        for match_threshold in IoUs:
            _, mAP = self.calculate_AP(match_threshold)
            mAPs += mAP
        result_mAP = mAPs / len(IoUs)

        return result_mAP

class DetectionValidator:
    def __init__(self,
                 data_loader,
                 criterion,
                 conf_thresh=0.5,
                 nms_threshold=0.5,
                 ):


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_loader = data_loader
        self.criterion = criterion
        self.conf_thresh = conf_thresh
        self.nms_threshold = nms_threshold

        n_classes = len(data_loader.dataset.class_names)
        self.metrics = DetectionMetrics(n_classes)

    def __call__(self, model, params):
        loss_avg = None

        for images, targets in self.data_loader:
            images = images.to(self.device)

            loss_out, model_out = model(images)

            losses = self.criterion(loss_out, targets)
            if loss_avg is None:
                loss_avg = {loss_name: AverageMeter() for loss_name in losses}
            for loss_name, loss_val in losses.items():
                loss_avg[loss_name].update(loss_val.item())

            pred_boxes, pred_labels, pred_scores = model_out
            pred_boxes = pred_boxes.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
            pred_boxes, pred_labels, pred_scores = postprocess(pred_boxes, pred_labels, pred_scores,
                                                               score_thresh=0.01,
                                                               nms_threshold=self.nms_threshold
                                                               )

            gt_boxes = targets[0]
            gt_labels = [target - 1 for target in targets[1]]
            self.metrics.add_batch(
                pred_boxes,
                pred_scores,
                pred_labels,
                gt_boxes,
                gt_labels
            )

        precision, recall = self.metrics.precision_recall(conf_threshold=0.5)
        mPrecision = np.mean(precision)
        mRecall = np.mean(recall)
        _, mAP05 = self.metrics.AP(0.5)
        COCO_mAP = self.metrics.COCO_mAP()
        self.metrics.clear()

        if params["tensorboard"]:
            eval_writer = params["eval_writer"]
            iteration = params["iteration"]

            for name, value in loss_avg.items():
                eval_writer.add_scalar("losses/" + name, value.avg, iteration)
            eval_writer.add_scalar('mAP_.5_', mAP05, iteration)
            eval_writer.add_scalar('mAP_.5_.95_', COCO_mAP, iteration)
            eval_writer.add_scalar('mPrecision_.5_', mPrecision, iteration)
            eval_writer.add_scalar('mRecall_.5_', mRecall, iteration)


        print(f"Validation: mAP@[.5:.95] {COCO_mAP:.4f}, mAP@[.5] {mAP05:.4f}, mPrecision@[.5] {mPrecision:.4f}, mRecall@[.5] {mRecall:.4f}", end=" ")
        for name, value in loss_avg.items():
            print("{} {:.4f},".format(name, value.avg), end=" ")
        print()