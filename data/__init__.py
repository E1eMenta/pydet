import torch
import numpy as np

from .coco import CocoDataset

def detection_collate(batch):
    images = np.stack([sample[0] for sample in batch]).astype(np.float32)
    images = torch.from_numpy(images)
    boxes = [sample[1][0] for sample in batch]
    labels = [sample[1][1] for sample in batch]

    return images, (boxes, labels)