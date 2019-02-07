import torch
import numpy as np

def detection_collate(batch):
    images = np.stack([sample['image'] for sample in batch])
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.from_numpy(images)
    targets = [(torch.FloatTensor(sample['bboxes']), torch.LongTensor(sample['labels'])) for sample in batch]

    return images, targets