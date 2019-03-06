import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import albumentations as albu
from albumentations import Compose, OneOf, Blur
from albumentations import DualTransform, ImageOnlyTransform
import albumentations.augmentations.functional as F


class ChannelsFirst(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        return np.transpose(image, (2, 0, 1))

class ImageToTensor(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        return torch.from_numpy(image.astype(np.float32))


class Expand(DualTransform):
    def __init__(self, value, p=0.5, expand_ratio=4.0, always_apply=False):
        super().__init__(always_apply, p)
        self.value = value
        self.expand_ratio = expand_ratio

    def update_params(self, params, **kwargs):
        params = super().update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']

        ratio = random.uniform(1, self.expand_ratio)
        left = int(random.uniform(0, cols * ratio - cols))
        top = int(random.uniform(0, rows * ratio - rows))

        params.update({'pad_top': top,
                       'pad_bottom': 0,
                       'pad_left': left,
                       'pad_right': 0})
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right,
                                 border_mode=cv2.BORDER_CONSTANT, value=self.value)

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = F.denormalize_bbox(bbox, rows, cols)
        bbox = [x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top]
        return F.normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, a, s = keypoint
        return [x + pad_left, y + pad_top, a, s]

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]
def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]



class RandomSSDCrop(DualTransform):

    def __init__(self, scale_range=(0.3, 1.0), ar_range=(0.5, 2.0),
                 always_apply=False, p=0.8, **params):
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.ar_range = ar_range
        self.sample_options = (
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # (None, None)
        )

    def apply(self, img, x_min=0, y_min=0, x_max=0, y_max=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.crop(img, x_min, y_min, x_max, y_max)

    def get_random_params(self, height, weight):
        ar = random.uniform(self.ar_range[0], self.ar_range[1])
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        w = min(width * scale * np.sqrt(ar), width)
        h = min(height * scale / np.sqrt(ar), height)

        x_shift = random.uniform(width - w)
        y_shift = random.uniform(height - h)

        return {'x_min': int(x_shift),
                'y_min': int(y_shift),
                'x_max': int(x_shift + w),
                'y_max': int(y_shift + h)}

    def get_params_dependent_on_targets(self, params):
        height, width, _ = params['image'].shape

        if 'bboxes' not in params:
            return self.get_random_params(height, width)
        if len(params['bboxes']) == 0:
            return self.get_random_params(height, width)
        boxes = np.array(params['bboxes'])[:, :4].copy()
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        while True:
            # randomly choose a mode
            r_idx = random.choice(list(range(len(self.sample_options))))
            mode = self.sample_options[r_idx]

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                ar = random.uniform(self.ar_range[0], self.ar_range[1])
                scale = random.uniform(self.scale_range[0], self.scale_range[1])

                w = min(width * scale * np.sqrt(ar), width)
                h = min(height * scale / np.sqrt(ar), height)

                x_shift = random.uniform(width - w)
                y_shift = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(x_shift), int(y_shift), int(x_shift + w), int(y_shift + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue
                return {'x_min': int(x_shift),
                        'y_min': int(y_shift),
                        'x_max': int(x_shift + w),
                        'y_max': int(y_shift + h)}

    def apply_to_bbox(self, bbox, x_min=0, y_min=0, x_max=0, y_max=0, rows=0, cols=0, **params):
        return F.bbox_crop(bbox, x_min, y_min, x_max, y_max, rows, cols)

    @property
    def targets_as_params(self):
        return ['image', 'bboxes']

class InternalCompose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


# class PhotometricDistort_old(object):
#     def __init__(self):
#         self.pd = [
#             RandomContrast(),  # RGB
#             ConvertColor(current="RGB", transform='HSV'),  # HSV
#             RandomSaturation(),  # HSV
#             RandomHue(),  # HSV
#             ConvertColor(current='HSV', transform='RGB'),  # RGB
#             RandomContrast()  # RGB
#         ]
#         self.rand_brightness = RandomBrightness()
#         self.rand_light_noise = RandomLightingNoise()
#
#     def __call__(self, image, boxes, labels):
#         im = image.copy()
#         im, boxes, labels = self.rand_brightness(im, boxes, labels)
#         if random.randint(2):
#             distort = InternalCompose(self.pd[:-1])
#         else:
#             distort = InternalCompose(self.pd[1:])
#         im, boxes, labels = distort(im, boxes, labels)
#         return self.rand_light_noise(im, boxes, labels)
    
class PhotometricDistort(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def apply(self, image, **params):
        im = image.copy()
        im, _, _ = self.rand_brightness(im, None, None)
        if random.randint(0, 1):
            distort = InternalCompose(self.pd[:-1])
        else:
            distort = InternalCompose(self.pd[1:])
        im, _, _ = distort(im, None, None)
        im, _, _ = self.rand_light_noise(im, None, None)
        return im
