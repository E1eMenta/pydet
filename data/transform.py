import numpy as np
import random
import cv2

from utils.numpy import iou_b2v

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels

class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        image = np.maximum(image, 0)
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, p=0.5):
        self.lower = lower
        self.upper = upper
        self.p = p
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0, p=0.5):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.p = p

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
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
        image = image[:, :, self.swaps]
        return image
class RandomLightingNoise(object):
    def __init__(self, p=0.5):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        self.p = p

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            swap = self.perms[random.randint(0, len(self.perms)-1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, p=0.5):
        self.lower = lower
        self.upper = upper
        self.p = p
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32, p=0.5):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.p = p

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class PhotometricDistort(object):
    def __init__(self, p=0.5):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        self.p = p

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.random() < self.p:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class Expand(object):
    def __init__(self, ratio=4, p=0.5):
        self.ratio = ratio
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() > self.p:
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, self.ratio)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class SSDRandomCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for idx in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = iou_b2v(rect, boxes)


                # is min and max overlap constraint satisfied? if not try again
                if len(overlap) > 0:
                    if overlap.min() < min_iou and max_iou < overlap.max():
                        continue
                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

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

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                return current_image, current_boxes, current_labels


class RandomMirror(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.random() < self.p:
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class Resize(object):
    def __init__(self, size=(300, 300), interpolation=cv2.INTER_AREA):
        '''
        Resize image
        :param size: size of new image. size[0] - height, size[1] - width
        '''
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=self.interpolation)
        return image, boxes, labels

class SSDAugmentation(object):
    def __init__(self, size=(300, 300), mean=(0, 0, 0)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(),
            SSDRandomCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            ToAbsoluteCoords(),
            SubtractMeans(self.mean)
        ])

    def __call__(self, **sample):
        img = sample["image"]
        boxes = sample["bboxes"]
        labels = sample["labels"]
        img, boxes, labels = self.augment(img, boxes, labels)
        return {"image": img, "bboxes": boxes, "labels": labels}

class BaseTransform(object):
    def __init__(self, size=(300, 300), mean=(0, 0, 0)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ToPercentCoords(),
            Resize(self.size),
            ToAbsoluteCoords(),
            SubtractMeans(self.mean)
        ])

    def __call__(self, **sample):
        img = sample["image"]
        boxes = sample["bboxes"]
        labels = sample["labels"]
        img, boxes, labels = self.augment(img, boxes, labels)
        return {"image": img, "bboxes": boxes, "labels": labels}