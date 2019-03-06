import os
import random
import urllib.request
import zipfile
import numpy as np
import cv2

from torch.utils.data import Dataset

from pycocotools.coco import COCO

class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None, download=False):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        if download and not os.path.exists(os.path.join(root_dir, "train2017")):
            print("Download COCO")
            def download(path, root):
                basename = os.path.basename(path)
                urllib.request.urlretrieve(path, os.path.join(root, basename))

                with zipfile.ZipFile(os.path.join(root, basename), 'r') as zip_ref:
                    zip_ref.extractall(root)

                os.remove(os.path.join(root, basename))

            download("http://images.cocodataset.org/zips/val2017.zip", root_dir)
            download("http://images.cocodataset.org/zips/train2017.zip", root_dir)
            download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", root_dir)

            print("Done")

        annotation_path = os.path.join(
            self.root_dir,
            'annotations',
            'instances_' + self.set_name + '.json'
        )
        self.coco = COCO(annotation_path)
        self.image_ids = self.coco.getImgIds()
        self.image_ids = [id for id in self.image_ids if len(self.coco.getAnnIds(imgIds=id, iscrowd=False)) > 0]

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.class_names = list(self.classes.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annot = self.load_annotations(idx)

        bboxes = annot[:, :4].astype(np.float32)
        labels = annot[:, 4].astype(np.int64) + 1

        # print("start", len(bboxes))
        if self.transform:
            data = self.transform(image=image, bboxes=bboxes, labels=labels)
            image, bboxes, labels = data["image"], data["bboxes"], data["labels"]

        channels, height, width = image.shape
        labels = np.array(labels, dtype=np.int64)
        bboxes = np.array(bboxes, dtype=np.float32)
        if len(bboxes) > 0:
            bboxes[:, 0] /= width
            bboxes[:, 2] /= width
            bboxes[:, 1] /= height
            bboxes[:, 3] /= height
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)

        return image, (bboxes, labels)

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])


# from torchvision.transforms import Compose