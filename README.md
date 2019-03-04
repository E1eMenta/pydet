# Pytorch detection utils
This repo implements detection heads, criterions and utils. Aim of this repo is to separate detection layers from training infrastructure.

## Dependencies
1. pytorch 1.0 or newer
2. pycocotools

## Implemented layers

### SSD(Single Shot MultiBox Detector)
SSDHead - SSD Detection Head

MultiboxLoss - SSD loss function

### Data
CocoDataset - MSCOCO dataset

VOCDataset - Pascal VOC dataset

Set of transforms for data augmentation

### Other
FPN - Feature Pyramid block. Can be added before detection head to improve small object detection.

## Detecton infrastructure
Examples of layers usage: https://github.com/E1eMenta/detection
