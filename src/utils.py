import torch
import torchvision

# coco
COCO_CLASSES = [
  '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
  'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
  'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
  'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
COCO_NAME_TO_IDX = {name:idx for (idx, name) in enumerate(COCO_CLASSES)}

# my stuff
STATE_DICT_PATH = './state_dict.pt'

def bbox_to_corners(bbox):
  x1, y1, x2, y2 = bbox
  return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))

MODEL_CONSTRUCTOR =  torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn
MODEL_WEIGHTS = 'COCO_V1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ModelType = torchvision.models.detection.FasterRCNN

DATASET_IMAGES = '../data/Oxford_IIIT_Pet_Dataset/images'
DATASET_TRIMAPS = '../data/Oxford_IIIT_Pet_Dataset/annotations/trimaps'