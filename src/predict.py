import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import typing

from utils import COCO_CLASSES, COCO_NAME_TO_IDX, STATE_DICT_PATH, bbox_to_corners

class KBDPredictor:
  def __init__(self, state_dict_path: str, threshold=0.2):
    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    self.model.load_state_dict(torch.load(state_dict_path))
    self.model.eval()

    self.threshold = threshold

  @torch.no_grad()
  def predict(self, img: Image.Image) -> typing.Tuple[bool, torch.Tensor, torch.Tensor]:
    """
    Given a (PIL) image, predict whether it shows a cat on a keyboard.
    Args:
      img: (Image) Image to analyze
    Returns:
      (Tuple[bool, Tensor, Tensor]) Tuple where first element is whether
        there is a cat on the keyboard and the second and third are Tensors
        of cat and keyboard bounding boxes of shape N x 4 each.
    """
    transformed_img = torchvision.transforms.transforms.ToTensor()(
      torchvision.transforms.ToPILImage()(np.array(img.convert("RGB")))
    )

    result = self.model([transformed_img])[0]
    boxes: torch.Tensor = result['boxes']
    labels: torch.Tensor = result['labels']
    scores: torch.Tensor = result['scores']

    # nms results to remove noisy predictions
    parsed = torch.concat(
      [labels.reshape(-1, 1), scores.reshape(-1, 1), boxes.reshape(-1, 4)], dim=1
    )[torchvision.ops.nms(boxes, scores, iou_threshold=0.4)]

    # [label, score, x1, y1, x2, y2]
    cat_results = parsed[parsed[:, 0] == COCO_NAME_TO_IDX['cat']]
    cat_bboxes = cat_results[:, 2:]
    kb_results = parsed[parsed[:, 0] == COCO_NAME_TO_IDX['keyboard']]
    kb_bboxes = kb_results[:, 2:]

    if (len(cat_results) == 0 or len(kb_results) == 0):
      return (False, torch.tensor([]), torch.tensor([]))

    kb_cover = torch.empty((cat_bboxes.size(0), kb_bboxes.size(0)))
    for (cat_idx, cat_bbox) in enumerate(cat_bboxes):
      for (kb_idx, kb_bbox) in enumerate(kb_bboxes):
        cat_poly = Polygon(bbox_to_corners(cat_bbox))
        kb_poly = Polygon(bbox_to_corners(kb_bbox))
        ixn = cat_poly.intersection(kb_poly).area
        kb_cover[cat_idx, kb_idx] = ixn / kb_poly.area

    return (kb_cover.max() > self.threshold, cat_bboxes, kb_bboxes)
