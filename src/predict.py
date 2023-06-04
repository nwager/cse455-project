import torch
import torchvision
import numpy as np
from shapely.geometry import Polygon
import typing

from utils import COCO_NAME_TO_IDX, bbox_to_corners, MODEL_CONSTRUCTOR, DEVICE

SCORE_THRESHOLD = 0.4

class KBDPredictor:
  def __init__(self, state_dict_path: str, threshold=0.01):
    self.model = MODEL_CONSTRUCTOR(weights=None)
    self.model.load_state_dict(torch.load(state_dict_path))
    self.model.to(DEVICE)
    self.model.eval()

    self.threshold = threshold

  @torch.no_grad()
  def predict(self, img: np.ndarray) -> typing.Tuple[bool, torch.Tensor, torch.Tensor]:
    """
    Given a (ndarray) image, predict whether it shows a cat on a keyboard.
    Args:
      img: (ndarray) RGB image to analyze
    Returns:
      (Tuple[bool, Tensor, Tensor]) Tuple where first element is whether
        there is a cat on the keyboard and the second and third are Tensors
        of cat and keyboard bounding boxes of shape N x 4 each.
    """
    transformed_img = torchvision.transforms.transforms.ToTensor()(
      torchvision.transforms.ToPILImage()(img)
    ).to(DEVICE)

    result = self.model([transformed_img])[0]
  
    boxes: torch.Tensor = result['boxes']
    labels: torch.Tensor = result['labels']
    scores: torch.Tensor = result['scores']

    # remove low scores
    parsed = torch.concat(
      [labels.reshape(-1, 1), scores.reshape(-1, 1), boxes.reshape(-1, 4)], dim=1
    )[scores > SCORE_THRESHOLD]

    # [label, score, x1, y1, x2, y2]
    cat_results = parsed[parsed[:, 0] == COCO_NAME_TO_IDX['cat']]
    kb_results = parsed[parsed[:, 0] == COCO_NAME_TO_IDX['keyboard']]

    if (len(cat_results) == 0 or len(kb_results) == 0):
      return (False, torch.tensor([]), torch.tensor([]))
    
    cat_bboxes = cat_results[:, 2:]
    cat_full_bb = torch.concat([cat_bboxes[:, :2].max(dim=0)[0], cat_bboxes[:, 2:].max(dim=0)[0]])
    
    kb_bboxes = kb_results[:, 2:]
    kb_full_bb = torch.concat([kb_bboxes[:, :2].max(dim=0)[0], kb_bboxes[:, 2:].max(dim=0)[0]])

    cat_poly = Polygon(bbox_to_corners(cat_full_bb))
    kb_poly = Polygon(bbox_to_corners(kb_full_bb))
    ixn = cat_poly.intersection(kb_poly)
    kb_cover = ixn.area / kb_poly.area

    return (kb_cover > self.threshold, cat_bboxes, kb_bboxes)
