import torch
from torch.utils.data import Dataset
import torchvision
import pandas as pd
import os
from PIL import Image
import numpy as np
from shapely import Polygon
import reference

from utils import COCO_NAME_TO_IDX, bbox_to_corners

PET_LABELS = {s: COCO_NAME_TO_IDX[s] for s in ['cat', 'dog']}

TRAIN_IMG_SIZE = (128, 128)

class OxfordPetDataset(Dataset):
  def __init__(self, image_dir: str, trimap_dir: str, transform=None):
    """
    Constructs a Dataset for the Oxford IIIT Pet Dataset.

    Args:
      image_dir (str): Path to directory with pet images. Should only have
        JPG files.
      trimap_dir (str): Path to directory with trimap annotations.
        Should only have trimap image files.
    """
    self.image_dir = image_dir
    self.transform = transform
    # frame to map index to image and its bbox
    entry_list = filter(
      lambda x: x != None,
      [ entry_from_trimap(os.path.join(trimap_dir, f))
        for f in os.listdir(trimap_dir) ]
    )
    self.ann_frame = pd.DataFrame(
      entry_list, columns=['name', 'label', 'x1', 'y1', 'x2', 'y2']
    )
  
  def __len__(self):
    return len(self.ann_frame)
  
  def __getitem__(self, idx):
    img_path = os.path.join(self.image_dir, self.ann_frame.iloc[idx, 0] + '.jpg')
    image = Image.open(img_path).convert('RGB')
    
    label = self.ann_frame.iloc[idx, 1]
    bbox = self.ann_frame.iloc[idx, 2:].astype('float').to_numpy()
    
    target = {
      'boxes': torch.tensor(bbox.reshape((1, 4))),
      'labels': torch.tensor([label]),
      'image_id': torch.tensor([idx]),
      'area': torch.tensor([Polygon(bbox_to_corners(bbox)).area]),
      'iscrowd': False
    }

    if self.transform:
      image, target = self.transform(image, target)

    return image, target

# Transforms

def get_transform(train: bool):
  transforms = []
  transforms.append(reference.PILToTensor())
  transforms.append(Rescale(TRAIN_IMG_SIZE))
  transforms.append(reference.ConvertImageDtype(torch.float))
  if train:
    transforms.append(reference.RandomHorizontalFlip(0.5))
  return reference.Compose(transforms)

class Rescale(object):
  """Rescale the tensor image in a sample to a given size.

  Args:
    output_size (tuple or int): Desired output size. If int, resizes to
      square with that side length.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, tuple):
      self.output_size = output_size
    else:
      self.output_size = (output_size, output_size)

  def __call__(self, image: torch.Tensor, target: dict):
    h, w = image.shape[1:]
    new_h, new_w = self.output_size
    new_h, new_w = int(new_h), int(new_w)

    image = torchvision.transforms.Resize((new_h, new_w), antialias=True)(image)

    scale_x, scale_y = new_w / w, new_h / h
    boxes = target['boxes']
    area = target['area']
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    area *= scale_x * scale_y

    target['boxes'] = boxes
    target['area'] = area

    return image, target

# Helpers
        
def entry_from_trimap(path: str) -> dict:
  trimap_im = Image.open(path)
  np_tri = np.array(trimap_im)

  # mask all pixels with a pet
  pet_mask: np.ndarray = np_tri == 1
  # check if no mask
  if (~pet_mask).all():
    return None
  # get bounding box of pet pixels
  rows = np.any(pet_mask, axis=1)
  cols = np.any(pet_mask, axis=0)
  y1, y2 = np.where(rows)[0][[0, -1]]
  x1, x2 = np.where(cols)[0][[0, -1]]

  name = os.path.basename(os.path.normpath(path))
  name = name.rsplit('.', 1)[0]

  # Oxford specifies uppercase first letter as cats, lowercase as dogs
  entry = {
    'name': name,
    'label': PET_LABELS.get('cat' if name[0].isupper() else 'dog'),
    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
  }

  return entry