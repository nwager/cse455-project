"""
This file is a combination of functions found in the Pytorch detection
reference: https://github.com/pytorch/vision/tree/main/references/detection
"""

import torch
from torch import nn, Tensor
from torchvision.transforms import functional as F, transforms as T
from typing import Dict, Optional, Tuple, List
from PIL import Image

from utils import DEVICE

# engine.py

def train_one_epoch(model, optimizer, data_loader, epoch, print_freq=0):
  model.train()

  epoch_losses: List[float] = []
  
  for i, (images, targets) in enumerate(data_loader):
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v[j].to(DEVICE) for k, v in targets.items()} for j in range(len(images))]
    loss_dict = model(images, targets)
    loss: torch.Tensor = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    epoch_losses.append(loss.item())
  
    if print_freq > 0 and (i+1) % print_freq == 0:
      print(f'{epoch}, {i+1}: {loss}')

  return epoch_losses

# transforms.py

class Compose:
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, image, target):
    for t in self.transforms:
      image, target = t(image, target)
    return image, target

class PILToTensor(nn.Module):
  def forward(
    self, image: Image.Image, target: Optional[Dict[str, Tensor]] = None
  ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    image: Tensor = F.pil_to_tensor(image)
    if (image.shape[0] != 3):
      print(image.shape)
    return image, target
  
class RandomHorizontalFlip(T.RandomHorizontalFlip):
  def forward(
    self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
  ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    if torch.rand(1) < self.p:
      image = F.hflip(image)
      if target is not None:
        _, _, width = F.get_dimensions(image)
        target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
    return image, target

class ConvertImageDtype(nn.Module):
  def __init__(self, dtype: torch.dtype) -> None:
    super().__init__()
    self.dtype = dtype

  def forward(
    self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
  ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    image = F.convert_image_dtype(image, self.dtype)
    return image, target