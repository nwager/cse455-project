import torch
import torchvision

from utils import STATE_DICT_PATH

def main():
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
  )
  model.train()

  # do some training

  torch.save(model.state_dict(), STATE_DICT_PATH)

if __name__ == "__main__":
  main()