import torch
import torchvision

from utils import STATE_DICT_PATH, MODEL_CONSTRUCTOR, MODEL_WEIGHTS

def main():
  model = MODEL_CONSTRUCTOR(
    weights=MODEL_WEIGHTS
  )
  model.train()

  # do some training

  torch.save(model.state_dict(), STATE_DICT_PATH)

if __name__ == "__main__":
  main()