import torch
from torch.utils.data import DataLoader, Subset
from reference import train_one_epoch
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os

from utils import *
from OxfordPetDataset import OxfordPetDataset, get_transform

def main():
  print("Loading dataset...")
  dataset = OxfordPetDataset(
    DATASET_IMAGES, DATASET_TRIMAPS, transform=get_transform(train=True)
  )
  dataset_test = OxfordPetDataset(
    DATASET_IMAGES, DATASET_TRIMAPS, transform=get_transform(train=False)
  )

  # split the dataset in train and test set
  indices = torch.randperm(len(dataset)).tolist()
  test_len = int(0.2 * len(dataset)) # 80-20 split
  dataset = Subset(dataset, indices[:-test_len])
  dataset_test = Subset(dataset_test, indices[-test_len:])

  # define training and validation data loaders
  data_loader = DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
  )
  data_loader_test = DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
  )
  
  print("Done loading dataset")

  model = MODEL_CONSTRUCTOR(weights=MODEL_WEIGHTS).to(DEVICE)

  print("Starting training...")
  epochs = 20
  losses = train(model, data_loader, epochs, print_freq=100)
  print("Done training")
  torch.save(model.state_dict(), STATE_DICT_PATH)

  if not os.path.exists('runs'): os.makedirs('runs')

  plt.plot(range(epochs), losses)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.savefig(f'runs/train_loss.png')

  mean_ap = evaluate(model, data_loader_test)
  with open('runs/train_mean_ap.txt', 'w') as f:
    f.write(str(mean_ap))

def train(model: ModelType,
          data_loader: DataLoader,
          epochs=1,
          lr=0.01,
          momentum=0.9,
          decay=0.0005,
          print_freq=100):
  
  model.train()
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=decay)
  mean_losses = []

  for epoch in range(epochs):
    losses = train_one_epoch(model, optimizer, data_loader, epoch, print_freq)
    mean_losses.append(sum(losses) / len(losses))

  return mean_losses

def evaluate(model: ModelType, data_loader_test: DataLoader):
  was_train = model.training
  model.eval()

  mean_ap = MeanAveragePrecision()
  for images, targets in data_loader_test:
    images = list(img.to(DEVICE) for img in images)
    targets = [{k: v[j].to(DEVICE) for k, v in targets.items()} for j in range(len(images))]
    outputs = model(images)

    mean_ap.update(outputs, targets)
  
  map_val = float(mean_ap.compute()['map_50'])
    
  model.train(was_train)
  return map_val

if __name__ == "__main__":
  main()