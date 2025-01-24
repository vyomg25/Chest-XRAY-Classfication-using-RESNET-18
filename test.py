import os
import torch
import torch.nn as nn
from resnet_18 import ResNet18
import torch.optim as optim
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
from tqdm.notebook import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using:', device)

## Data Loading
transform_pipe = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val = torchvision.datasets.ImageFolder(root = "./chest_xray/test", transform = transform_pipe)
num_classes = len(set(val.targets))
print("Number of classes:", num_classes)

## Splitting the dataset into training and validation sets
val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=True)         # Loads the validation data in batches of 128 with shuffling

num_batches = len(val_loader)
print("Number of batches:", num_batches)

## Configuring model parameters
model = ResNet18(num_classes)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.to(device)
num_correct, num_samples = 0, 0         # Variables to store the number of correct predictions and the number of samples

model.eval()                            # Sets the model to training mode
for batch_num, (images, labels) in enumerate(val_loader):      
    images, labels = images.to(device), labels.to(device)

    ## Forward Pass
    output = model(images)

    ## Calculating the number of correct predictions
    _, predictions = output.max(1)                  # Predictions are the indices of the maximum value in the output tensor
    num_correct += (predictions == labels).sum()    # Sums the number of correct predictions
    num_samples += predictions.size(0)              # Sums the number of samples

val_accuracy = num_correct / num_samples
print("Validation Accuracy Percentage:", val_accuracy*100)