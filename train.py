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
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(root = "./chest_xray/train", transform = train_transform)
num_classes = len(set(train_dataset.targets))
print("Number of classes:", num_classes)

## Splitting the dataset into training and validation sets
train, val = torch.utils.data.random_split(train_dataset, [4000, 5232-4000])        # 4000 images for training, 1232 images for validation - Total Images in training set = 5232
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)     # Loads the training data in batches of 128 with shuffling
val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=True)         # Loads the validation data in batches of 128 with shuffling

num_batches = len(train_loader)
print("Number of batches in Training data set:", num_batches)
num_batches = len(val_loader)
print("Number of batches in Validation data set:", num_batches)

## Configuring model parameters
model = ResNet18(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Training loop of the model
epochs = 10

for epoch in range(epochs):
    loss_list = []                          # List to store the loss of each batch in the training set
    num_correct, num_samples = 0, 0         # Variables to store the number of correct predictions and the number of samples

    model.train()                           # Sets the model to training mode
    for batch_num, (images, labels) in enumerate(train_loader):      # tqdm is used to display a progress bar for the training loop
        images, labels = images.to(device), labels.to(device)

        ## Forward Pass
        output = model(images)
        loss = criterion(output, labels)
        loss_list.append(loss.item())

        ## Backward Pass
        optimizer.zero_grad()                           # Zeroes the gradients of the model parameters - This is done to prevent the gradients from accumulating
        loss.backward()                                 # Computes the gradients of the loss w.r.t the model parameters using backpropagation
        optimizer.step()                                # Updates the model parameters using the computed gradients

        ## Calculating the number of correct predictions
        _, predictions = output.max(1)                  # Predictions are the indices of the maximum value in the output tensor
        num_correct += (predictions == labels).sum()    # Sums the number of correct predictions
        num_samples += predictions.size(0)              # Sums the number of samples

    accuracy = num_correct / num_samples
    print("Epoch:", epoch, "Loss:", sum(loss_list)/len(loss_list), "Accuracy Percentage:", accuracy*100)

    ## Validation loop of the model
    val_loss_list = []                          # List to store the loss of each batch in the validation set
    val_num_correct, val_num_samples = 0, 0     # Variables to store the number of correct predictions and the number of samples

    model.eval()                               # Sets the model to evaluation mode

    with torch.no_grad():                      # Disables the gradient computation
        for val_batch_num, (val_images, val_labels) in enumerate(val_loader):

            val_images, val_labels = val_images.to(device), val_labels.to(device)

            ## Forward Pass
            val_output = model(val_images)
            val_loss = criterion(val_output, val_labels)
            val_loss_list.append(val_loss.item())

            ## Calculating the number of correct predictions
            _, val_predictions = val_output.max(1)                      # Predictions are the indices of the maximum value in the output tensor
            val_num_correct += (val_predictions == val_labels).sum()
            val_num_samples += val_predictions.size(0)

    val_accuracy = float(val_num_correct) / float(val_num_samples)
    print("Validation Loss:", sum(val_loss_list)/len(val_loss_list), "Validation Accuracy Percentage:", val_accuracy*100)

## Saving the model parameters
torch.save(model.state_dict(), 'model.pth')