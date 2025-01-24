## This file is used to test the model with random data of size 1 X 3 X 224 X 224

from resnet_18 import ResNet18

import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ResNet18(100).to(device)
data = torch.randn(100, 3, 224, 224).to(device)   ## Random data of size 1 X 3 X 224 X 224
output = model(data)

print(output.shape)
print(model)