import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResNet18(nn.Module):
    def __init__(self, num_classes):                    ## num_classes means the number of classes in which the model will classify the images - For example if the model is classifying between cats and dogs, then num_classes = 2
        super(ResNet18, self).__init__()                ## Ensures that the ResNet18 class inherits the properties of the nn.Module class

        self.dropout_percentage = 0.5
        self.relu = nn.ReLU()

        ### Defining the blocks of the ResNet18 model - No interconnections between the blocks are defined here
        
        ## Block 1 -> Input = 3 X 224 X 224, Output = 64 X 56 X 56, 64 Kernels = 7 X 7, Stride = 2, Padding = 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## Block 2
        # Identity Block -> Input = 64 X 56 X 56, Output = 64 X 56 X 56
        self.conv2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        #self.dropout2_1 = nn.Dropout(self.dropout_percentage) - Dropout is not used in RESNET networks as they already have skip connections to prevent overfitting, vanishing gradient problem and exploding gradient problem

        # Identity Block -> Input = 64 X 56 X 56, Output = 64 X 56 X 56
        self.conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        #self.dropout2_2 = nn.Dropout(self.dropout_percentage)

        ## Block 3
        # Convolution Block -> Input = 64 X 56 X 56, Output = 128 X 28 X 28
        self.conv3_1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)
        self.concat_3_1_skip_connection = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)  ## Skip connection with different dimensions are handled by using a 1 X 1 convolutional layer
        #self.dropout3_1 = nn.Dropout(self.dropout_percentage)

        # Identity Block -> Input = 128 X 28 X 28, Output = 128 X 28 X 28
        self.conv3_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        #self.dropout3_2 = nn.Dropout(self.dropout_percentage)

        ## Block 4
        # Convolution Block -> Input = 128 X 28 X 28, Output = 256 X 14 X 14
        self.conv4_1_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.batchnorm4_1_1 = nn.BatchNorm2d(256)
        self.conv4_1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4_1_2 = nn.BatchNorm2d(256)
        self.concat_4_1_skip_connection = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)  ## Skip connection with different dimensions are handled by using a 1 X 1 convolutional layer
        #self.dropout4_1 = nn.Dropout(self.dropout_percentage)

        # Identity Block -> Input = 256 X 14 X 14, Output = 256 X 14 X 14
        self.conv4_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4_2_1 = nn.BatchNorm2d(256)
        self.conv4_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4_2_2 = nn.BatchNorm2d(256)
        #self.dropout4_2 = nn.Dropout(self.dropout_percentage)

        ## Block 5
        # Convolution Block -> Input = 256 X 14 X 14, Output = 512 X 7 X 7
        self.conv5_1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.batchnorm5_1_1 = nn.BatchNorm2d(512)
        self.conv5_1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.batchnorm5_1_2 = nn.BatchNorm2d(512)
        self.concat_5_1_skip_connection = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)  ## Skip connection with different dimensions are handled by using a 1 X 1 convolutional layer
        #self.dropout5_1 = nn.Dropout(self.dropout_percentage)

        # Identity Block -> Input = 512 X 7 X 7, Output = 512 X 7 X 7
        self.conv5_2_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.batchnorm5_2_1 = nn.BatchNorm2d(512)
        self.conv5_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.batchnorm5_2_2 = nn.BatchNorm2d(512)
        #self.dropout5_2 = nn.Dropout(self.dropout_percentage)

        ## Final Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)                                  ## Average pooling is used to reduce the dimensions of the output of the last convolutional layer and the kernel size defines the size of the output tensor, if kernel_size = 1, then the output tensor will be of size 1 X 1
        self.fc = nn.Linear(in_features=1*1*512, out_features=1000)             ## The output tensor of the last convolutional layer is flattened and passed through a fully connected layer to get the final output tensor
        self.out = nn.Linear(in_features=1000, out_features=num_classes)        ## The output tensor of the fully connected layer is passed through another fully connected layer to get the final output tensor

    def forward(self, x):

        ### Forward Propagation of the input tensor through the ResNet18 model

        ## Block 1
        block1_op = self.maxpool1(self.relu(self.batchnorm1(self.conv1(x))))

        ## Block 2
        # Identity Block
        block2_1_op = self.relu(block1_op + self.batchnorm2_1_2(self.conv2_1_2(self.relu(self.batchnorm2_1_1(self.conv2_1_1(block1_op))))))
        # Identity Block
        block2_2_op = self.relu(block2_1_op + self.batchnorm2_2_2(self.conv2_2_2(self.relu(self.batchnorm2_2_1(self.conv2_2_1(block2_1_op))))))

        ## Block 3
        # Convolution Block
        block3_1_op = self.relu(self.concat_3_1_skip_connection(block2_2_op) + self.batchnorm3_1_2(self.conv3_1_2(self.relu(self.batchnorm3_1_1(self.conv3_1_1(block2_2_op))))))
        # Identity Block
        block3_2_op = self.relu(block3_1_op + self.batchnorm3_2_2(self.conv3_2_2(self.relu(self.batchnorm3_2_1(self.conv3_2_1(block3_1_op))))))

        ## Block 4
        # Convolution Block
        block4_1_op = self.relu(self.concat_4_1_skip_connection(block3_2_op) + self.batchnorm4_1_2(self.conv4_1_2(self.relu(self.batchnorm4_1_1(self.conv4_1_1(block3_2_op))))))
        # Identity Block
        block4_2_op = self.relu(block4_1_op + self.batchnorm4_2_2(self.conv4_2_2(self.relu(self.batchnorm4_2_1(self.conv4_2_1(block4_1_op))))))

        ## Block 5
        # Convolution Block
        block5_1_op = self.relu(self.concat_5_1_skip_connection(block4_2_op) + self.batchnorm5_1_2(self.conv5_1_2(self.relu(self.batchnorm5_1_1(self.conv5_1_1(block4_2_op))))))
        # Identity Block
        block5_2_op = self.relu(block5_1_op + self.batchnorm5_2_2(self.conv5_2_2(self.relu(self.batchnorm5_2_1(self.conv5_2_1(block5_1_op))))))

        ## Final Fully Connected Layer
        avgpool_op = self.avgpool(block5_2_op)
        ## The output tensor of the last convolutional layer is flattened and passed through a fully connected layer to get the final output tensor
        ## avgpool_op.view(avgpool_op.size(0), -1) - The output tensor of the last convolutional layer is flattened using the view function from 1X1X512 to 512 and passed through the fully connected layer
        out = self.out(self.relu(self.fc(avgpool_op.view(avgpool_op.size(0), -1))))         


        ## out_sm = nn.functional.softmax(out, dim=1)  
        # The output of the final fully connected layer is passed through a softmax layer to get the final output tensor
        # The softmax layer is not used here as the loss function used is CrossEntropyLoss which already has softmax layer in it
        # However the softmax should be used when the model is used for inference and not training


        ## If input has batch size of more than 1, then the output of the final fully connected layer will be of size batch_size X num_classes
        ## Checked by printing the output shape of the model in rand_test.py
        ## print(output.shape)
        ## torch.Size([batch_size, num_classes])
        return out
