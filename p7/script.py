import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import medmnist
from medmnist import INFO
import os
from MyNetworkTemplate import MyNetwork
from torch.utils.tensorboard import SummaryWriter

nChannels = INFO['dermamnist']['n_channels']
nClasses = len(INFO['dermamnist']['label'])
DataClass = medmnist.DermaMNIST

# Transforming images to Torch Tensor and Normalizing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1])
])

## Obtaining the training, validation and test datasets
trainingData = DataClass(split='train', transform=data_transform,  download= os.path.exists('./dermamnist.npz'), root = './')
validationData = DataClass(split='val', transform=data_transform, download= os.path.exists('./dermamnist.npz'), root = './')
testData = DataClass(split='test', transform=data_transform,  download= os.path.exists('./dermamnist.npz'), root = './')

## This code will show a preview of the images
#a = trainingData.montage(length=5)
#plt.imshow(a)
#plt.show()

## Configuring the batch size and creating data loaders
batchSize = 200
trainLoader = data.DataLoader(dataset=trainingData, batch_size=batchSize, shuffle=True)
validationLoader = data.DataLoader(dataset=validationData, batch_size=batchSize, shuffle=True)
#testLoader = data.DataLoader(dataset=testData, batch_size=batchSize, shuffle=False)

model = MyNetwork(nChannels, nClasses, nEpochs=160, learningRate=1E-4)
model.trainModel(trainLoader, validationLoader,'log/MyNetwork')

#model.save('full_model.pth')


model.save('full_model1.pth')