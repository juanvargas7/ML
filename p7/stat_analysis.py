import torch
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

testData = DataClass(split='test', transform=data_transform,  download= os.path.exists('./dermamnist.npz'), root = './')
batchSize = 100


model = MyNetwork(nChannels, nClasses, nEpochs=160)
model.load_state_dict(torch.load('full_model.pth', map_location=torch.device('cuda')))





# Create a data loader for the test dataset
testLoader = data.DataLoader(dataset=testData, batch_size=batchSize, shuffle=True)
# Assuming you have imported torch at the beginning of your script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Ensure the model is on the right device


all_labels = []
all_predictions = []

with torch.no_grad():  # No need to track gradients for evaluation
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

from sklearn.metrics import accuracy_score

overall_accuracy = accuracy_score(all_labels, all_predictions)
print(f'Overall Accuracy: {overall_accuracy:.2f}')

from sklearn.metrics import classification_report, confusion_matrix

# Dermamnist class names
class_names = [
    'actinic keratoses and intraepithelial carcinoma', 
    'basal cell carcinoma', 
    'benign keratosis-like lesions', 
    'dermatofibroma', 
    'melanoma', 
    'melanocytic nevi', 
    'vascular lesions'
]

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Compute sensitivity, specificity, and other metrics
report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
print(classification_report(all_labels, all_predictions, target_names=class_names))

# Calculate specificity for each class
specificity = []
for i in range(len(class_names)):
    true_negatives = sum([conf_matrix[j][j] for j in range(len(class_names)) if j != i])
    false_positives = sum([conf_matrix[j][i] for j in range(len(class_names)) if j != i])
    spec = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    specificity.append(spec)
    print(f'Specificity of class {class_names[i]}: {spec:.2f}')




