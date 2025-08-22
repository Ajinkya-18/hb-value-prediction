from utils import ConjunctivaDataset, train_transforms, val_transforms, HgbCNN
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import cv2

train_dataset = ConjunctivaDataset(image_dir='../data/Conjunctiva/Regression/regression_dataset/Box_Conjunctiva_Images', 
                                   metadata_csv_path='../data/Conjunctiva/Regression/regression_dataset/regression_complete_metadata.csv', 
                                   transforms=train_transforms)

val_dataset = ConjunctivaDataset(image_dir='../data/Conjunctiva/Regression/LMH_patient_data_mk2/HB_Cropped_Conjunctiva_Validation_Set', 
                                 metadata_csv_path='../data/Conjunctiva/Regression/LMH_patient_data_mk2/PPG_HB_Data_with_images.csv', 
                                 transforms=val_transforms)

hb_targets = train_dataset.metadata['Hgb'].values

from torch.utils.data import WeightedRandomSampler

bins = np.digitize(hb_targets, bins=np.linspace(min(hb_targets), max(hb_targets), 10))
class_sample_count = np.bincount(bins)
weights = 1. / class_sample_count[bins]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


train_loader = DataLoader(dataset=train_dataset, batch_size=24, sampler=sampler, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=24, shuffle=False)



hgbcnn = HgbCNN()

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in resnet18.parameters():
    param.requires_grad = False

in_features = resnet18.fc.in_features

resnet18.fc = nn.Sequential(
    nn.Linear(in_features, out_features=256), 
    nn.ReLU(),  
    nn.Linear(256, 128),
    nn.ReLU(), 
    nn.Linear(128, 64), 
    nn.ReLU(), 
    nn.Linear(64, 1)
)

convnext = models.convnext_tiny(weights='IMAGENET1K_V1')

for param in convnext.parameters():
    param.requires_grad = False

    
convnext.classifier[2] = nn.Sequential(
    nn.Linear(768, 256), 
    nn.ReLU(), 
    nn.Linear(256, 128), 
    nn.ReLU(), 
    nn.Linear(128, 64), 
    nn.ReLU(),
    nn.Linear(64, 1)
)


# train_model(hgbcnn, 'HgbCNN', train_loader, val_loader, epochs=30)

# train_model(resnet18, 'ResNet18', train_loader, val_loader, epochs=30)

# train_model(convnext, 'ConvNeXt', train_loader, val_loader, epochs=30)


