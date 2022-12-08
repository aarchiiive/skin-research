import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import cv2
import copy
import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def save_params(save_path,
                model_name,
                num_epochs=25,
                input_size=600,
                num_classes=4,
                optimizer="Adam",
                learning_rate=0.0001,
                weight_decay=0.0005,
                drop_rate=0.2,
                batch_size=8,
                num_workers=10):
    
    df = pd.DataFrame.from_dict({
        "model_name" : [model_name],
        "num_epochs" : [num_epochs],
        "input_size" : [input_size],
        "num_classes" : [num_classes],
        "optimizer" : [optimizer],
        "learning_rate" : [learning_rate],
        "weight_decay" : [weight_decay],
        "drop_rate" : [drop_rate],
        "batch_size" : [batch_size],
        "num_workers" : [num_workers]
    }, orient="columns")
    
    df.to_csv(os.path.join(save_path, "hyperparameters.csv"))


def dataloader(train_path,
               test_path,
               label_path,
               input_size,
               batch_size, 
               num_workers):
    
    train_label_path, test_label_path = label_path
    
    # augentations
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.1)),
        transforms.RandomResizedCrop(size=(input_size, input_size), scale=(0.75, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomErasing(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = CustomImageDataset(train_label_path, train_path, train_transform)
    test_data = CustomImageDataset(test_label_path, test_path, test_transform)

    print("Initializing Datasets and Dataloaders...")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_dataloader, test_dataloader

def get_testloader(image_path, 
                   label_path, 
                   input_size,
                   batch_size=32):
    
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_data = CustomImageDataset(label_path, image_path, test_transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return test_dataloader

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir=None, transform=None, target_transform=None, phase=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.phase = phase

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 1]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
            
        image = torch.FloatTensor(image)
        label = self.img_labels.iloc[idx, 3]

        return image, label
    
    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    
    
if __name__ == "__main__":
    pass