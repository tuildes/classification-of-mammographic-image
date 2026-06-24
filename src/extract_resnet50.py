import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)

# MARK: - Dataset
class MammographyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(row['cropped image file path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.ones((224, 224), dtype=np.uint8) * 128

        img_pil = Image.fromarray(img).convert('RGB')
        label = self.row['target_label']

        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, label


# MARK: - Setup
def setupDevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando o pipeline em: {device}")
    return device

def prepareModel(device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def prepareTransform():
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extractFeatures(model, dataframe, device, batch_size=16):
    transform = prepareTransform()
    dataset = MammographyDataset(dataframe, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    X_features = []
    y_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Extraindo features ResNet50"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            X_features.append(outputs.cpu().numpy())
            y_labels.extend(labels.numpy())

    X = np.concatenate(X_features, axis=0)
    y = np.array(y_labels)
    return X, y


def buildFeaturesDataFrame(X, y, pseudo_index, pathology):
    df_features = pd.DataFrame(X, columns=[f'resnet_{i}' for i in range(X.shape[1])])
    df_features['pseudo_index'] = pseudo_index
    df_features['pathology'] = pathology
    df_features['target_label'] = y
    return df_features
