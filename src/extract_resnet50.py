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

import dataset as ds

torch.manual_seed(42)
np.random.seed(42)

# MARK: - Dataset

class MammographyDataset(Dataset):
    LABEL_MAPPING = {'BENIGN': 0, 'MALIGNANT': 1}

    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.ones((240, 240), dtype=np.uint8) * 128

        img_pil = Image.fromarray(img).convert('RGB')
        label = self.LABEL_MAPPING[row['pathology']]

        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, label


# MARK: - Setup

def setupDevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando o pipeline em: {device}")
    return device


# MARK: - Loaders

def loadDataFrame(dataset_filepath, csv_target):
    data = pd.read_csv(os.path.join(dataset_filepath, csv_target))
    data = data[data['pathology'].isin(['BENIGN', 'MALIGNANT'])].reset_index(drop=True)

    dicom_map = ds.getDicomTranslationMap(dataset_filepath)

    data['SeriesInstanceUID'] = data['cropped image file path'].apply(
        lambda x: x.split('/')[-2] if pd.notna(x) else None
    )
    data['image_path'] = data['cropped image file path'].apply(
        lambda x: ds.translationImagePath(dataset_filepath, dicom_map, x, 'cropped images') if pd.notna(x) else None
    )

    return data.dropna(subset=['image_path']).reset_index(drop=True)


# MARK: - Preparation

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


# MARK: - Feature Extraction

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


def buildFeaturesDataFrame(X, y, series_instance_uids):
    df_features = pd.DataFrame(X, columns=[f'resnet_{i}' for i in range(X.shape[1])])
    df_features['SeriesInstanceUID'] = series_instance_uids
    df_features['target_label'] = y
    return df_features
