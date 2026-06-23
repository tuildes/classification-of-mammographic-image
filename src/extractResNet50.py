import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configuração de sementes para reprodutibilidade
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executando o pipeline em: {device}")

# Trocar para ficar generico
INPUT_DIR = '~/.cache/kagglehub/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/versions/1/csv/'

print("Carregando os arquivos de metadados CSV...")
# Carrega a ficha clínica das lesões (contém os diagnósticos/biópsias)
mass_train = pd.read_csv(os.path.join(INPUT_DIR, 'mass_case_description_train_set.csv'))
dicom_info = pd.read_csv(os.path.join(INPUT_DIR, 'dicom_info.csv'))

mass_train = mass_train[mass_train['pathology'].isin(['BENIGN', 'MALIGNANT'])].reset_index(drop=True)
# Pega o ID da img que fica no nome dela
mass_train['SeriesInstanceUID'] = mass_train['cropped image file path'].apply(
    lambda x: x.split('/')[-2] if pd.notna(x) else None
)

roi_dicom = dicom_info[dicom_info['SeriesDescription'] == 'cropped images'].copy()

#junta o dataset com o dignostico
dataset_final = pd.merge(mass_train, roi_dicom[['SeriesInstanceUID', 'image_path']], on='SeriesInstanceUID', how='inner')

df_total = dataset_final.reset_index(drop=True)

print(f"\n[SUCESSO] Total de ROIs oficiais integradas para o pipeline: {len(df_total)}")
print(df_total['pathology'].value_counts())

class CBISDDSMRapidoDataset(Dataset):
	def __init__(self, dataframe, transform=None):
		self.df = dataframe
		self.transform = transform
		self.label_mapping = {'BENIGN': 0, 'MALIGNANT': 1}

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		
		BASE_DIR = os.path.expanduser('~/.cache/kagglehub/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/versions/1/')
		
		rel_path = row['image_path'].replace("CBIS-DDSM/", "")
		full_path = os.path.join(BASE_DIR, rel_path)

		base_path, _ = os.path.splitext(rel_path)
		for ext in ['.jpg', '.png', '.jpeg']:
			if os.path.exists(base_path + ext):
				full_path = base_path + ext
				break
		
		img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)	
		if img is not None:
			img = np.ones((240, 240), dtype=np.uint8) * 128

			img_pil = Image.fromarray(img).convert('RGB')
			label = self.label_mapping[row['pathology']]

			if self.transform:
				img_tensor = self.transform(img_pil)

			return img_tensor, label

transform_resnet = transforms.Compose([
	transforms.Resize((240, 240)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = CBISDDSMRapidoDataset(df_total, transform=transform_resnet)
loader = DataLoader(full_dataset, batch_size=16, shuffle=False)

print("Inicializando a ResNet50 pré-treinada...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

model.fc = nn.Identity()
model = model.to(device)
model.eval()

X_features = []
y_labels = []

print("Extraindo vetores de características das imagens de nódulos...")
with torch.no_grad():
    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        X_features.append(outputs.cpu().numpy())
        y_labels.extend(labels.numpy())

X = np.concatenate(X_features, axis=0)
y = np.array(y_labels)

# Salva o vetor de caracteristicas para não precisar rodar de novo
np.save('X_features.npy', X)
np.save('y_labels.npy', y)

print(f"\nExtração Concluída! Matriz de Características: {X.shape} (Imagens, Dimensões)")
