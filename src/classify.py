import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def classify(csv_path, classifier):
	# Extrai as features e labels do csv
	df = pd.read_csv(csv_path)
	y = df['target_label']
	X = df.drop(columns=['target_label', 'SeriesInstanceUID', 'patient_id'], errors='ignore')
	
	# Garante que só tenham numeros nas features
	X = X.select_dtypes(include=['number'])
	
	# Separa os dados em dados de treino e de teste
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	
	# Padroniza os dados
	scaler = StandardScaler()
	X_train_std = scaler.fit_transform(X_train)
	X_test_std = scaler.transform(X_test)
	
	classifier.fit(X_train_std, y_train)
	
	# Classifica em benigno e maligno
	y_pred = classifier.predict(X_test_std)
	
	return y_test, y_pred 
