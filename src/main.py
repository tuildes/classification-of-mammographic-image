import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# MARK: - Local libraries
import extract_radiomics as exrad
import extract_resnet50 as exres
import dataset as ds
import classify as cls

def print_results(y_test, y_pred, featurer_classifier):
    print(f"\n--- RELATÓRIO DE CLASSIFICAÇÃO {featurer_classifier} OTIMIZADO ---")
    print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))
    print(f"Nova Acurácia Global Atingida: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

# MARK: - MAIN PROGRAM

def main():
    results_folder = "./results"
    radiomics_train_csv = os.path.join(results_folder, "radiomics_features_train.csv")
    resnet_train_csv = os.path.join(results_folder, "resnet_features_train.csv")
    merged_train_csv = os.path.join(results_folder, "merged_features_train.csv")
    radiomics_test_csv = os.path.join(results_folder, "radiomics_features_test.csv")
    resnet_test_csv = os.path.join(results_folder, "resnet_features_test.csv")
    merged_test_csv = os.path.join(results_folder, "merged_features_test.csv")

    csv_target = "csv/mass_case_description_train_set.csv"
    csv_test_target = "csv/mass_case_description_test_set.csv"

    dataset_filepath = ds.getDataSetFilePath()
    data_train = ds.getPreparedDataSet(dataset_filepath, csv_target)
    data_test = ds.getPreparedDataSet(dataset_filepath, csv_test_target)
    print(data_train['pathology'].value_counts(), "\n")

    # Create the results folder if needed
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # MARK: - ResNet50 analyzes
    # Only execute if the resnet_features_train.csv doesn't exists
    if not os.path.isfile(resnet_train_csv):
        device = exres.setupDevice()
        resnet_model = exres.prepareModel(device)

        X, y = exres.extractFeatures(resnet_model, data_train, device)
        df_resnet_features = exres.buildFeaturesDataFrame(X, y, data_train['pseudo_index'].values, data_train['pathology'].values)
        df_resnet_features.to_csv(resnet_train_csv, index=False)
        print(f"\nResNet50: Extract finished! Features: {X.shape} (Image, Dimension)")

        del X
        del y
        del df_resnet_features
        gc.collect()
    else:
        print(f"ResNet50 training already exists - skipping training ({resnet_train_csv})")

    if not os.path.isfile(resnet_test_csv):
        device = exres.setupDevice()
        resnet_model = exres.prepareModel(device)

        X, y = exres.extractFeatures(resnet_model, data_test, device)
        df_resnet_features = exres.buildFeaturesDataFrame(X, y, data_test['pseudo_index'].values, data_test['pathology'].values)
        df_resnet_features.to_csv(resnet_test_csv, index=False)
        print(f"\nResNet50: Extract finished! Features: {X.shape} (Image, Dimension)")

        del X
        del y
        del df_resnet_features
        gc.collect()
    else:
        print(f"ResNet50 test already exists - skipping training ({resnet_train_csv})")

    # MARK: - Pyradiomics analyzes
    # Only execute if the radiomics_features_train.csv doesn't exists
    if not os.path.isfile(radiomics_train_csv):
        exrad.setupLoggin()
        pyrad_extractor = exrad.prepareExtractor()
        pyrad_features = []

        for index, row in data_train.iterrows():
            image, mask = exrad.loadImageAndMask(row["image file path"],
                                                 row["ROI mask file path"])
            features = exrad.calculateFeatures(pyrad_extractor, image, mask)
            features['pseudo_index'] = row['pseudo_index']
            features['target_label'] = row['target_label']
            features['pathology'] = row['pathology']
            pyrad_features.append(features)
            print(f"PYRADIOMICS: Extracted {len(features)} Features - {(index + 1)}/{len(data_train)}")

        dataframe_pyrad_features = pd.DataFrame(pyrad_features)
        dataframe_pyrad_features.to_csv(radiomics_train_csv, index=False)

        del pyrad_features
        del dataframe_pyrad_features
        gc.collect()
    else:
        print(f"Radiomics training already exists - skipping training ({radiomics_train_csv})")

    if not os.path.isfile(radiomics_test_csv):
        exrad.setupLoggin()
        pyrad_extractor = exrad.prepareExtractor()
        pyrad_features = []

        for index, row in data_test.iterrows():
            image, mask = exrad.loadImageAndMask(row["image file path"],
                                                 row["ROI mask file path"])
            features = exrad.calculateFeatures(pyrad_extractor, image, mask)
            features['pseudo_index'] = row['pseudo_index']
            features['target_label'] = row['target_label']
            features['pathology'] = row['pathology']
            pyrad_features.append(features)
            print(f"PYRADIOMICS: Extracted {len(features)} Features - {(index + 1)}/{len(data_test)}")

        dataframe_pyrad_features = pd.DataFrame(pyrad_features)
        dataframe_pyrad_features.to_csv(radiomics_test_csv, index=False)

        del pyrad_features
        del dataframe_pyrad_features
        gc.collect()
    else:
        print(f"Radiomics training already exists - skipping training ({radiomics_train_csv})")

    # MARK: - Read all csv (features) and merge
    dataset_train_radiom = pd.read_csv(radiomics_train_csv)
    dataset_train_resnet = pd.read_csv(resnet_train_csv)
    dataset_train_merged = pd.merge(dataset_train_radiom,
                                    dataset_train_resnet,
                                    on=['pseudo_index', 'pathology', 'target_label'])
    dataset_train_merged.to_csv(merged_test_csv, index=False)
    dataset_test_radiom = pd.read_csv(radiomics_test_csv)
    dataset_test_resnet = pd.read_csv(resnet_test_csv)
    dataset_test_merged = pd.merge(dataset_test_radiom,
                                    dataset_test_resnet,
                                    on=['pseudo_index', 'pathology', 'target_label'])

    dataset_test_merged.to_csv(merged_test_csv, index=False)
    dataset_train_merged.to_csv(merged_train_csv, index=False)

    # MARK: - Prepare dataset to training/testing
    datasets = {
        "radiomics": (dataset_train_radiom, dataset_test_radiom),
        "resnet50":  (dataset_train_resnet, dataset_test_resnet),
        "merged":    (dataset_train_merged, dataset_test_merged),
    }
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)
    }

    for dataset_name, (df_train, df_test) in datasets.items():
        for model_name, classifier in models.items():
            X_train = df_train.drop(columns=['pseudo_index', 'pathology', 'target_label'])
            y_train = df_train['target_label']
            X_test = df_test.drop(columns=['pseudo_index', 'pathology', 'target_label'])
            y_test = df_test['target_label']

            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(X_train)
            X_test_std = scaler.transform(X_test)

            classifier.fit(X_train_std, y_train)
            y_pred = classifier.predict(X_test_std)

            # Metrics
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = (tn / (tn + fp))
            sensitivity = (tp / (tp + fn))
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['BENIGN', 'MALIGNANT'])

            loc_txt = os.path.join(results_folder, f'report_{dataset_name}_{model_name}.txt')
            loc = os.path.join(results_folder, f'confusion_matrix_{dataset_name}_{model_name}.png')

            with open(loc_txt, 'w', encoding='utf-8') as file:
                file.write(f"Results of testing: {dataset_name} with model: {model_name}\n")
                file.write(f"Acurácia:       {accuracy:.4f}\n")
                file.write(f"F1-Score:       {f1:.4f}\n")
                file.write(f"Sensibilidade:  {sensitivity:.4f}\n")
                file.write(f"Especificidade: {specificity:.4f}\n\n")
                file.write("RELATÓRIO COMPLETO (SCIKIT-LEARN):\n")
                file.write(report)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'MALIGNANT'], yticklabels=['BENIGN', 'MALIGNANT'])
            plt.title(f'Confusion Matrix - {dataset_name} - {model_name}')
            plt.ylabel('True')
            plt.xlabel('Prevision Model')
            plt.savefig(loc)
            plt.clf()

            print(f"\nAnalyzes of {dataset_name} with model {model_name} finished")
            print(f"File created: {loc}")
            print(f"File created: {loc_txt}")

if __name__ == "__main__":
    main()
