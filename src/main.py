import pandas as pd
import os
import gc
from sklearn.metrics import classification_report, accuracy_score

# MARK: - Local libraries
import extract_radiomics as exrad
import extract_resnet50 as exres
import dataset as ds
import svm_classifier as svm

def print_results(y_test, y_pred, classifier):
    print("\n--- RELATÓRIO DE CLASSIFICAÇÃO {classifier} OTIMIZADO (SVM RBF) ---")
    print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))
    print(f"Nova Acurácia Global Atingida: {accuracy_score(y_test, y_pred)*100:.2f}%")	
    

# MARK: - MAIN PROGRAM

def main():
    results_folder = "./results"
    radiomics_train_csv = os.path.join(results_folder, "radiomics_features_train.csv")
    resnet_train_csv = os.path.join(results_folder, "resnet_features_train.csv")
    csv_target = "csv/mass_case_description_train_set.csv"

    dataset_filepath = ds.getDataSetFilePath()
    data_train = ds.getPreparedDataSet(dataset_filepath, "csv/mass_case_description_train_set.csv")
    print(data_train['pathology'].value_counts(), "\n")

    # Create the results folder if needed
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # MARK: - ResNet50 analyzes
    # Only execute if the resnet_features_train.csv doesn't exists
    if not os.path.isfile(resnet_train_csv):
        device = exres.setupDevice()
        resnet_model = exres.prepareModel(device)

        # TODO: - Arruamr isso aqui, nao faz sentido der duas funcoes diferentes para a mesma coisa
        resnet_dataframe = exres.loadDataFrame(dataset_filepath, csv_target)

        X, y = exres.extractFeatures(resnet_model, resnet_dataframe, device)
        df_resnet_features = exres.buildFeaturesDataFrame(X, y, resnet_dataframe['SeriesInstanceUID'].values)
        df_resnet_features.to_csv(resnet_train_csv, index=False)
        #df_resnet_features.to_csv(os.path.join(results_folder, resnet_train_csv), index=False)
        print(f"\nResNet50: Extract finished! Features: {X.shape} (Image, Dimension)")
    else:
        print(f"ResNet50 training already exists - skipping training ({resnet_train_csv})")

    # MARK: - Pyradiomics analyzes
    # Only execute if the radiomics_features_train.csv doesn't exists
    if not os.path.isfile(radiomics_train_csv):
        data_train = ds.getPreparedDataSet(dataset_filepath, csv_target)

        exrad.setupLoggin()
        pyrad_extractor = exrad.prepareExtractor()
        pyrad_features = []

        for index, row in data_train.iterrows():
            image, mask = exrad.loadImageAndMask(row["image file path"], row["ROI mask file path"])
            features = exrad.calculateFeatures(pyrad_extractor, image, mask)
            features['patient_id'] = row['patient_id']
            features['target_label'] = 1 if row['pathology'] else 0
            
            pyrad_features.append(features)
            print(f"PYRADIOMICS: Extracted {len(features)} Features - {(index + 1)}/{len(data_train)}")

        dataframe_pyrad_features = pd.DataFrame(pyrad_features)
        dataframe_pyrad_features.to_csv(radiomics_train_csv, index=False)

        del pyrad_features
        del dataframe_pyrad_features
        gc.collect()
    else:
        print(f"Radiomics training already exists - skipping training ({radiomics_train_csv})")

    # MARK: - Read all csv (features) and merge

    # MARK: - SVM
    y_test_resnet, y_pred_resnet = svm.svm_classifier(resnet_train_csv)
    y_test_radiomics, y_pred_radiomics = svm.svm_classifier(radiomics_train_csv)
    
    # MARK: - Print Results
    print_results(y_test_resnet, y_pred_resnet, "RESNET")
    print_results(y_test_radiomics, y_pred_radiomics, "RADIOMICS")  
         
    # MARK: - Random Forest


if __name__ == "__main__":
    main()
