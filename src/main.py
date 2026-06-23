import pandas as pd
import os

# MARK: - Local libraries
import extract_radiomics as exrad
import extract_resnet50 as exres
import dataset as ds


# MARK: - MAIN PROGRAM

def main():
    results_folder = "./results"
    radiomics_train_csv = "radiomics_train.csv"
    resnet_train_csv = "resnet_features_train.csv"
    csv_target = "csv/mass_case_description_train_set.csv"

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    dataset_filepath = ds.getDataSetFilePath()

    # MARK: - ResNet50 analyzes
    device = exres.setupDevice()
    resnet_model = exres.prepareModel(device)

    resnet_dataframe = exres.loadDataFrame(dataset_filepath, csv_target)
    print(f"\n[ResNet50] Total de ROIs carregadas: {len(resnet_dataframe)}")
    print(resnet_dataframe['pathology'].value_counts())

    X, y = exres.extractFeatures(resnet_model, resnet_dataframe, device)
    df_resnet_features = exres.buildFeaturesDataFrame(X, y, resnet_dataframe['SeriesInstanceUID'].values)
    df_resnet_features.to_csv(os.path.join(results_folder, resnet_train_csv), index=False)
    print(f"\n[ResNet50] Extração concluída! Matriz de características: {X.shape} (Imagens, Dimensões)")

    # MARK: - Pyradiomics analyzes
    data_train = ds.getPreparedDataSet(dataset_filepath, csv_target)
    print(f"\n[Pyradiomics] Total de amostras: {len(data_train)}")
    print(data_train['pathology'].value_counts())

    exrad.setupLoggin()
    pyrad_extractor = exrad.prepareExtractor()
    pyrad_features = []
    for index, row in data_train.iterrows():
        image, mask = exrad.loadImageAndMask(row["image file path"], row["ROI mask file path"])
        features = exrad.calculateFeatures(pyrad_extractor, image, mask)
        features['patient_id'] = row['patient_id']
        pyrad_features.append(features)
        print(f"PYRADIOMICS: Extracted {len(features)} Features - {(index + 1)}/{len(data_train)}")

    df_pyrad_features = pd.DataFrame(pyrad_features)
    df_pyrad_features.to_csv(os.path.join(results_folder, radiomics_train_csv), index=False)

    # MARK: - Read all csv (features) and merge

    # MARK: - SVM
    # MARK: - Random Forest


if __name__ == "__main__":
    main()
