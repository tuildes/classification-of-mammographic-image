import pandas as pd
import os
import gc

# MARK: - Local libraries
import extract_radiomics as exrad
import extract_resnet50 as exres
import dataset as ds


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
        df_resnet_features.to_csv(os.path.join(results_folder, resnet_train_csv), index=False)
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
    # MARK: - Random Forest


if __name__ == "__main__":
    main()
