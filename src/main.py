import pandas as pd
import cv2
import numpy as np
import os

# MARK: - Local libraries
import extract_radiomics as exrad
import dataset as ds

# MARK: - MAIN PROGRAM
def main():
    results_folder = "./results"
    radiomics_train_csv = "radiomics_train.csv"
    resnet_train_csv = "resnet_train_csv"

    dataset_filepath = ds.getDataSetFilePath()
    data_train = ds.getPreparedDataSet(dataset_filepath, "csv/mass_case_description_train_set.csv")

    # DEBUG Info
    print(data_train['pathology'].value_counts())

    # MARK: - ResNet50 analyzes

    # MARK: - Pyradiomics analyzes
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
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    dataframe_pyrad_features.to_csv(os.path.join(results_folder, radiomics_train_csv), index=False)

    # MARK: - Read all csv (features) and merge


    # MARK: - SVM
    # MARK: - Random Forest

if __name__ == "__main__":
    main()
