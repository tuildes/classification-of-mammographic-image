import pandas as pd
import cv2
import numpy as np
import os
import kagglehub

# MARK: - SETUP FUNCTIONS
def dataSetFilePath():
    return kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")

def prepareData(dataFilePath):
    data = pd.read_csv(dataFilePath, usecols=[
        'patient_id',
        'pathology',
        'image file path',
        'ROI mask file path'
    ])

    return data

def openImageWithMask(imageFilePath, maskFilePath):
    print(imageFilePath)
    print(maskFilePath)

# MARK: - MAIN PROGRAM

def main():
    dataset_path = dataSetFilePath()
    mcd_train_filepath = os.path.join(dataset_path, "csv/mass_case_description_train_set.csv")
    dicom_translation_filepath = os.path.join(dataset_path, "csv/dicom_info.csv")
    # mcd_tests_filepath = os.path.join(dataset_path, "csv/mass_case_description_test_set.csv")

    data_train = prepareData(mcd_train_filepath)
    # data_tests = prepareData(mcd_train_filepath)

    print(f"DEBUG: Dataset path: {dataset_path}")
    print(f"DEBUG: Train Dataset size: {len(data_train)}")

    for index, row in data_train.iterrows():
        if (index == 0):
            # imageFilePath = os.path.join(dataset_path, 'jpeg', row['ROI mask file path'])
            openImageWithMask(row['image file path'], row['ROI mask file path'])

if __name__ == "__main__":
    main()
