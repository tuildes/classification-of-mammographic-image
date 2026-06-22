import pandas as pd
import cv2
import numpy as np
import os
import kagglehub

# MARK: - SETUP FUNCTIONS
def dataSetFilePath():
    return kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")

# Prepare the dataset and get only relevant columns
def prepareData(dataFilePath):
    data = pd.read_csv(dataFilePath, usecols=[
        'patient_id',
        'pathology',
        'image file path',
        'ROI mask file path'
    ])
    return data

# prepare the translation map (dicom -> jpeg local)
def prepareDicomConversionMap(filePath):
    data = pd.read_csv(filePath)
    cmap = {}

    # Break all lines in conversation unit (ID, DESC) -> JPEG_FILEPATH
    for _, row in data.iterrows():
        uid = str(row["SeriesInstanceUID"]).strip()
        desc = str(row["SeriesDescription"]).strip()
        imagePath = "/".join(str(row["image_path"]).strip().split("/")[1:])
        cmap[(uid, desc)] = imagePath

    return cmap

# Translate using conversation map
def translationImagePath(cmap, filePath, ftype):
    uid = str(filePath).split("/")[-2]
    return cmap.get((uid, ftype), filePath)

# Only opens the image in horizontal row to DEBUG
def openImageWithMask(imageFilePath, maskFilePath):
    image = cv2.imread(imageFilePath)
    mask = cv2.imread(maskFilePath)

    h, w = image.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    row = np.concatenate((image, mask), axis=1)

    cv2.imshow('sei la', row)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# MARK: - MAIN PROGRAM
def main():
    dataset_path = dataSetFilePath()
    mcd_train_filepath = os.path.join(dataset_path, "csv/mass_case_description_train_set.csv")
    # mcd_tests_filepath = os.path.join(dataset_path, "csv/mass_case_description_test_set.csv")

    dicom_translation_filepath = os.path.join(dataset_path, "csv/dicom_info.csv")
    dicom_translation_map = prepareDicomConversionMap(dicom_translation_filepath)

    data_train = prepareData(mcd_train_filepath)
    # data_tests = prepareData(mcd_train_filepath)

    print(f"DEBUG: Dataset path: {dataset_path}")
    print(f"DEBUG: Train Dataset size: {len(data_train)}")

    for index, row in data_train.iterrows():
        if (index % 3):
            imageFilePath = os.path.join(dataset_path, translationImagePath(dicom_translation_map, row["image file path"], "full mammogram images"))
            maskFilePath = os.path.join(dataset_path, translationImagePath(dicom_translation_map, row["ROI mask file path"], "ROI mask images"))
            openImageWithMask(imageFilePath, maskFilePath)

if __name__ == "__main__":
    main()
