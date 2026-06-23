import kagglehub
import pandas as pd
import os

# Get the absolute path from dataset, if needed download it
def getDataSetFilePath():
    dataset_url = "awsaf49/cbis-ddsm-breast-cancer-image-dataset"
    return kagglehub.dataset_download(dataset_url)

def prepareData(dataset_filepath):
    data = pd.read_csv(dataset_filepath, usecols=[
        'patient_id',
        'pathology',
        'image file path',
        'ROI mask file path'
    ])
    return data

def getDicomTranslationMap(dataset_filepath):
    filepath = os.path.join(dataset_filepath, "csv/dicom_info.csv")
    data = pd.read_csv(filepath)
    cmap = {}

    # Break all lines in conversation unit (ID, DESC) -> JPEG_FILEPATH
    for _, row in data.iterrows():
        uid = str(row["SeriesInstanceUID"]).strip()
        desc = str(row["SeriesDescription"]).strip()
        image_path = "/".join(str(row["image_path"]).strip().split("/")[1:])
        cmap[(uid, desc)] = image_path

    return cmap

def translationImagePath(absolute_dataset_path, cmap, filePath, ftype):
    uid = str(filePath).split("/")[-2]
    return os.path.join(absolute_dataset_path, cmap.get((uid, ftype), filePath))

def getPreparedDataSet(dataset_filepath, target):
    mcd_filepath = os.path.join(dataset_filepath, target)
    dicom_translation_map = getDicomTranslationMap(dataset_filepath)

    data = prepareData(mcd_filepath)

    # Fix the "image filepaths" in dataset to local jpegs
    data['image file path'] = data["image file path"].apply(lambda x: translationImagePath(
        dataset_filepath, dicom_translation_map, x, "full mammogram images"
    ))
    data['ROI mask file path'] = data["ROI mask file path"].apply(lambda x: translationImagePath(
        dataset_filepath, dicom_translation_map, x, "ROI mask images"
    ))

    # Make the pathology to binary (malign is true)
    data["pathology"] = data["pathology"].apply(lambda x: x == 'MALIGNANT')

    return data
