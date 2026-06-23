import pandas as pd
import cv2
import numpy as np
import os

# MARK: - Local libraries
import extract_radiomics as er
import dataset as ds

# MARK: - SETUP FUNCTIONS
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
    dataset_filepath = ds.getDataSetFilePath()
    data_train = ds.getPreparedDataSet(dataset_filepath, "csv/mass_case_description_train_set.csv")

    for index, row in data_train.iterrows():
        if (index % 3):
            print(row["pathology"])
            # openImageWithMask(row["image file path"], row["ROI mask file path"])

if __name__ == "__main__":
    main()
