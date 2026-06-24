import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import numpy as np
import logging
import cv2

import preprocessing as pp

def setupLoggin():
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

def loadImageAndMask(image_file_path, mask_file_path):
    image = sitk.ReadImage(image_file_path)
    mask = sitk.ReadImage(mask_file_path)

    img_array = sitk.GetArrayFromImage(image)
    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_array_preprocessed = pp.preprocessing_filter(img_array)

    new = sitk.GetImageFromArray(img_array_preprocessed)
    new.CopyInformation(image)
    image = new

    if image.GetSize() != mask.GetSize():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask = resampler.Execute(mask)
    else:
        mask.CopyInformation(image)

    return image, mask

def prepareExtractor():
    settings = {}
    settings["binWidth"] = 25
    settings["interpolator"] = sitk.sitkBSpline
    settings["resampledPixelSpacing"] = None
    settings["force2D"] = True
    settings["normalize"] = True
    settings["normalizeScale"] = 100

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    return extractor

def calculateFeatures(extractor, image, mask):
    f = extractor.execute(image, mask)
    cf = {
        key: float(value)
        for key, value in f.items()
        if not key.startswith('diagnostics_')
    }
    return cf
