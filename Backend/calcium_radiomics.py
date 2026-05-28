import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label


class CalciumRadiomics:
    """
    Extrae las 23 features EXACTAS usadas en entrenamiento.
    """

    def extract_features(self, image_uint8, mask_uint8):

        # Binarizar máscara
        mask = (mask_uint8 > 0.5).astype(np.uint8)

        if mask.sum() == 0:
            raise ValueError("Mask is empty. Cannot extract radiomics.")

        # ------------------------
        # REGION PROPERTIES
        # ------------------------
        labels = label(mask)

        largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
        mask = (labels == largest_label).astype(np.uint8)

        props = regionprops(mask)[0]

        area_px = props.area
        eccentricity = props.eccentricity
        solidity = props.solidity
        perimeter = props.perimeter
        bbox_area = props.bbox_area

        # ------------------------
        # INTENSITY FEATURES
        # ------------------------
        image_norm = image_uint8.astype("float32")/255.0
        lesion_pixels = image_norm[mask == 1]

        int_mean = np.mean(lesion_pixels)
        int_std = np.std(lesion_pixels)
        int_min = np.min(lesion_pixels)
        int_max = np.max(lesion_pixels)
        int_p10 = np.percentile(lesion_pixels, 10)
        int_p90 = np.percentile(lesion_pixels, 90)

        # ------------------------
        # GLCM TEXTURE FEATURES
        # ------------------------
        levels = 256
        img_quant = (image_norm * (levels - 1)).astype(np.uint8)
        img_quant = img_quant * mask
        
        min_row, min_col, max_row, max_col = props.bbox
        img_quant_cropped = img_quant[min_row:max_row, min_col:max_col]

        glcm = graycomatrix(
            img_quant_cropped, 
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=levels,
            symmetric=True,
            normed=True
        )

        def glcm_stat(prop):
            values = graycoprops(glcm, prop).flatten()
            return np.mean(values), np.std(values)

        glcm_contrast_mean, glcm_contrast_std = glcm_stat("contrast")
        glcm_dissimilarity_mean, glcm_dissimilarity_std = glcm_stat("dissimilarity")
        glcm_homogeneity_mean, glcm_homogeneity_std = glcm_stat("homogeneity")
        glcm_energy_mean, glcm_energy_std = glcm_stat("energy")
        glcm_correlation_mean, glcm_correlation_std = glcm_stat("correlation")
        glcm_ASM_mean, glcm_ASM_std = glcm_stat("ASM")

        # ------------------------
        # FEATURE DICT FINAL
        # ------------------------
        return {
            'area_px': area_px,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'perimeter': perimeter,
            'bbox_area': bbox_area,
            'int_mean': int_mean,
            'int_std': int_std,
            'int_min': int_min,
            'int_max': int_max,
            'int_p10': int_p10,
            'int_p90': int_p90,
            'glcm_contrast_mean': glcm_contrast_mean,
            'glcm_contrast_std': glcm_contrast_std,
            'glcm_dissimilarity_mean': glcm_dissimilarity_mean,
            'glcm_dissimilarity_std': glcm_dissimilarity_std,
            'glcm_homogeneity_mean': glcm_homogeneity_mean,
            'glcm_homogeneity_std': glcm_homogeneity_std,
            'glcm_energy_mean': glcm_energy_mean,
            'glcm_energy_std': glcm_energy_std,
            'glcm_correlation_mean': glcm_correlation_mean,
            'glcm_correlation_std': glcm_correlation_std,
            'glcm_ASM_mean': glcm_ASM_mean,
            'glcm_ASM_std': glcm_ASM_std,
        }