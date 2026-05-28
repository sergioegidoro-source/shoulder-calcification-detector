import cv2
import numpy as np


class CalciumPipeline:
    """
    Pipeline completo:
    U-Net → Radiomics → Clasificador
    """

    def __init__(self, unet_model, radiomics_extractor, classifier):
        self.unet = unet_model
        self.radiomics = radiomics_extractor
        self.classifier = classifier

    def run(self, img_uint8):
        """
        img_uint8: imagen grayscale uint8
        """

        # -------------------------
        # 1️⃣ U-Net
        # -------------------------
        mask_512 = self.unet.predict_mask_from_array(img_uint8)

        if mask_512 is None:
            return {
                "prediction": None,
                "mask": np.zeros((img_uint8.shape[0], img_uint8.shape[1]), dtype=np.uint8),
            }

        mask_resized = cv2.resize(
            mask_512,
            (img_uint8.shape[1], img_uint8.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Si máscara vacía, no clasificar
        if np.sum(mask_resized) < 50:
            return {
                "prediction": None,
                "mask": mask_resized
            }

        # -------------------------
        # 2️⃣ Radiomics
        # -------------------------
        features = self.radiomics.extract_features(
            img_uint8,
            mask_resized
        )

        # -------------------------
        # 3️⃣ Clasificador
        # -------------------------
        result = self.classifier.predict(features)

        return {
            "prediction": result["prediction"] + " (prediction)",
            "mask": mask_resized
        }
