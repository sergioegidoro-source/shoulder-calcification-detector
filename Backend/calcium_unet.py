import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from config import MODELPTH_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CalciumUNet:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODELPTH_PATH):
            print("Calcium U-Net weights not found.")
            return

        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b2",
            encoder_weights=None,
            in_channels=1,
            classes=1
        )

        state_dict = torch.load(MODELPTH_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self.model = model
        print("Calcium U-Net loaded successfully.")

    # --------------------------------------------------------
    # DICOM LOADER
    # --------------------------------------------------------
    def _load_dicom(self, file_stream):
        file_stream.seek(0)
        ds = pydicom.dcmread(file_stream)

        # Aplicar VOI LUT si existe
        if "WindowWidth" in ds and "WindowCenter" in ds:
            img = apply_voi_lut(ds.pixel_array, ds)
        else:
            img = ds.pixel_array.astype(np.float32)

        # Invertir si MONOCHROME1
        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = np.max(img) - img

        # Normalización min-max
        img = img - np.min(img)
        if np.max(img) != 0:
            img = img / np.max(img)

        img_uint8 = (img * 255).astype(np.uint8)

        return img_uint8

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
    def predict_mask_from_dicom(self, file_stream, threshold: float = 0.5):
        if self.model is None:
            return None

        img_uint8 = self._load_dicom(file_stream)

        return self.predict_mask_from_array(img_uint8, threshold)

    def predict_mask_from_array(self, img_uint8: np.ndarray, threshold: float = 0.5):
        if self.model is None:
            return None

        # Resize a 512x512
        img_resized = cv2.resize(img_uint8, (512, 512), interpolation=cv2.INTER_AREA)

        # Normalizar
        img_norm = img_resized.astype("float32") / 255.0

        # Convertir a tensor [1,1,H,W]
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.sigmoid(output)[0, 0].cpu().numpy()

        mask_bin = (prob > threshold).astype(np.uint8) * 255

        return mask_bin