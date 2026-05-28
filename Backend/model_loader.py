import os
import gdown
import tf_keras
import tensorflow as tf
import torch
import segmentation_models_pytorch as smp
import joblib
from config import *
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from ultralytics import YOLO

# ============================================================
# DOWNLOAD RESOURCES
# ============================================================
def download_file(path: str | None, drive_id: str | None, name: str) -> None:
    if not path or not drive_id:
        return
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    if os.path.exists(path):
        try: os.remove(path)
        except: pass
    print(f"Downloading {name}...")
    try:
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, path, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"Download failed for {name}: {e}")

# Descargar todo al importar
download_file(CNN_PATH, CNN_DRIVE_ID, "CNN Model")
download_file(ML_PATH, ML_DRIVE_ID, "Hybrid ML")
download_file(FEAT_PATH, FEAT_DRIVE_ID, "Features")
download_file(UNET_PATH, UNET_DRIVE_ID, "U-Net")
download_file(CLASS300_PATH, CLASS300_DRIVE_ID, "Classifier 300")
download_file(RE_PT_PATH, RE_PT_DRIVE_ID, "YOLO RE")
download_file(RI_PT_PATH, RI_PT_DRIVE_ID, "YOLO RI")
download_file(MODELJOBLIB_PATH, MODELJOBLIB_DRIVE_ID, "LR Model")
download_file(MODELPTH_PATH, MODELPTH_DRIVE_ID, "UNet Model")

# ============================================================
# LOAD MODELS
# ============================================================
cnn_model = None
feature_extractor = None
combined_cnn = None   # CNN prob + GMP features en un solo forward pass
ml_model = None
class_300_model = None
unet_model = None
yolo_re = None
yolo_ri = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    if CNN_PATH and os.path.exists(CNN_PATH):
        cnn_model = tf_keras.models.load_model(CNN_PATH)
        gmp_layer = cnn_model.get_layer("global_max_pooling2d")
        feature_extractor = tf_keras.Model(
            inputs=cnn_model.input,
            outputs=gmp_layer.output,
        )
        # Modelo combinado: obtiene predicción Y features en un único forward pass
        combined_cnn = tf_keras.Model(
            inputs=cnn_model.input,
            outputs=[cnn_model.output, gmp_layer.output],
        )
    if ML_PATH and os.path.exists(ML_PATH):
        ml_model = joblib.load(ML_PATH)
        def parchear_modelo_antiguo(estimator):
            """
            Inyecta atributos faltantes para que modelos de scikit-learn < 1.4
            funcionen en versiones modernas (1.4+).
            """
            if isinstance(estimator, Pipeline):
                for _, step in estimator.steps:
                    parchear_modelo_antiguo(step)
            
            elif isinstance(estimator, ColumnTransformer):
                if not hasattr(estimator, '_name_to_fitted_passthrough'):
                    estimator._name_to_fitted_passthrough = {}
                    print(" Modelo ML parcheado para compatibilidad con scikit-learn 1.4+")

        parchear_modelo_antiguo(ml_model)

    if CLASS300_PATH and os.path.exists(CLASS300_PATH):
        class_300_model = tf_keras.models.load_model(CLASS300_PATH)

    # Warmup: elimina la latencia de compilación TF en la primera petición real.
    import numpy as _np
    _d512 = _np.zeros((1, 512, 512, 3), dtype=_np.float32)
    if combined_cnn:
        combined_cnn.predict(_d512, verbose=0)
    elif cnn_model:
        cnn_model.predict(_d512, verbose=0)
    if class_300_model:
        class_300_model.predict(_np.zeros((1, 300, 300, 3), dtype=_np.float32), verbose=0)
    print(" TF Models Loaded & warmed up")
except Exception as e:
    print(f" TF load error: {e}")

try:
    if UNET_PATH and os.path.exists(UNET_PATH):
        model = smp.UnetPlusPlus(encoder_name="efficientnet-b2", encoder_weights=None, in_channels=3, classes=1)
        checkpoint = torch.load(UNET_PATH, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {k.replace("model.", "").replace("net.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device).eval()
        unet_model = model
        print(" PyTorch U-Net Loaded")
except Exception as e:
    print(f" U-Net Error: {e}")


try:
    if RE_PT_PATH and os.path.exists(str(RE_PT_PATH)):
        yolo_re = YOLO(RE_PT_PATH)
        print("RE model Loaded")

    if RI_PT_PATH and os.path.exists(str(RI_PT_PATH)):
        yolo_ri = YOLO(RI_PT_PATH)
        print("RI model Loaded")

    # Warmup: la primera inferencia YOLO siempre es lenta (compilación JIT).
    # Hacer un pase en vacío al arrancar para que las peticiones reales sean rápidas.
    import numpy as _np
    _dummy = _np.zeros((896, 896, 3), dtype=_np.uint8)
    if yolo_re:
        yolo_re.predict(_dummy, imgsz=896, verbose=False)
    if yolo_ri:
        yolo_ri.predict(_dummy, imgsz=896, verbose=False)
    print("YOLO warmup done")
except Exception as e:
    print(f"YOLO Load Error: {e}")