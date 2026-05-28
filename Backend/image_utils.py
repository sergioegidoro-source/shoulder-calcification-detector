import cv2
import numpy as np
import base64
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import re

def img_to_base64(img_array: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", img_array)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buffer).decode("utf-8")

def letterbox_square(img: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona a size×size con padding negro, sin distorsionar el ratio."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size) + img.shape[2:], dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def img_to_base64_display(img_array: np.ndarray, max_side: int = 1024) -> str:
    """Escala la imagen proporcionalmente si supera max_side px en cualquier dimensión.
    No recorta ni distorsiona — el CSS object-fit:contain gestiona el encaje."""
    h, w = img_array.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_array = cv2.resize(img_array, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    return img_to_base64(img_array)

# image_utils.py

def procesar_imagen_base(file_stream, filename: str) -> tuple[np.ndarray, str, dict]:
    """
    Carga DICOM o imagen estándar.
    Devuelve: (img_uint8, laterality_str, metadata_dict)
    """
    file_stream.seek(0)
    laterality = "Unknown"
    # Valores por defecto (None) para saber si encontramos algo o no
    metadata = {"age": None, "sex": None} 
    
    img_final: np.ndarray | None = None

    if filename.lower().endswith(".dcm"):
        ds = None
        try:
            ds = pydicom.dcmread(file_stream)

            # --- 1. Extraer Lateralidad ---
            tag_lat = getattr(ds, "ImageLaterality", getattr(ds, "Laterality", None))
            if tag_lat:
                laterality = {"R": "Right", "L": "Left"}.get(tag_lat, "Unknown")

            # --- 2. Extraer Edad ---
            age_raw = getattr(ds, "PatientAge", None)
            if age_raw:
                digits = re.sub(r"[^0-9]", "", str(age_raw))
                if digits:
                    metadata["age"] = int(digits)

            # --- 3. Extraer Sexo ---
            sex_str = getattr(ds, "PatientSex", None)
            if sex_str == "M":
                metadata["sex"] = 1
            elif sex_str == "F":
                metadata["sex"] = 0

            # --- Procesamiento de imagen ---
            if "WindowWidth" in ds and "WindowCenter" in ds:
                img_array = apply_voi_lut(ds.pixel_array, ds)
            else:
                img_array = ds.pixel_array.astype(float)

            if ds.PhotometricInterpretation == "MONOCHROME1":
                img_array = np.amax(img_array) - img_array

            img_array = img_array - np.min(img_array)
            if np.max(img_array) != 0:
                img_array = img_array / np.max(img_array)

            img_final = (img_array * 255).astype(np.uint8)

        except Exception as e:
            print(f"Warning: DICOM processing error ({filename}): {e}")
            if ds is None:
                raise RuntimeError(f"Cannot read DICOM file: {filename}") from e
            # ds se leyó OK pero apply_voi_lut u otra operación falló: usar píxeles crudos
            img_arr = ds.pixel_array.astype(np.float32)
            mn, mx = img_arr.min(), img_arr.max()
            img_final = ((img_arr - mn) / max(mx - mn, 1e-10) * 255).astype(np.uint8)
    else:
        # JPG/PNG no tienen metadatos clínicos fiables incrustados
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        img_final = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img_final = img_final.astype(np.float32)
        img_final = ((img_final - np.min(img_final)) / (np.max(img_final) - np.min(img_final)) * 255).astype(np.uint8)

    # Heurística de lateralidad si falta
    if laterality in ("Unknown", "N/A"):
        h, w = img_final.shape
        avg_left = np.mean(img_final[:, : w // 2])
        avg_right = np.mean(img_final[:, w // 2 :])
        laterality = "Right" if avg_right > avg_left else "Left"

    return img_final, laterality, metadata