import cv2
import numpy as np
import torch
import tf_keras
import tensorflow as tf
import base64
import hashlib
import time as _time
from model_loader import unet_model, class_300_model, cnn_model, device
from image_utils import img_to_base64

# Cache para crop_with_unet: evita repetir la inferencia del UNet
# cuando varias rutas procesan la misma imagen en la misma sesión.
_crop_cache: dict = {}
_CROP_CACHE_TTL = 180  # segundos

def _crop_fingerprint(img: np.ndarray) -> str:
    """Huella rápida basada en los primeros 4096 bytes + forma."""
    sample = img.ravel()[:4096].tobytes()
    return hashlib.md5(sample + str(img.shape).encode()).hexdigest()

# ============================================================
# LOGICA DE RECORTE (U-NET)
# ============================================================
def crop_with_unet(img_uint8: np.ndarray, unet_model, mask_thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Versión con caché: si la misma imagen se pide varias veces en la misma
    sesión (p.ej. /predict y /predict_yolo), el UNet solo corre una vez.
    """
    key = _crop_fingerprint(img_uint8)
    entry = _crop_cache.get(key)
    if entry and _time.time() - entry[2] < _CROP_CACHE_TTL:
        return entry[0], entry[1]

    crop, mask_bin = _crop_with_unet_impl(img_uint8, unet_model, mask_thr)
    _crop_cache[key] = (crop, mask_bin, _time.time())
    return crop, mask_bin


def _crop_with_unet_impl(img_uint8: np.ndarray, unet_model, mask_thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if unet_model is None:
        h, w = img_uint8.shape
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
        return img_uint8[y0 : y0 + side, x0 : x0 + side], np.zeros((512, 512), dtype=np.uint8)

    img_512 = cv2.resize(img_uint8, (512, 512), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_512, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = unet_model(img_tensor)
        mask = torch.sigmoid(out)[0, 0].detach().cpu().numpy()

    mask_bin = (mask > mask_thr).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    orig_h, orig_w = img_uint8.shape
    scale_x = orig_w / 512.0
    scale_y = orig_h / 512.0

    if not contours:
        cx, cy = orig_w // 2, orig_h // 2
        side_size = min(orig_w, orig_h) // 2 
    else:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        x, y, w, h = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)

        HEAD_BIAS_RATIO = 0.2 

        cx = int(x + w // 2)                  
        cy = int(y + (h * HEAD_BIAS_RATIO))   

        max_dim = max(w, h)
        
        side_size = int(max_dim * 1.1)

    half_side = side_size // 2
    x0, y0 = cx - half_side, cy - half_side
    x1, y1 = x0 + side_size, y0 + side_size

    if x0 < 0: x1 += abs(x0); x0 = 0
    if x1 > orig_w: x0 -= (x1 - orig_w); x1 = orig_w
    if y0 < 0: y1 += abs(y0); y0 = 0
    if y1 > orig_h: y0 -= (y1 - orig_h); y1 = orig_h

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(orig_w, x1), min(orig_h, y1)
    
    crop = img_uint8[y0:y1, x0:x1]
    
    # Padding final para garantizar cuadrado
    ch, cw = crop.shape
    if ch != cw:
        diff = abs(ch - cw)
        p1 = diff // 2
        p2 = diff - p1
        if ch < cw: crop = np.pad(crop, ((p1, p2), (0, 0)), mode='constant')
        else: crop = np.pad(crop, ((0, 0), (p1, p2)), mode='constant')

    return crop, mask_bin

# ============================================================
# CLASIFICACIÓN 300
# ============================================================
def classify_crop(crop_uint8: np.ndarray, model_300) -> tuple[np.ndarray, float]:
    crop_300 = cv2.resize(crop_uint8, (300, 300), interpolation=cv2.INTER_AREA)
    crop_rgb = cv2.cvtColor(crop_300, cv2.COLOR_GRAY2RGB)
    crop_norm = crop_rgb.astype("float32") / 255.0
    crop_in = np.expand_dims(crop_norm, axis=0)

    prob = 0.0
    if model_300 is not None:
        try:
            prob = float(model_300.predict(crop_in, verbose=0)[0][0])
        except: pass
    return crop_300, prob

def pipeline_300(img_uint8: np.ndarray, laterality: str, threshold: float = 0.5):
    # 1. Crop
    crop_uint8, mask_512 = crop_with_unet(img_uint8, unet_model, mask_thr=0.5)
    
    # 2. Mirroring logic for Left shoulders
    is_left = laterality.lower().startswith(("l", "left", "i"))
    # (Opcional: Si quieres voltear el crop antes de clasificar, descomenta esto)
    # if is_left: crop_uint8 = cv2.flip(crop_uint8, 1)

    crop_display = crop_uint8.copy()
    
    # 3. Classify
    crop_300, prob = classify_crop(crop_uint8, class_300_model)
    
    mask_rgb = cv2.cvtColor(mask_512, cv2.COLOR_GRAY2BGR)
    
    return (
        cv2.resize(img_uint8, (320, 320), interpolation=cv2.INTER_AREA),
        img_to_base64(mask_rgb),
        img_to_base64(crop_display),
        prob,
        "Tendinopathy Detected" if prob >= threshold else "No Tendinopathy Detected",
    )

# ============================================================
# GRAD-CAM
# ============================================================

# Caché de modelos GradCAM: se construyen una vez por modelo Keras
# (keyed by id(model), que es estable durante toda la ejecución).
_gradcam_cache: dict = {}


def _build_gradcam_models(model):
    """Construye y cachea los sub-modelos necesarios para Grad-CAM."""
    key = id(model)
    if key in _gradcam_cache:
        return _gradcam_cache[key]

    last_conv_layer_name = "block5_conv4"
    base_model = model.get_layer("vgg19") if hasattr(model, "get_layer") else model

    conv_layer = None
    try:
        conv_layer = base_model.get_layer(last_conv_layer_name)
    except Exception:
        for layer in reversed(base_model.layers):
            if "Conv2D" in layer.__class__.__name__:
                conv_layer = layer
                break

    if conv_layer is None:
        return None, None

    conv_model = tf_keras.models.Model(inputs=base_model.input, outputs=conv_layer.output)
    classifier_input = tf_keras.Input(shape=conv_model.output.shape[1:])
    x = classifier_input
    try:
        gmp = [l for l in model.layers if "GlobalMax" in l.__class__.__name__][0]
        dense = [l for l in model.layers if "Dense" in l.__class__.__name__][-1]
        x = gmp(x)
        x = tf_keras.layers.Dense(units=dense.units, weights=dense.get_weights())(x)
    except Exception:
        return None, None

    classifier_model = tf_keras.models.Model(classifier_input, x)
    _gradcam_cache[key] = (conv_model, classifier_model)
    return conv_model, classifier_model


def get_gradcam_image(model, img_input_tensor, original_uint8_bgr):
    try:
        conv_model, classifier_model = _build_gradcam_models(model)
        if conv_model is None:
            return None

        with tf.GradientTape() as tape:
            conv_output = conv_model(img_input_tensor)
            tape.watch(conv_output)
            preds = classifier_model(conv_output)
            class_idx = preds[:, 0]

        grads = tape.gradient(class_idx, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_output[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_output, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        heatmap = 1.0 - heatmap # Invertir si es necesario segun entrenamiento

        heatmap = cv2.resize(heatmap, (original_uint8_bgr.shape[1], original_uint8_bgr.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return img_to_base64(cv2.addWeighted(heatmap_color, 0.4, original_uint8_bgr, 1.0, 0))
    except Exception as e:
        print(f"GradCAM Error: {e}")
        return None