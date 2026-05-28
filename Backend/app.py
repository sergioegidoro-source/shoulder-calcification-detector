import pandas as pd
import numpy as np
import cv2
import base64
import joblib
import traceback
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import config
from config import FEAT_PATH
from stats_manager import update_stats, get_stats_data
from image_utils import procesar_imagen_base, img_to_base64, img_to_base64_display, letterbox_square
from model_loader import cnn_model, ml_model, feature_extractor, combined_cnn, unet_model, class_300_model, yolo_re, yolo_ri
from ai_logic import pipeline_300, get_gradcam_image, crop_with_unet
from calcium_unet import CalciumUNet
from calcium_radiomics import CalciumRadiomics
from calcium_classifier import CalciumClassifier
from Calcium_Pipeline import CalciumPipeline

app = Flask(__name__, static_folder="../Frontend")
CORS(app)

# ============================================================
# 1. LOAD FEATURE LIST
# ============================================================
try:
    print(f"Loading feature list from: {FEAT_PATH}")
    SELECTED_FEATURES = joblib.load(FEAT_PATH)
    # Cleaning feature names
    SELECTED_FEATURES = [str(f).strip() for f in SELECTED_FEATURES]
    print(f"Feature list loaded. Model expects {len(SELECTED_FEATURES)} variables.")
except Exception as e:
    print(f"ERROR: Could not load final_features.pkl: {e}")
    SELECTED_FEATURES = []

DEFAULT_AGE = 75
DEFAULT_SEX = 0

# ------------------------------
# Modelos globales calcio
# ------------------------------

try:
    calcium_unet       = CalciumUNet()
    calcium_radiomics  = CalciumRadiomics()
    calcium_classifier = CalciumClassifier()
    calcium_pipeline   = CalciumPipeline(calcium_unet, calcium_radiomics, calcium_classifier)
except Exception as e:
    print(f"Calcium pipeline not available: {e}")
    calcium_pipeline = None

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def serve_index(): return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path): return send_from_directory(app.static_folder, path)

@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify(get_stats_data())

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    threshold = float(request.form.get("threshold", 0.5))

    try:
        img_uint8, laterality, _ = procesar_imagen_base(file, file.filename)
        img_320, mask_b64, crop_b64, prob, diagnosis = pipeline_300(img_uint8, laterality, threshold=threshold)

        update_stats("seg", laterality, prob >= threshold)

        return jsonify({
            "success": True,
            "mode": "SEG_300",
            "laterality": laterality,
            "image_original": f"data:image/png;base64,{img_to_base64_display(img_uint8)}",
            "image_segmented": f"data:image/png;base64,{mask_b64}" if mask_b64 else None,
            "image_cropped": f"data:image/png;base64,{crop_b64}" if crop_b64 else None,
            "result_seg": {"prob": prob, "diagnosis": diagnosis},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict_512_fast", methods=["POST"])
def predict_512_fast():
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    threshold = float(request.form.get("threshold", 0.5))

    try:
        # 1. Image and metadata
        img_uint8, laterality, meta = procesar_imagen_base(file, file.filename)
        
        # 2. Clinical Data
        manual_age = request.form.get("age")
        manual_sex = request.form.get("sex")

        final_age = DEFAULT_AGE
        if manual_age and manual_age != "null" and manual_age != "": final_age = int(manual_age)
        elif meta.get("age") is not None: final_age = meta["age"]

        final_sex = DEFAULT_SEX
        if manual_sex and manual_sex != "null" and manual_sex != "": final_sex = int(manual_sex)
        elif meta.get("sex") is not None: final_sex = meta["sex"]

        # 3. Prepare Image
        img_512 = cv2.resize(img_uint8, (512, 512), interpolation=cv2.INTER_AREA)
        img_ai = np.expand_dims(cv2.cvtColor(img_512, cv2.COLOR_GRAY2RGB).astype("float32")/255.0, axis=0)
        
        # 4. CNN + Features (un único forward pass si combined_cnn está disponible)
        prob_cnn = 0.0
        prob_hyb = 0.0

        if combined_cnn and ml_model and len(SELECTED_FEATURES) > 0:
            try:
                prob_arr, feats = combined_cnn.predict(img_ai, verbose=0)
                prob_cnn = float(prob_arr[0][0])
                if feats.ndim > 2:
                    feats = feats.reshape(1, -1)
                cols_generated = [f"feature_{i}" for i in range(feats.shape[1])]
                pool_df = pd.DataFrame(feats, columns=cols_generated)
                pool_df["sex"] = final_sex
                pool_df["edad"] = final_age
                final_df = pool_df.reindex(columns=SELECTED_FEATURES, fill_value=0)
                prob_hyb = float(ml_model.predict_proba(final_df)[0][1])
            except Exception as e_hybrid:
                print(f"Hybrid Model Error (Ignored): {e_hybrid}")
                prob_hyb = prob_cnn
        elif cnn_model:
            prob_cnn = float(cnn_model.predict(img_ai, verbose=0)[0][0])
            prob_hyb = prob_cnn

        update_stats("cnn", laterality, prob_cnn >= threshold)

        return jsonify({
            "success": True,
            "laterality": laterality,
            "image_original": f"data:image/png;base64,{img_to_base64_display(img_uint8)}",
            "result_cnn": {
                "prob": prob_cnn, 
                "diagnosis": "Tendinopathy Detected" if prob_cnn >= threshold else "No Tendinopathy Detected"
            },
            "result_hybrid": {
                "prob": prob_hyb, 
                "diagnosis": "Tendinopathy Detected" if prob_hyb >= threshold else "No Tendinopathy Detected"
            },
        })

    except Exception as e: 
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict_512_gradcam", methods=["POST"])
def predict_512_gradcam_route():
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    try:
        img_uint8, _, _ = procesar_imagen_base(file, file.filename)
        img_512 = cv2.resize(img_uint8, (512, 512), interpolation=cv2.INTER_AREA)
        img_ai = np.expand_dims(cv2.cvtColor(img_512, cv2.COLOR_GRAY2RGB).astype("float32")/255.0, axis=0)

        gradcam = None
        if cnn_model:
            img_512_bg = letterbox_square(img_uint8, 512)
            bg_img = cv2.cvtColor(cv2.cvtColor(img_512_bg, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2BGR)
            gradcam = get_gradcam_image(cnn_model, img_ai, bg_img)
            
        return jsonify({
            "success": True, 
            "gradcam_preview": f"data:image/png;base64,{gradcam}" if gradcam else None
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/predict_300_gradcam', methods=['POST'])
def predict_300_gradcam_route():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    try:
        img_uint8, laterality, _ = procesar_imagen_base(file, file.filename)
        
        is_left = laterality.lower().startswith(("l", "left", "i"))
        
        img_input_stage = cv2.flip(img_uint8, 1) if is_left else img_uint8
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        img_enhanced = clahe.apply(img_input_stage)
        
        crop_final, _ = crop_with_unet(img_enhanced, unet_model, mask_thr=0.5)
        
        if crop_final is None or crop_final.size == 0: 
            return jsonify({'error': 'No crop found'}), 400

        crop_300 = cv2.resize(crop_final, (300, 300), interpolation=cv2.INTER_AREA)
        crop_ai = np.expand_dims(cv2.cvtColor(crop_300, cv2.COLOR_GRAY2RGB).astype('float32')/255.0, axis=0)
        crop_300_bg = letterbox_square(crop_final, 300)
        crop_bgr = cv2.cvtColor(crop_300_bg, cv2.COLOR_RGB2BGR)
        
        gradcam_b64 = None
        if class_300_model:
            gradcam_b64 = get_gradcam_image(class_300_model, crop_ai, crop_bgr)
            
            if is_left and gradcam_b64:
                try:
                    nparr = np.frombuffer(base64.b64decode(gradcam_b64), np.uint8)
                    img_grad = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img_grad = cv2.flip(img_grad, 1)
                    gradcam_b64 = img_to_base64(img_grad)
                except: pass

        return jsonify({'success': True, 'gradcam_preview': f"data:image/png;base64,{gradcam_b64}" if gradcam_b64 else None})
    except Exception as e: 
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route("/predict_yolo", methods=["POST"])
def predict_yolo():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # ---------------------------------------------------------
        # 1. PREPROCESADO
        # ---------------------------------------------------------
        img_uint8, laterality, _ = procesar_imagen_base(file, file.filename)

        crop_img, _ = crop_with_unet(img_uint8, unet_model)

        img_896 = cv2.resize(crop_img, (896, 896), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_896, cv2.COLOR_GRAY2RGB)

        # ---------------------------------------------------------
        # 2. YOLO DETECCIÓN
        # ---------------------------------------------------------
        conf_min_detect = 0.15
        SAFE_THRESHOLD = 0.40

        # Inferencia RI y RE en paralelo: son instancias distintas y PyTorch
        # libera el GIL durante la computación tensorial, por lo que los dos
        # modelos corren solapados tanto en CPU como en GPU.
        def _run_ri():
            return yolo_ri.predict(img_rgb, conf=conf_min_detect, imgsz=896) if yolo_ri else []

        def _run_re():
            return yolo_re.predict(img_rgb, conf=conf_min_detect, imgsz=896) if yolo_re else []

        with ThreadPoolExecutor(max_workers=2) as _pool:
            _f_ri = _pool.submit(_run_ri)
            _f_re = _pool.submit(_run_re)
            results_ri = _f_ri.result()
            results_re = _f_re.result()

        # ---------------------------------------------------------
        # 3. UNA SOLA CAJA: LA DE MAYOR CONFIANZA DE TODOS LOS MODELOS
        # ---------------------------------------------------------
        best_detection = None  # {"cls_name": str, "conf": float, "xyxy": np.ndarray}

        for results, yolo_model in [(results_ri, yolo_ri), (results_re, yolo_re)]:
            if not yolo_model or len(results) == 0 or len(results[0].boxes) == 0:
                continue
            boxes = results[0].boxes
            for i in range(len(boxes)):
                conf_val = float(boxes.conf[i].item())
                if conf_val < SAFE_THRESHOLD:
                    continue
                if best_detection is None or conf_val > best_detection["conf"]:
                    cls_name = yolo_model.names[int(boxes.cls[i].item())]
                    best_detection = {
                        "cls_name": cls_name,
                        "conf": conf_val,
                        "xyxy": boxes.xyxy[i].cpu().numpy().astype(int),
                    }

        # ---------------------------------------------------------
        # 4. DECISIÓN FINAL
        # ---------------------------------------------------------
        final_img_yolo = img_rgb.copy()
        final_img_calcium = None
        prediction_text = None
        model_info = "No Calcification Detected"

        if best_detection:
            # Dibujar la única caja ganadora
            x1, y1, x2, y2 = best_detection["xyxy"]
            detected_class = best_detection["cls_name"]
            detected_conf  = best_detection["conf"]
            label = f"{detected_class} {detected_conf:.2f}"
            cv2.rectangle(final_img_yolo, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.putText(
                final_img_yolo, label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2,
            )
            model_info = f"{detected_class} detected - Conf: {detected_conf:.2f}"

            # Pipeline de calcio solo para supr/infr
            if detected_class in ("supr", "infr"):
                img_for_calcium = crop_img.copy()
                if len(img_for_calcium.shape) == 3:
                    img_for_calcium = cv2.cvtColor(img_for_calcium, cv2.COLOR_RGB2GRAY)
                if img_for_calcium.dtype != np.uint8:
                    img_for_calcium = (
                        (img_for_calcium * 255).astype(np.uint8)
                        if img_for_calcium.max() <= 1.0
                        else img_for_calcium.astype(np.uint8)
                    )

                try:
                    calcium_result = calcium_pipeline.run(img_for_calcium) if calcium_pipeline else None
                except Exception as e_calc:
                    print(f"Calcium pipeline error: {e_calc}")
                    calcium_result = None

                if calcium_result and calcium_result.get("prediction") is not None:
                    prediction_text = calcium_result["prediction"]
                    mask_final = calcium_result["mask"]

                    base_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
                    img_overlay = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
                    alpha = 0.35
                    red = np.zeros_like(img_overlay)
                    red[:, :, 2] = 255
                    mask_bool = mask_final > 0
                    img_overlay[mask_bool] = (
                        alpha * red[mask_bool] + (1 - alpha) * img_overlay[mask_bool]
                    ).astype(np.uint8)
                    final_img_calcium = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------
        # 5. RESPUESTA
        # ---------------------------------------------------------
        _, buffer_yolo = cv2.imencode(".png", final_img_yolo)
        img_yolo_base64 = base64.b64encode(buffer_yolo).decode("utf-8")

        img_calcium_base64 = None
        if final_img_calcium is not None:
            _, buffer_calcium = cv2.imencode(".png", final_img_calcium)
            img_calcium_base64 = base64.b64encode(buffer_calcium).decode("utf-8")

        return jsonify({
            "success": True,
            "laterality": laterality,
            "image_yolo": f"data:image/png;base64,{img_yolo_base64}",
            "image_calcium": f"data:image/png;base64,{img_calcium_base64}" if img_calcium_base64 else None,
            "prediction": prediction_text,
            "model_info": model_info,
            "debug_scores": f"Class:{detected_class} | Conf:{detected_conf:.2f}" if best_detection else "No detection above threshold"
        })

    except Exception as e:
        print(f"Error at predict_yolo: {str(e)}")
        return jsonify({"error": str(e)}), 500    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)