import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────
# CARGA DE ENTORNO LOCAL
# HuggingFace Spaces siempre inyecta SPACE_ID; cuando no existe
# se asume ejecución local y se carga .env.local (no versionado).
# ─────────────────────────────────────────────────────────────────
def _load_local_env() -> None:
    if os.environ.get("SPACE_ID"):          # En HF → usar secretos del Space
        return
    env_path = os.path.join(BASE_DIR, ".env.local")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                # setdefault: las variables ya presentes en el entorno tienen prioridad
                os.environ.setdefault(key.strip(), val.strip())


_load_local_env()


def _path(env_var: str) -> str | None:
    """Devuelve la ruta absoluta al archivo o None si la variable no está definida."""
    name = os.environ.get(env_var)
    return os.path.join(BASE_DIR, name) if name else None


# ─────────────────────────────────────────────────────────────────
# RUTAS Y IDs DE DRIVE
# ─────────────────────────────────────────────────────────────────

# CNN (VGG19 + GlobalMaxPooling — clasificador principal)
CNN_DRIVE_ID  = os.environ.get("CNN_DRIVE_ID")
CNN_PATH      = _path("CNN_FILENAME_URL")

# Hybrid ML (scikit-learn pipeline)
ML_DRIVE_ID   = os.environ.get("ML_DRIVE_ID")
ML_PATH       = _path("ML_FILENAME_URL")

# Lista de features seleccionadas para el modelo híbrido
FEAT_DRIVE_ID = os.environ.get("FEAT_DRIVE_ID")
FEAT_PATH     = _path("FEATURES_FILENAME_URL")

# Cropping U-Net (EfficientNet-B2 UNet++, usado en crop_with_unet)
UNET_DRIVE_ID = os.environ.get("UNET_DRIVE_ID")
UNET_PATH     = _path("UNET_FILENAME_URL")

# Clasificador 300px (SEG_300 pipeline)
CLASS300_DRIVE_ID = os.environ.get("CLASS300_DRIVE_ID")
CLASS300_PATH     = _path("CLASS300_FILENAME_URL")

# YOLO RE (Rotator External / supraespinoso vista RE)
RE_PT_DRIVE_ID = os.environ.get("RE_PT_DRIVE_ID")
RE_PT_PATH     = _path("RE_PT_FILENAME_URL") or _path("RE_PT_PATH")

# YOLO RI (Rotator Internal / infraespinoso)
RI_PT_DRIVE_ID = os.environ.get("RI_PT_DRIVE_ID")
RI_PT_PATH     = _path("RI_PT_FILENAME_URL") or _path("RI_PT_PATH")

# Calcium LR classifier (joblib)
MODELJOBLIB_DRIVE_ID = os.environ.get("MODELJOBLIB_DRIVE_ID")
MODELJOBLIB_PATH     = _path("MODELJOBLIB_FILENAME_URL")

# Calcium U-Net weights (.pth)
MODELPTH_DRIVE_ID = os.environ.get("MODELPTH_DRIVE_ID")
MODELPTH_PATH     = _path("MODELPTH_FILENAME_URL")

# Fichero de estadísticas de uso
STATS_FILE = os.path.join(BASE_DIR, "stats.json")
