import json
import os
from config import STATS_FILE

DEFAULT_STATS = {
    "models": {"cnn": 0, "seg": 0},
    "laterality": {"Left": 0, "Right": 0, "Unknown": 0},
    "diagnosis": {"Positive": 0, "Negative": 0},
    # Backward-compat root keys
    "cnn": 0,
    "seg": 0,
}

# In-memory cache: evita leer y escribir el JSON en cada predicción.
_cache: dict | None = None


def load_stats() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    if not os.path.exists(STATS_FILE):
        _cache = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_STATS.items()}
        return _cache

    try:
        with open(STATS_FILE, "r") as f:
            data = json.load(f)

        # Añadir claves nuevas que puedan faltar en archivos viejos
        for key, val in DEFAULT_STATS.items():
            if key not in data:
                data[key] = val.copy() if isinstance(val, dict) else val
        for sub_key in DEFAULT_STATS["laterality"]:
            data["laterality"].setdefault(sub_key, 0)

        # Sincronizar claves raíz con el subdict (migración)
        if data["models"]["cnn"] == 0 and data.get("cnn", 0) > 0:
            data["models"]["cnn"] = data["cnn"]
        if data["models"]["seg"] == 0 and data.get("seg", 0) > 0:
            data["models"]["seg"] = data["seg"]

        _cache = data
        return _cache
    except Exception:
        _cache = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_STATS.items()}
        return _cache


def save_stats(stats: dict) -> None:
    global _cache
    _cache = stats
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=4)
    except Exception as e:
        print(f"Error saving stats: {e}")


def update_stats(model_type: str, laterality: str, diagnosis_is_positive: bool) -> None:
    stats = load_stats()

    if model_type:
        stats["models"][model_type] = stats["models"].get(model_type, 0) + 1
        if model_type in ("cnn", "seg"):
            stats[model_type] = stats.get(model_type, 0) + 1

    lat_key = "Unknown"
    if laterality:
        lat_clean = laterality.upper()
        if lat_clean.startswith("L") or "IZQ" in lat_clean:
            lat_key = "Left"
        elif lat_clean.startswith("R") or "DER" in lat_clean:
            lat_key = "Right"
    stats["laterality"][lat_key] = stats["laterality"].get(lat_key, 0) + 1

    diag_key = "Positive" if diagnosis_is_positive else "Negative"
    stats["diagnosis"][diag_key] = stats["diagnosis"].get(diag_key, 0) + 1

    save_stats(stats)


def get_stats_data() -> dict:
    return load_stats()
