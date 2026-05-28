import joblib
import pandas as pd

from config import MODELJOBLIB_PATH


class CalciumClassifier:
    """
    Clasificador final para tipo de calcio:
    - 0 → Tipo I-II
    - 1 → Tipo III

    Usa Logistic Regression entrenado previamente.
    """

    def __init__(self):
        self.model = joblib.load(MODELJOBLIB_PATH)

        # Columnas EXACTAS usadas durante entrenamiento (23 features)
        self.feature_cols = [
            'area_px',
            'eccentricity',
            'solidity',
            'perimeter',
            'bbox_area',
            'int_mean',
            'int_std',
            'int_min',
            'int_max',
            'int_p10',
            'int_p90',
            'glcm_contrast_mean',
            'glcm_contrast_std',
            'glcm_dissimilarity_mean',
            'glcm_dissimilarity_std',
            'glcm_homogeneity_mean',
            'glcm_homogeneity_std',
            'glcm_energy_mean',
            'glcm_energy_std',
            'glcm_correlation_mean',
            'glcm_correlation_std',
            'glcm_ASM_mean',
            'glcm_ASM_std'
        ]

        print("Calcium classifier loaded.")
        print("Expected features:", len(self.feature_cols))

    def predict(self, features_dict):
        """
        features_dict: diccionario con radiomics extraídas desde la máscara
        """

        # Convertir a DataFrame
        df = pd.DataFrame([features_dict])

        # Forzar orden EXACTO
        df = df.reindex(columns=self.feature_cols, fill_value=0)

        # Verificación de seguridad
        if df.shape[1] != 23:
            raise ValueError(
                f"Model expects 23 features but received {df.shape[1]}"
            )

        prob = float(self.model.predict_proba(df)[0, 1])
        label = "Type III" if prob >= 0.5 else "Type I-II"

        return {"prediction": label, "probability": prob}