import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from config import COL_TIMESTAMP, COL_KWH, COL_CUPS, COL_CIUDAD

BASE_DIR = Path(__file__).resolve().parent.parent

# Características requeridas por Ridge
FEATURES = [
    "hora",
    "dia_semana",
    "mes",
    "es_finde",
    "lag_24h",
    "lag_168h",
    "rolling_24h_media",
    "rolling_7d_media",
    "temperatura",
]

def cargar_dataset_features() -> pd.DataFrame | None:
    """Carga el dataset procesado con todas las features calculadas."""
    ruta = BASE_DIR / "modelos" / "batch" / "dataset_features.csv"
    if not ruta.exists():
        return None
    df = pd.read_csv(ruta)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def cargar_modelo_ridge(cups_id: str):
    """Carga el modelo individual guardado en .joblib"""
    ruta = BASE_DIR / "modelos" / "batch" / "ridge_modelos" / f"ridge_{cups_id}.joblib"
    if not ruta.exists():
        ruta_pkl = BASE_DIR / "modelos" / "batch" / "ridge_modelos" / f"ridge_{cups_id}.pkl"
        if not ruta_pkl.exists():
            return None
        ruta = ruta_pkl
    return joblib.load(ruta)

def ejecutar_inferencia_batch(df_sub: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta la inferencia en batch agrupando por vivienda (CUPS)."""
    df_res = df_sub.copy()
    predicciones = pd.Series(index=df_res.index, dtype=float)

    # Inferencia casa por casa
    for cups, group in df_res.groupby(COL_CUPS):
        modelo = cargar_modelo_ridge(cups)
        if modelo is not None:
            preds = modelo.predict(group[FEATURES])
            predicciones.loc[group.index] = np.clip(preds, a_min=0, a_max=None)

    df_res["Prediccion"] = predicciones
    df_res = df_res.dropna(subset=["Prediccion"])

    # Si hay varias viviendas se hace la media
    if df_res[COL_CUPS].nunique() > 1:
        df_res = (
            df_res.groupby("timestamp")[["kWh", "Prediccion"]]
            .mean()
            .reset_index()
        )
        
    df_res = df_res.rename(columns={"kWh": "Consumo_Real"})
    return df_res