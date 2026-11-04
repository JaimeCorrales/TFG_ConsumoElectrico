import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula MAE, RMSE, WAPE y R² dado arrays de valores reales y predichos.
    Devuelve un diccionario con las métricas redondeadas.
    """
    y_real = np.array(y_real, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Eliminar pares donde el real es NaN
    mask  = ~np.isnan(y_real) & ~np.isnan(y_pred)
    y_real = y_real[mask]
    y_pred = y_pred[mask]

    if len(y_real) == 0:
        return {"mae": None, "rmse": None, "wape": None, "r2": None}

    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)

    # WAPE: evita división por cero
    suma_real = np.sum(np.abs(y_real))
    wape = (np.sum(np.abs(y_real - y_pred)) / suma_real * 100) if suma_real > 0 else None

    return {
        "mae":  round(mae,  4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 2) if wape is not None else None,
        "r2":   round(r2,   4),
    }

def exportar_csv(df_resultados: pd.DataFrame, metricas: dict) -> bytes:
    """
    Exporta predicciones y métricas en CSV.
    Devuelve bytes para usar con st.download_button.
    """
    df_exp = df_resultados.copy()
    for k, v in metricas.items():
        df_exp[k] = v
    return df_exp.to_csv(index=False, sep=';', decimal=',').encode("utf-8-sig")