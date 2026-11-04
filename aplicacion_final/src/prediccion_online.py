import math
import os
import pickle
import pandas as pd
from river import linear_model, optim, preprocessing, time_series
from config import COL_KWH, COL_TEMP, COL_TIMESTAMP, DIR_ONLINE

def cargar_modelo_snarimax_inicial(cups_sel: str) -> dict:
    ruta_modelo = os.path.join(DIR_ONLINE, "river_snarimax_warmup.pkl")
    if not os.path.exists(ruta_modelo):
        motivo = f"no existe {ruta_modelo}"
    else:
        try:
            with open(ruta_modelo, "rb") as f:
                modelos_dict = pickle.load(f)
            if cups_sel in modelos_dict:
                obj = modelos_dict[cups_sel]
                obj["origen"] = "warmup"
                return obj
            motivo = "la vivienda no está en el fichero de warm-up"
        except Exception as e:
            motivo = f"error al leer el .pkl ({e})"

    return {
        "model": time_series.SNARIMAX(
            p=1, d=0, q=1, m=96, sp=1, sd=0, sq=0,
            regressor=linear_model.LinearRegression(optimizer=optim.SGD(lr=0.01)),
        ),
        "scaler": preprocessing.StandardScaler(),
        "mejor_lr": 0.01,
        "origen": "cold_start",
        "motivo": motivo,
    }


def extraer_features_online(ts: pd.Timestamp, temperatura: float = 0.0) -> dict:
    """Calcula el diccionario de características exógenas que espera el modelo."""
    hora_frac = ts.hour + (ts.minute / 60.0)
    hora_sin = math.sin(2 * math.pi * hora_frac / 24.0)
    hora_cos = math.cos(2 * math.pi * hora_frac / 24.0)
    es_fin_de_semana = 1 if ts.weekday() >= 5 else 0

    return {
        "temperatura": temperatura,
        "hora_sin": hora_sin,
        "hora_cos": hora_cos,
        "es_fin_de_semana": es_fin_de_semana,
    }


def avanzar_simulacion(df_casa: pd.DataFrame, inicio: int, fin: int, obj_modelo: dict) -> list:
    """
    Ejecuta el bucle de evaluación precuencial.
    Devuelve una lista de diccionarios con el resultado de las nuevas predicciones.
    """
    nuevos_resultados = []

    for idx in range(inicio, fin):
        fila = df_casa.iloc[idx]
        ts = fila[COL_TIMESTAMP]
        y_real = fila[COL_KWH]
        temp = fila.get(COL_TEMP, 0.0)

        x_actual = extraer_features_online(ts, temp)

        # Inferencia y escalado SNARIMAX
        if isinstance(obj_modelo, dict):
            model = obj_modelo["model"]
            scaler = obj_modelo["scaler"]

            # Escalar características (aprender distribución de X)
            scaler.learn_one(x_actual)
            x_scaled = scaler.transform_one(x_actual)

            # Predecir (Prequential: antes de conocer Y)
            try:
                preds = model.forecast(horizon=1, xs=[x_scaled])
                pred = preds[0] if preds else 0.0
            except Exception:
                pred = (
                    model.predict_one(x_scaled)
                    if hasattr(model, "predict_one")
                    else 0.0
                )

            pred = max(0.0, float(pred)) if pred is not None else 0.0
            
            # Aprender (Actualizar pesos conociendo Y)
            model.learn_one(y_real, x_scaled)
        
        else:
            # Por si llega un modelo sin envolver en diccionario
            pred = obj_modelo.predict_one(x_actual)
            pred = max(0.0, float(pred)) if pred is not None else 0.0
            obj_modelo.learn_one(x_actual, y_real)

        # Almacenar resultado del paso actual
        nuevos_resultados.append({
            "timestamp": ts,
            "Consumo_Real": y_real,
            "Prediccion": pred,
        })

    return nuevos_resultados