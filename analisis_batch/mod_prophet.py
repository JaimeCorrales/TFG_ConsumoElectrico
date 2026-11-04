import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

NOMBRE_MODELO = "prophet"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
ARCHIVO_DATASET = BASE_DIR / "preprocesamiento" / "dataset_final_IA.csv"

if not ARCHIVO_DATASET.exists():
    print(f"Error: No se encuentra '{ARCHIVO_DATASET}'.")
    exit()

DIR_METRICAS = SCRIPT_DIR / "metricas" / NOMBRE_MODELO
DIR_PREDICCIONES = SCRIPT_DIR / "predicciones" / NOMBRE_MODELO
DIR_METRICAS.mkdir(parents=True, exist_ok=True)
DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)

print(f"--------- PASO 1: Cargando datos para {NOMBRE_MODELO.upper()} ---------")

df = pd.read_csv(ARCHIVO_DATASET)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["cups_id", "timestamp"]).reset_index(drop=True)
df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")

# Regresores: temperatura + lags
REGRESORES = ["temperatura", "lag_24h", "lag_168h", "rolling_24h_media", "rolling_7d_media"]

fecha_corte = pd.to_datetime("2021-06-05")
fecha_subperiodo = pd.to_datetime("2021-12-31")

param_grid = {
    "changepoint_prior_scale": [0.01, 0.05, 0.1],
    "seasonality_prior_scale": [1.0, 10.0],
}


def nuevo_prophet(**params):
    m = Prophet(
        daily_seasonality=20,
        weekly_seasonality=15,
        yearly_seasonality=True,
        **params,
    )
    for r in REGRESORES:
        m.add_regressor(r)
    return m


print("\n--------- PASO 2: Entrenando PROPHET con lags (casa por casa) ---------")

lista_resultados = []
lista_metricas = []

for cups in df["cups_id"].unique():
    inicio_casa = time.time()  
    
    df_cups = df[df["cups_id"] == cups].copy()
    ciudad_actual = df_cups["ciudad_asignada"].iloc[0]

    df_cups.set_index("timestamp", inplace=True)
    df_cups = df_cups[~df_cups.index.duplicated(keep="first")]
    df_cups = df_cups.asfreq("h").ffill()

    # Lags calculados sobre la serie completa 
    kwh = df_cups["kWh"]
    df_cups["lag_24h"] = kwh.shift(24)
    df_cups["lag_168h"] = kwh.shift(168)
    df_cups["rolling_24h_media"] = df_cups["lag_24h"].rolling(24, min_periods=24).mean()
    df_cups["rolling_7d_media"] = df_cups["lag_24h"].rolling(168, min_periods=168).mean()

    df_cups = df_cups.dropna(subset=REGRESORES).reset_index()
    df_cups = df_cups.rename(columns={"timestamp": "ds", "kWh": "y"})
    df_cups = df_cups[["ds", "y"] + REGRESORES]

    train = df_cups[df_cups["ds"] < fecha_corte].copy()
    test = df_cups[df_cups["ds"] >= fecha_corte].copy()

    if len(train) < 100 or len(test) == 0:
        print(f"  Datos insuficientes para la casa {cups}. Saltando...")
        continue

    # Selección de hiperparámetros con un único split temporal 70/30 dentro de train
    idx_split = int(len(train) * 0.7)
    train_fold1 = train.iloc[:idx_split]
    val_fold1 = train.iloc[idx_split:]

    best_mae = float("inf")
    best_params = {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0}

    for params in ParameterGrid(param_grid):
        m_cv = nuevo_prophet(**params)
        m_cv.fit(train_fold1)
        pred_cv = m_cv.predict(val_fold1[["ds"] + REGRESORES])
        mae_cv = mean_absolute_error(val_fold1["y"], pred_cv["yhat"])
        if mae_cv < best_mae:
            best_mae = mae_cv
            best_params = params

    print(f" Casa {cups} -> parámetros óptimos: {best_params}")

    model = nuevo_prophet(**best_params)
    model.fit(train)

    prediccion_bruta = model.predict(test[["ds"] + REGRESORES])
    y_pred = prediccion_bruta["yhat"].clip(lower=0).values
    y_real = test["y"].values

    df_res_cups = pd.DataFrame({
        "cups_id": cups,
        "ciudad": ciudad_actual,
        "timestamp": test["ds"].values,
        "Consumo_Real": y_real,
        "Prediccion": y_pred,
    })
    lista_resultados.append(df_res_cups)

    sub_A = df_res_cups[df_res_cups["timestamp"] <= fecha_subperiodo]
    sub_B = df_res_cups[df_res_cups["timestamp"] > fecha_subperiodo]

    fin_casa = time.time()
    latencia_s = fin_casa - inicio_casa

    lista_metricas.append({
        "cups_id": cups,
        "ciudad_asignada": ciudad_actual,
        "changepoint_prior_scale": best_params["changepoint_prior_scale"],
        "seasonality_prior_scale": best_params["seasonality_prior_scale"],
        "mae_total": mean_absolute_error(y_real, y_pred),
        "mae_sub_A": mean_absolute_error(sub_A["Consumo_Real"], sub_A["Prediccion"]) if len(sub_A) > 0 else np.nan,
        "mae_sub_B": mean_absolute_error(sub_B["Consumo_Real"], sub_B["Prediccion"]) if len(sub_B) > 0 else np.nan,
        "rmse": np.sqrt(mean_squared_error(y_real, y_pred)),
        "wape": (np.sum(np.abs(y_real - y_pred)) / np.sum(y_real)) * 100, 
        "r2": r2_score(y_real, y_pred),
        "Latencia_s": latencia_s  
    })

print(f"\n--------- PASO 3: Resultados consolidados {NOMBRE_MODELO.upper()} ---------")
print(f"Casas procesadas con éxito: {len(lista_metricas)}")

df_metricas = pd.DataFrame(lista_metricas)

if lista_resultados:
    df_resultados = pd.concat(lista_resultados).reset_index(drop=True)
    ruta_predicciones = DIR_PREDICCIONES / f"predicciones_{NOMBRE_MODELO}.csv"
    df_resultados.to_csv(ruta_predicciones, index=False)
    print(f"Predicciones exportadas en: {ruta_predicciones}")

df_export = df_metricas.rename(columns={
    "ciudad_asignada": "ciudad",
    "mae_total": "MAE", "mae_sub_A": "MAE_A", "mae_sub_B": "MAE_B",
    "rmse": "RMSE", "wape": "WAPE", "r2": "R2",
})
ruta_metricas = DIR_METRICAS / f"metricas_{NOMBRE_MODELO}.csv"
df_export[["cups_id", "ciudad", "MAE", "MAE_A", "MAE_B", "RMSE", "WAPE", "R2", "Latencia_s",
           "changepoint_prior_scale", "seasonality_prior_scale"]].to_csv(ruta_metricas, index=False)
print(f"Métricas exportadas en: {ruta_metricas}")

print("\n[RENDIMIENTO GLOBAL (1 AÑO)]")
print(f"  MAE:  {df_metricas['mae_total'].mean():.4f} ± {df_metricas['mae_total'].std():.4f} kWh")
print(f"  RMSE: {df_metricas['rmse'].mean():.4f} ± {df_metricas['rmse'].std():.4f} kWh")
print(f"  WAPE: {df_metricas['wape'].mean():.2f} ± {df_metricas['wape'].std():.2f} %")
print(f"  R²:   {df_metricas['r2'].mean()*100:.2f} ± {df_metricas['r2'].std()*100:.2f} %")
print(f"  Latencia media: {df_metricas['Latencia_s'].mean():.2f} s por casa") 

print("\n[ESTABILIDAD TEMPORAL]")
mae_A_mean = df_metricas["mae_sub_A"].mean()
mae_B_mean = df_metricas["mae_sub_B"].mean()
degradacion = ((mae_B_mean - mae_A_mean) / mae_A_mean) * 100 
print(f"  MAE Semestre A (Jun-Dic 21): {mae_A_mean:.4f} kWh")
print(f"  MAE Semestre B (Ene-Jun 22): {mae_B_mean:.4f} kWh")
print(f"  Variación del error: {'+' if degradacion > 0 else ''}{degradacion:.2f}%")

print("\n[HIPERPARÁMETROS MÁS COMUNES ELEGIDOS]")
print("Changepoint Prior Scale:\n", df_metricas["changepoint_prior_scale"].value_counts())
print("Seasonality Prior Scale:\n", df_metricas["seasonality_prior_scale"].value_counts())

resumen_ciudad = (
    df_metricas.groupby("ciudad_asignada")
    .agg(
        MAE_media=("mae_total", "mean"),
        MAE_std=("mae_total", "std"),
        RMSE_media=("rmse", "mean"),
        WAPE_media=("wape", "mean"),
        R2_media=("r2", lambda x: x.mean() * 100),
        Latencia_media=("Latencia_s", "mean"), 
        n_casas=("cups_id", "count"),
    )
    .round(4)
)
print("\n[RENDIMIENTO POR CIUDAD]")
print(resumen_ciudad.to_string())

est_ciudad = (
    df_metricas.groupby("ciudad_asignada")
    .agg(MAE_sub_A=("mae_sub_A", "mean"), MAE_sub_B=("mae_sub_B", "mean"))
    .round(4)
)
est_ciudad["Degradacion_pct"] = (
    (est_ciudad["MAE_sub_B"] - est_ciudad["MAE_sub_A"]) / est_ciudad["MAE_sub_A"] * 100
).round(2)
print("\n[ESTABILIDAD TEMPORAL POR CIUDAD]")
print(est_ciudad.to_string())

resumen_ciudad.reset_index().to_csv(DIR_METRICAS / f"metricas_{NOMBRE_MODELO}_por_ciudad.csv", index=False)
est_ciudad.reset_index().to_csv(DIR_METRICAS / f"estabilidad_{NOMBRE_MODELO}_por_ciudad.csv", index=False)
print(f"\nExportados desgloses por ciudad en: {DIR_METRICAS}")