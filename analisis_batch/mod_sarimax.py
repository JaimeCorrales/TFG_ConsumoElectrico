import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

NOMBRE_MODELO = "sarimax"
HORIZONTE = 24  

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
df = df.sort_values(["cups_id", "timestamp"]).reset_index(drop=True)

df["hora"] = df["timestamp"].dt.hour
df["dia_semana"] = df["timestamp"].dt.dayofweek
df["mes"] = df["timestamp"].dt.month
df["es_finde"] = (df["dia_semana"] >= 5).astype(int)

features = ["hora", "dia_semana", "mes", "es_finde", "temperatura"]
df[features] = df[features].astype(float)
df["kWh"] = pd.to_numeric(df["kWh"], errors="coerce")

print(f"Variables exógenas: {features} | Horizonte rolling: {HORIZONTE} h")

print("\n--------- PASO 2: Entrenando SARIMAX (auto_arima + rolling 24 h) ---------")

fecha_corte = pd.to_datetime("2021-06-05")
fecha_subperiodo = pd.to_datetime("2021-12-31")

lista_resultados = []
lista_metricas = []
tscv = TimeSeriesSplit(n_splits=5)

for cups in df["cups_id"].unique():
    inicio_casa = time.time()  
    print(f"\n Procesando casa: {cups}...")

    df_cups = df[df["cups_id"] == cups].sort_values("timestamp").copy()
    ciudad_actual = df_cups["ciudad_asignada"].iloc[0]

    df_cups.set_index("timestamp", inplace=True)
    df_cups = df_cups[~df_cups.index.duplicated(keep="first")]
    df_cups = df_cups.asfreq("h").ffill()

    train = df_cups[df_cups.index < fecha_corte]
    test = df_cups[df_cups.index >= fecha_corte]

    if len(train) < 100 or len(test) == 0:
        print(f"   Advertencia: datos insuficientes para {cups}.")
        continue

    # Selección de orden con auto_arima sobre el primer pliegue
    train_fold1_idx, _ = list(tscv.split(train))[0]
    train_fold1 = train.iloc[train_fold1_idx]

    modelo_auto = pm.auto_arima(
        train_fold1["kWh"],
        X=train_fold1[features],
        seasonal=True, m=24,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=2, max_q=2, max_P=1, max_Q=1,
    )
    best_order = modelo_auto.order
    best_seasonal_order = modelo_auto.seasonal_order
    print(f"   Estructura fijada: {best_order} x {best_seasonal_order}")

    # Ajuste final sobre todo el train (parámetros fijos a partir de aquí)
    model = SARIMAX(
        endog=train["kWh"],
        exog=train[features],
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    # Predicción rolling: predecir 24 h -> incorporar consumo real -> repetir
    print(f"   Predicción rolling sobre {len(test)} horas de test...")
    bloques_pred = []
    n_test = len(test)
    for i in range(0, n_test, HORIZONTE):
        bloque = test.iloc[i:i + HORIZONTE]
        pred = res.forecast(steps=len(bloque), exog=bloque[features].values)
        bloques_pred.append(np.asarray(pred))

        if i + HORIZONTE < n_test:
            # Actualiza el estado con el consumo real sin reestimar parámetros
            res = res.extend(endog=bloque["kWh"], exog=bloque[features])

    predicciones = np.clip(np.concatenate(bloques_pred), a_min=0, a_max=None)

    df_res_cups = pd.DataFrame({
        "cups_id": cups,
        "ciudad": ciudad_actual,
        "timestamp": test.index,
        "Consumo_Real": test["kWh"].values,
        "Prediccion": predicciones,
    })
    lista_resultados.append(df_res_cups)

    sub_A = df_res_cups[df_res_cups["timestamp"] <= fecha_subperiodo]
    sub_B = df_res_cups[df_res_cups["timestamp"] > fecha_subperiodo]

    y_real = df_res_cups["Consumo_Real"]
    y_pred = df_res_cups["Prediccion"]

    fin_casa = time.time() 
    latencia_s = fin_casa - inicio_casa

    lista_metricas.append({
        "cups_id": cups,
        "ciudad_asignada": ciudad_actual,
        "mejor_orden": str(best_order),
        "mejor_estacional": str(best_seasonal_order),
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
           "mejor_orden", "mejor_estacional"]].to_csv(ruta_metricas, index=False)
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

print("\n[HIPERPARÁMETROS MÁS COMUNES ELEGIDOS POR AUTO_ARIMA]")
print("Orden (p,d,q):\n", df_metricas["mejor_orden"].value_counts())
print("Orden estacional (P,D,Q,m):\n", df_metricas["mejor_estacional"].value_counts())

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