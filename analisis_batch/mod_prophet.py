import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
sns.set_theme(style="whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

ARCHIVO_DATASET = BASE_DIR / "preprocesamiento" / "dataset_final_IA.csv"

if not ARCHIVO_DATASET.exists():
    print(f" Error: No se encuentra '{ARCHIVO_DATASET}'.")
    exit()

NOMBRE_MODELO = "prophet"

DIR_METRICAS = SCRIPT_DIR / "metricas" / NOMBRE_MODELO
DIR_PREDICCIONES = SCRIPT_DIR / "predicciones" / NOMBRE_MODELO

DIR_METRICAS.mkdir(parents=True, exist_ok=True)
DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)

print(f"--------- PASO 1: Cargando datos y procesando variables para Prophet ---------")
df = pd.read_csv(ARCHIVO_DATASET)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(by=["cups_id", "timestamp"]).reset_index(drop=True)

df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")

print(f"Dataset cargado por completo.")

print("\n--------- PASO 2: Entrenando PROPHET (Casa por Casa) ---------")

fecha_corte = pd.to_datetime("2021-06-05")
fecha_subperiodo = pd.to_datetime("2021-12-31")

lista_resultados = []
lista_metricas = [] 

casas_a_procesar = df["cups_id"].unique()

param_grid = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1],
    'seasonality_prior_scale': [1.0, 10.0]
}

for cups in casas_a_procesar:
    
    df_cups = df[df["cups_id"] == cups].copy()
    ciudad_actual = df_cups["ciudad_asignada"].iloc[0]
    
    df_cups.set_index("timestamp", inplace=True)
    df_cups = df_cups[~df_cups.index.duplicated(keep='first')]
    df_cups = df_cups.asfreq("h").ffill()
    df_cups.reset_index(inplace=True)

    df_cups = df_cups[["timestamp", "kWh", "temperatura"]]
    df_cups.rename(columns={"timestamp": "ds", "kWh": "y"}, inplace=True)
    
    train = df_cups[df_cups["ds"] < fecha_corte].copy()
    test  = df_cups[df_cups["ds"] >= fecha_corte].copy()
    
    if len(train) < 100 or len(test) == 0:
        print(f"  Datos insuficientes para la casa {cups}. Saltando...")
        continue
    
    idx_split = int(len(train) * 0.7)
    train_fold1 = train.iloc[:idx_split]
    val_fold1 = train.iloc[idx_split:]
    
    best_mae = float('inf')
    best_params = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}
    
    for params in ParameterGrid(param_grid):
        m_cv = Prophet(
            daily_seasonality=20, 
            weekly_seasonality=15, 
            yearly_seasonality=True,
            **params
        )
        m_cv.add_regressor('temperatura')
        m_cv.fit(train_fold1)
        
        pred_cv = m_cv.predict(val_fold1[["ds", "temperatura"]])
        mae_cv = mean_absolute_error(val_fold1["y"], pred_cv["yhat"])
        
        if mae_cv < best_mae:
            best_mae = mae_cv
            best_params = params
            
    print(f" Procesando casa: {cups} -> Parámetros óptimos: {best_params}")

    model = Prophet(        
        daily_seasonality=20, 
        weekly_seasonality=15, 
        yearly_seasonality=True,
        **best_params 
    )
    model.add_regressor('temperatura')
    model.fit(train)
    
    prediccion_bruta = model.predict(test[["ds", "temperatura"]])
    
    y_pred = prediccion_bruta["yhat"].clip(lower=0).values
    y_real = test["y"].values
    
    df_res_cups = pd.DataFrame({
        "cups_id":      cups,
        "ciudad":       ciudad_actual,
        "timestamp":    test["ds"].values,
        "Consumo_Real": y_real,
        "Prediccion":   y_pred
    })
    lista_resultados.append(df_res_cups)

    sub_A = df_res_cups[df_res_cups["timestamp"] <= fecha_subperiodo]
    sub_B = df_res_cups[df_res_cups["timestamp"] > fecha_subperiodo]

    mae_total = mean_absolute_error(y_real, y_pred)
    mae_A = mean_absolute_error(sub_A["Consumo_Real"], sub_A["Prediccion"]) if len(sub_A) > 0 else np.nan
    mae_B = mean_absolute_error(sub_B["Consumo_Real"], sub_B["Prediccion"]) if len(sub_B) > 0 else np.nan
    rmse_total = np.sqrt(mean_squared_error(y_real, y_pred))
    wape_total = (np.sum(np.abs(y_real - y_pred)) / np.sum(y_real)) * 100
    r2_total = r2_score(y_real, y_pred)
    
    lista_metricas.append({
        "cups_id": cups, 
        "ciudad_asignada": ciudad_actual, 
        "changepoint_prior_scale": best_params['changepoint_prior_scale'],
        "seasonality_prior_scale": best_params['seasonality_prior_scale'],
        "mae_total": mae_total, 
        "mae_sub_A": mae_A,
        "mae_sub_B": mae_B,
        "rmse": rmse_total, 
        "wape": wape_total, 
        "r2": r2_total
    })

print(f"\n--------- PASO 3: Resultados consolidados PROPHET ---------")
print(f"Casas procesadas con éxito: {len(lista_metricas)}")

df_metricas = pd.DataFrame(lista_metricas)

if len(lista_resultados) > 0:
    df_resultados = pd.concat(lista_resultados).reset_index(drop=True)
    ruta_predicciones = DIR_PREDICCIONES / "predicciones_prophet.csv"
    df_resultados.to_csv(ruta_predicciones, index=False)
    print(f"Exportado historial de predicciones en: {ruta_predicciones}")

df_export_metricas = df_metricas.rename(columns={
    "ciudad_asignada": "ciudad",
    "mae_total":  "MAE",
    "mae_sub_A":  "MAE_A",
    "mae_sub_B":  "MAE_B",
    "rmse": "RMSE",
    "wape": "WAPE",
    "r2":   "R2",
})

ruta_metricas = DIR_METRICAS / "metricas_prophet.csv"
df_export_metricas[[
    "cups_id", "ciudad", "MAE", "MAE_A", "MAE_B", "RMSE", "WAPE", "R2", 
    "changepoint_prior_scale", "seasonality_prior_scale"
]].to_csv(ruta_metricas, index=False)
print(f"Exportado resumen de métricas en: {ruta_metricas}")

print("\n[RENDIMIENTO GLOBAL (1 AÑO)]")
print(f"  MAE:  {df_metricas['mae_total'].mean():.4f} ± {df_metricas['mae_total'].std():.4f} kWh")
print(f"  RMSE: {df_metricas['rmse'].mean():.4f} ± {df_metricas['rmse'].std():.4f} kWh")
print(f"  WAPE: {df_metricas['wape'].mean():.2f} ± {df_metricas['wape'].std():.2f} %")
print(f"  R²:   {df_metricas['r2'].mean()*100:.2f} ± {df_metricas['r2'].std()*100:.2f} %")

print("\n[ESTABILIDAD TEMPORAL]")
mae_A_mean = df_metricas['mae_sub_A'].mean()
mae_B_mean = df_metricas['mae_sub_B'].mean()
degradacion = ((mae_B_mean - mae_A_mean) / mae_A_mean) * 100
print(f"  MAE Semestre A (Jun-Dic 21): {mae_A_mean:.4f} kWh")
print(f"  MAE Semestre B (Ene-Jun 22): {mae_B_mean:.4f} kWh")
signo = "+" if degradacion > 0 else ""
print(f"  Degradación del modelo:  {signo}{degradacion:.2f}% de error")

print("\n[HIPERPARÁMETROS MÁS COMUNES ELEGIDOS POR EL GRIDSEARCH]")
print("Changepoint Prior Scale:\n", df_metricas['changepoint_prior_scale'].value_counts())
print("Seasonality Prior Scale:\n", df_metricas['seasonality_prior_scale'].value_counts())

print("\n[RENDIMIENTO POR CIUDAD]")
resumen_ciudad = (
    df_metricas
    .groupby("ciudad_asignada")
    .agg(
        MAE_media  = ("mae_total", "mean"),
        MAE_std    = ("mae_total", "std"),
        RMSE_media = ("rmse",      "mean"),
        WAPE_media = ("wape",      "mean"),
        R2_media   = ("r2",        lambda x: x.mean() * 100),
        n_casas    = ("cups_id",   "count"),
    )
    .round(4)
)
print(resumen_ciudad.to_string())

print("\n[ESTABILIDAD TEMPORAL POR CIUDAD]")
est_ciudad = (
    df_metricas
    .groupby("ciudad_asignada")
    .agg(
        MAE_sub_A = ("mae_sub_A", "mean"),
        MAE_sub_B = ("mae_sub_B", "mean"),
    )
    .round(4)
)
est_ciudad["Degradacion_pct"] = (
    (est_ciudad["MAE_sub_B"] - est_ciudad["MAE_sub_A"]) / 
    est_ciudad["MAE_sub_A"] * 100
).round(2)
print(est_ciudad.to_string())

print("\n[CIUDAD MÁS DIFÍCIL Y MÁS FÁCIL PARA PROPHET]")
mejor_ciudad = resumen_ciudad["MAE_media"].idxmin()
peor_ciudad = resumen_ciudad["MAE_media"].idxmax()
print(f"  Mejor ciudad:  {mejor_ciudad} (MAE={resumen_ciudad.loc[mejor_ciudad, 'MAE_media']:.4f} kWh)")
print(f"  Peor ciudad:   {peor_ciudad} (MAE={resumen_ciudad.loc[peor_ciudad, 'MAE_media']:.4f} kWh)")

ruta_ciudad = DIR_METRICAS / "metricas_prophet_por_ciudad.csv"
ruta_estabilidad = DIR_METRICAS / "estabilidad_prophet_por_ciudad.csv"

resumen_ciudad.reset_index().to_csv(ruta_ciudad, index=False)
est_ciudad.reset_index().to_csv(ruta_estabilidad, index=False)
print(f"\n Exportados desgloses en: {ruta_ciudad} y {ruta_estabilidad}")