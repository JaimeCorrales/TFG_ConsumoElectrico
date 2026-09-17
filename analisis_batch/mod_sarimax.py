import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pmdarima as pm  
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

ARCHIVO_DATASET = BASE_DIR / "preprocesamiento" / "dataset_final_IA.csv"

if not ARCHIVO_DATASET.exists():
    print(f" Error: No se encuentra '{ARCHIVO_DATASET}'.")
    exit()

NOMBRE_MODELO = "sarimax"

DIR_METRICAS = SCRIPT_DIR / "metricas" / NOMBRE_MODELO
DIR_PREDICCIONES = SCRIPT_DIR / "predicciones" / NOMBRE_MODELO

DIR_METRICAS.mkdir(parents=True, exist_ok=True)
DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)

print(f"--------- PASO 1: Cargando datos y procesando variables para {NOMBRE_MODELO.upper()} ---------")

df = pd.read_csv(ARCHIVO_DATASET)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["cups_id", "timestamp"]).reset_index(drop=True)

df["hora"]       = df["timestamp"].dt.hour
df["dia_semana"] = df["timestamp"].dt.dayofweek
df["mes"]        = df["timestamp"].dt.month
df["es_finde"]   = (df["dia_semana"] >= 5).astype(int)

features = ["hora", "dia_semana", "mes", "es_finde", "temperatura"]

df[features] = df[features].astype(float)
df["kWh"] = pd.to_numeric(df["kWh"], errors="coerce")

print(f"Dataset preparado. Variables exógenas asignadas: {features}")

print("\n--------- PASO 2: Entrenando SARIMAX con Auto-ARIMA ---------")

fecha_corte = pd.to_datetime("2021-06-05")
fecha_subperiodo = pd.to_datetime("2021-12-31") 

lista_resultados = []
lista_metricas = [] 

casas_a_procesar = df["cups_id"].unique()

tscv = TimeSeriesSplit(n_splits=5)

for cups in casas_a_procesar:
    print(f"\n Procesando casa: {cups}...")
    
    df_cups = df[df["cups_id"] == cups].sort_values("timestamp").copy()
    
    ciudad_actual = df_cups["ciudad_asignada"].iloc[0]
    
    df_cups.set_index("timestamp", inplace=True)
    
    df_cups = df_cups[~df_cups.index.duplicated(keep='first')]
    df_cups = df_cups.asfreq("h").ffill()
    
    train = df_cups[df_cups.index < fecha_corte]
    test = df_cups[df_cups.index >= fecha_corte]
    
    if len(train) < 100 or len(test) == 0:
        print(f"   Advertencia: Datos insuficientes para {cups}.")
        continue
        
    splits = list(tscv.split(train))
    train_fold1_idx, _ = splits[0]
    train_fold1 = train.iloc[train_fold1_idx]
    
    print(f"   Buscando hiperparámetros con auto_arima en el pliegue 1 ({len(train_fold1)} horas)...")
    
    modelo_auto = pm.auto_arima(
        train_fold1["kWh"],
        X=train_fold1[features],
        seasonal=True, m=24,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=2, max_q=2, max_P=1, max_Q=1
    )
    
    best_order = modelo_auto.order
    best_seasonal_order = modelo_auto.seasonal_order
    print(f"   Estructura fijada: Orden {best_order}, Estacional {best_seasonal_order}")

    print("   Entrenando modelo SARIMAX final sobre todo el train...")
    model = SARIMAX(
        endog=train["kWh"],
        exog=train[features],
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    model_fitted = model.fit(disp=False)
    
    predicciones = model_fitted.predict(
        start=test.index[0],
        end=test.index[-1],
        exog=test[features]
    )
    
    predicciones = np.clip(predicciones, a_min=0, a_max=None)
    
    df_res_cups = pd.DataFrame({
        "cups_id":      cups,
        "ciudad":       ciudad_actual,
        "timestamp":    test.index,
        "Consumo_Real": test["kWh"].values,
        "Prediccion":   predicciones.values
    })
    lista_resultados.append(df_res_cups)

    sub_A = df_res_cups[df_res_cups["timestamp"] <= fecha_subperiodo]
    sub_B = df_res_cups[df_res_cups["timestamp"] > fecha_subperiodo]

    mae_total = mean_absolute_error(df_res_cups["Consumo_Real"], df_res_cups["Prediccion"])
    mae_A = mean_absolute_error(sub_A["Consumo_Real"], sub_A["Prediccion"]) if len(sub_A) > 0 else np.nan
    mae_B = mean_absolute_error(sub_B["Consumo_Real"], sub_B["Prediccion"]) if len(sub_B) > 0 else np.nan
    
    rmse_total = np.sqrt(mean_squared_error(df_res_cups["Consumo_Real"], df_res_cups["Prediccion"]))
    wape_total = (np.sum(np.abs(df_res_cups["Consumo_Real"] - df_res_cups["Prediccion"])) / np.sum(df_res_cups["Consumo_Real"])) * 100
    r2_total = r2_score(df_res_cups["Consumo_Real"], df_res_cups["Prediccion"])
    
    lista_metricas.append({
        "cups_id": cups, 
        "ciudad_asignada": ciudad_actual, 
        "mejor_orden": str(best_order),
        "mejor_estacional": str(best_seasonal_order),
        "mae_total": mae_total, 
        "mae_sub_A": mae_A,
        "mae_sub_B": mae_B,
        "rmse": rmse_total, 
        "wape": wape_total, 
        "r2": r2_total
    })

print(f"\n--------- PASO 3: Resultados Consolidados SARIMAX ---------")
print(f"Casas procesadas con éxito: {len(lista_metricas)}")

df_metricas = pd.DataFrame(lista_metricas)

if len(lista_resultados) > 0:
    df_resultados = pd.concat(lista_resultados).reset_index(drop=True)
    ruta_predicciones = DIR_PREDICCIONES / "predicciones_sarimax.csv"
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

ruta_metricas = DIR_METRICAS / "metricas_sarimax.csv"
df_export_metricas[[
    "cups_id", "ciudad", "MAE", "MAE_A", "MAE_B", "RMSE", "WAPE", "R2", 
    "mejor_orden", "mejor_estacional"
]].to_csv(ruta_metricas, index=False)
print(f"Exportado resumen de métricas en: {ruta_metricas}")

print("\n[RENDIMIENTO GLOBAL (1 AÑO)]")
print(f"  MAE:  {df_metricas['mae_total'].mean():.4f} ± {df_metricas['mae_total'].std():.4f} kWh")
print(f"  RMSE: {df_metricas['rmse'].mean():.4f} ± {df_metricas['rmse'].std():.4f} kWh")
print(f"  WAPE: {df_metricas['wape'].mean():.2f} ± {df_metricas['wape'].std():.2f} %")
print(f"  R²:   {df_metricas['r2'].mean()*100:.2f} ± {df_metricas['r2'].std()*100:.2f} %")

print("\n[ESTABILIDAD TEMPORAL (Análisis de Concept Drift)]")
mae_A_mean = df_metricas['mae_sub_A'].mean()
mae_B_mean = df_metricas['mae_sub_B'].mean()
degradacion = ((mae_B_mean - mae_A_mean) / mae_A_mean) * 100
print(f"  MAE Semestre A (Jun-Dic 21): {mae_A_mean:.4f} kWh")
print(f"  MAE Semestre B (Ene-Jun 22): {mae_B_mean:.4f} kWh")
signo = "+" if degradacion > 0 else ""
print(f"  Degradación del modelo:  {signo}{degradacion:.2f}% de error")

print("\n[HIPERPARÁMETROS MÁS COMUNES ELEGIDOS POR AUTO_ARIMA]")
print("Orden No Estacional (p,d,q):\n", df_metricas['mejor_orden'].value_counts())
print("Orden Estacional (P,D,Q,m):\n", df_metricas['mejor_estacional'].value_counts())

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

print("\n[CIUDAD MÁS DIFÍCIL Y MÁS FÁCIL PARA SARIMAX]")
mejor_ciudad = resumen_ciudad["MAE_media"].idxmin()
peor_ciudad = resumen_ciudad["MAE_media"].idxmax()
print(f"  Mejor ciudad:  {mejor_ciudad} (MAE={resumen_ciudad.loc[mejor_ciudad, 'MAE_media']:.4f} kWh)")
print(f"  Peor ciudad:   {peor_ciudad} (MAE={resumen_ciudad.loc[peor_ciudad, 'MAE_media']:.4f} kWh)")

ruta_ciudad = DIR_METRICAS / "metricas_sarimax_por_ciudad.csv"
ruta_estabilidad = DIR_METRICAS / "estabilidad_sarimax_por_ciudad.csv"

resumen_ciudad.reset_index().to_csv(ruta_ciudad, index=False)
est_ciudad.reset_index().to_csv(ruta_estabilidad, index=False)
print(f"\n Exportados desgloses en: {ruta_ciudad} y {ruta_estabilidad}")