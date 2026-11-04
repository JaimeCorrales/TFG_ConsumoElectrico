import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time 
from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

ARCHIVO_DATASET = BASE_DIR / "preprocesamiento" / "dataset_final_IA.csv"

if not ARCHIVO_DATASET.exists():
    print(f"Error: No se encuentra '{ARCHIVO_DATASET}'")
    exit()

NOMBRE_MODELO = "xgboost"

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

g = df.groupby("cups_id")["kWh"]
df["lag_24h"]           = g.shift(24)
df["lag_168h"]          = g.shift(168)
df["rolling_24h_media"] = df.groupby("cups_id")["lag_24h"].transform(lambda x: x.rolling(24, min_periods=24).mean())
df["rolling_7d_media"]  = df.groupby("cups_id")["lag_24h"].transform(lambda x: x.rolling(168, min_periods=168).mean())

features = ["hora", "dia_semana", "mes", "es_finde", "lag_24h", "lag_168h", "rolling_24h_media", "rolling_7d_media", "temperatura"]
target = "kWh"

df = df.dropna(subset=features).reset_index(drop=True)
print("Dataset preparado.")

print("\n--------- PASO 2: Entrenando, optimizando y evaluando ---------")

fecha_corte = pd.to_datetime("2021-06-05")
fecha_subperiodo = pd.to_datetime("2021-12-31")

lista_resultados = []
lista_metricas = []
casas_a_procesar = df["cups_id"].unique()

casas_saltadas = 0

tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'n_estimators': [150, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.03, 0.05]
}

tiempo_inicio_total = time.time()

for cups in casas_a_procesar:
    df_cups = df[df["cups_id"] == cups].copy()
    ciudad_actual = df_cups["ciudad_asignada"].iloc[0]
    
    train = df_cups[df_cups["timestamp"] < fecha_corte]
    test  = df_cups[df_cups["timestamp"] >= fecha_corte]
    
    if len(train) < 500 or len(test) == 0:
        casas_saltadas += 1
        print(f"  Saltada: {cups} (train={len(train)}, test={len(test)})")
        continue
        
    X_train, y_train = train[features], train[target]
    X_test,  y_test  = test[features],  test[target]
    
    tiempo_inicio_casa = time.time()
    
    modelo_base = xgb.XGBRegressor(
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1  
    )
    
    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    mejor_modelo = grid_search.best_estimator_
    
    y_pred = mejor_modelo.predict(X_test)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    
    tiempo_fin_casa = time.time()
    latencia_casa = tiempo_fin_casa - tiempo_inicio_casa
    
    lista_resultados.append(pd.DataFrame({
        "cups_id":      cups,
        "ciudad":       ciudad_actual,
        "timestamp":    test["timestamp"].values,
        "Consumo_Real": y_test.values,
        "Prediccion":   y_pred
    }))
    
    test_res = test.copy()
    test_res["Consumo_Real"] = y_test.values
    test_res["Prediccion"] = y_pred
    
    sub_A = test_res[test_res["timestamp"] <= fecha_subperiodo]
    sub_B = test_res[test_res["timestamp"] > fecha_subperiodo]
    
    mae_total = mean_absolute_error(test_res["Consumo_Real"], test_res["Prediccion"])
    mae_A = mean_absolute_error(sub_A["Consumo_Real"], sub_A["Prediccion"]) if len(sub_A) > 0 else np.nan
    mae_B = mean_absolute_error(sub_B["Consumo_Real"], sub_B["Prediccion"]) if len(sub_B) > 0 else np.nan
    
    rmse_total = np.sqrt(mean_squared_error(test_res["Consumo_Real"], test_res["Prediccion"]))
    wape_total = (np.sum(np.abs(test_res["Consumo_Real"] - test_res["Prediccion"])) / np.sum(test_res["Consumo_Real"])) * 100
    r2_total = r2_score(test_res["Consumo_Real"], test_res["Prediccion"])
    
    lista_metricas.append({
        "cups_id": cups, 
        "ciudad_asignada": ciudad_actual, 
        "best_n_estimators": grid_search.best_params_["n_estimators"],
        "best_max_depth":    grid_search.best_params_["max_depth"],
        "best_lr":           grid_search.best_params_["learning_rate"],
        "latencia_segundos": latencia_casa, 
        "mae_total": mae_total, 
        "mae_sub_A": mae_A,
        "mae_sub_B": mae_B,
        "rmse": rmse_total, 
        "wape": wape_total, 
        "r2": r2_total
    })

tiempo_fin_total = time.time()
latencia_total_script = tiempo_fin_total - tiempo_inicio_total

print(f"\n--------- PASO 3: Resultados consolidados XGBoost ---------")
print(f"Casas procesadas con éxito: {len(lista_metricas)}")
if casas_saltadas > 0:
    print(f"Casas saltadas por falta de datos: {casas_saltadas}/{len(casas_a_procesar)}")

df_metricas = pd.DataFrame(lista_metricas)

if len(lista_resultados) > 0:
    df_resultados = pd.concat(lista_resultados).reset_index(drop=True)
    ruta_predicciones = DIR_PREDICCIONES / "predicciones_xgboost.csv"
    df_resultados.to_csv(ruta_predicciones, index=False)
    print(f"Exportado historial de predicciones en: {ruta_predicciones}")

df_export_metricas = df_metricas.rename(columns={
    "ciudad_asignada": "ciudad",
    "latencia_segundos": "Latencia_s", 
    "mae_total":  "MAE",
    "mae_sub_A":  "MAE_A",
    "mae_sub_B":  "MAE_B",
    "rmse": "RMSE",
    "wape": "WAPE",
    "r2":   "R2",
})

ruta_metricas = DIR_METRICAS / "metricas_xgboost.csv"
df_export_metricas[["cups_id", "ciudad", "MAE", "MAE_A", "MAE_B", "RMSE", "WAPE", "R2", "Latencia_s", "best_n_estimators", "best_max_depth", "best_lr"]].to_csv(ruta_metricas, index=False)
print(f"Exportado resumen de métricas en: {ruta_metricas}")

print("\n[RENDIMIENTO GLOBAL (1 AÑO)]")
print(f"  MAE:  {df_metricas['mae_total'].mean():.4f} ± {df_metricas['mae_total'].std():.4f} kWh")
print(f"  RMSE: {df_metricas['rmse'].mean():.4f} ± {df_metricas['rmse'].std():.4f} kWh")
print(f"  WAPE: {df_metricas['wape'].mean():.2f} ± {df_metricas['wape'].std():.2f} %")
print(f"  R²:   {df_metricas['r2'].mean()*100:.2f} ± {df_metricas['r2'].std()*100:.2f} %")

print("\n[COSTE COMPUTACIONAL (LATENCIA)]")
print(f"  Latencia media por vivienda (GridSearch + Fit + Predict): {df_metricas['latencia_segundos'].mean():.4f} segundos")
print(f"  Latencia total del proceso ({len(lista_metricas)} viviendas): {latencia_total_script:.2f} segundos")

print("\n[ESTABILIDAD TEMPORAL]")
print(f"  MAE Semestre A (Jun-Dic 21): {df_metricas['mae_sub_A'].mean():.4f} kWh")
print(f"  MAE Semestre B (Ene-Jun 22): {df_metricas['mae_sub_B'].mean():.4f} kWh")
degradacion = ((df_metricas['mae_sub_B'].mean() - df_metricas['mae_sub_A'].mean()) / df_metricas['mae_sub_A'].mean()) * 100

if degradacion > 0:
    print(f"  Degradación del modelo:  +{degradacion:.2f}% de error")
elif degradacion < 0:
    print(f"  Mejora del modelo:       {degradacion:.2f}% de error")
else:
    print(f"  Estabilidad perfecta:    0.00% de variación entre subperíodos")

print("\n[HIPERPARÁMETROS MÁS COMUNES ELEGIDOS POR EL GRIDSEARCH]")
print("N_Estimators:\n", df_metricas['best_n_estimators'].value_counts())
print("Max Depth:\n", df_metricas['best_max_depth'].value_counts())
print("Learning Rate:\n", df_metricas['best_lr'].value_counts())

print("\n[RENDIMIENTO POR CIUDAD]")
resumen_ciudad = (
    df_metricas
    .groupby("ciudad_asignada")
    .agg(
        MAE_media  = ("mae_total", "mean"),
        MAE_std    = ("mae_total", "std"),
        RMSE_media = ("rmse",      "mean"),
        WAPE_media = ("wape",      "mean"),
        R2_media   = ("r2",        "mean"),
        n_casas    = ("cups_id",   "count"),
    )
    .round(4)
)
print(resumen_ciudad.to_string())

print("\n[ESTABILIDAD TEMPORAL POR CIUDAD]")
estabilidad_ciudad = (
    df_metricas
    .groupby("ciudad_asignada")
    .agg(
        MAE_sub_A = ("mae_sub_A", "mean"),
        MAE_sub_B = ("mae_sub_B", "mean"),
    )
    .round(4)
)
estabilidad_ciudad["Degradacion_pct"] = (
    (estabilidad_ciudad["MAE_sub_B"] - estabilidad_ciudad["MAE_sub_A"]) / 
    estabilidad_ciudad["MAE_sub_A"] * 100
).round(2)
print(estabilidad_ciudad.to_string())

print("\n[CIUDAD MÁS DIFÍCIL Y MÁS FÁCIL PARA XGBOOST]")
ciudad_mae = resumen_ciudad["MAE_media"]
print(f"  Mejor ciudad:  {ciudad_mae.idxmin()} (MAE={ciudad_mae.min():.4f} kWh)")
print(f"  Peor ciudad:   {ciudad_mae.idxmax()} (MAE={ciudad_mae.max():.4f} kWh)")

ruta_ciudad = DIR_METRICAS / "metricas_xgboost_por_ciudad.csv"
ruta_estabilidad = DIR_METRICAS / "estabilidad_xgboost_por_ciudad.csv"

resumen_ciudad.reset_index().to_csv(ruta_ciudad, index=False)
estabilidad_ciudad.reset_index().to_csv(ruta_estabilidad, index=False)
print(f"\n Exportados desgloses en: {ruta_ciudad} y {ruta_estabilidad}")