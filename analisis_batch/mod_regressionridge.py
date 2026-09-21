import warnings
import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

NOMBRE_MODELO = "ridge"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
ARCHIVO_DATASET = BASE_DIR / "preprocesamiento" / "dataset_final_IA.csv"

DIR_METRICAS = SCRIPT_DIR / "metricas" / NOMBRE_MODELO
DIR_PREDICCIONES = SCRIPT_DIR / "predicciones" / NOMBRE_MODELO

DIR_MODELOS_PROTOTIPO = BASE_DIR / "aplicacion_final" / "modelos" / "batch" / "ridge_modelos"
DIR_BATCH_PROTOTIPO = BASE_DIR / "aplicacion_final" / "modelos" / "batch"
ARCHIVO_DATASET_FEATURES = DIR_BATCH_PROTOTIPO / "dataset_features.csv"

DIR_METRICAS.mkdir(parents=True, exist_ok=True)
DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)
DIR_MODELOS_PROTOTIPO.mkdir(parents=True, exist_ok=True)
DIR_BATCH_PROTOTIPO.mkdir(parents=True, exist_ok=True) 

if not ARCHIVO_DATASET.exists():
    print(f"Error: No se encuentra '{ARCHIVO_DATASET}'.")
    exit()

# Carga y preprocesamiento 
print(f"--------- PASO 1: Cargando datos y procesando variables para Ridge ---------")

df = pd.read_csv(ARCHIVO_DATASET)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["cups_id", "timestamp"]).reset_index(drop=True)

# Características temporales
df["hora"]       = df["timestamp"].dt.hour
df["dia_semana"] = df["timestamp"].dt.dayofweek
df["mes"]        = df["timestamp"].dt.month
df["es_finde"]   = (df["dia_semana"] >= 5).astype(int)

# Lags y promedios móviles
g = df.groupby("cups_id")["kWh"]
df["lag_24h"]           = g.shift(24)
df["lag_168h"]          = g.shift(168)
df["rolling_24h_media"] = df.groupby("cups_id")["lag_24h"].transform(lambda x: x.rolling(24, min_periods=24).mean())
df["rolling_7d_media"]  = df.groupby("cups_id")["lag_24h"].transform(lambda x: x.rolling(168, min_periods=168).mean())

features = [
    "hora", "dia_semana", "mes", "es_finde", 
    "lag_24h", "lag_168h", 
    "rolling_24h_media", "rolling_7d_media", "temperatura"
]
target = "kWh"

# Eliminar NaNs generados por los lags
df = df.dropna(subset=features).reset_index(drop=True)
df.to_csv(ARCHIVO_DATASET_FEATURES, index=False)
print(f"'dataset_features.csv' exportado con éxito en: {ARCHIVO_DATASET_FEATURES}")

print("Dataset preparado y variables exógenas/lags calculadas.")

# Entrenamiento
print("\n--------- PASO 2: Entrenando, optimizando y evaluando ---------")

fecha_corte = pd.to_datetime("2021-06-05")
fecha_subperiodo = pd.to_datetime("2021-12-31")

lista_resultados = []
lista_metricas = []
lista_coeficientes = []  # <<< NUEVO: Guardar coeficientes XAI
casas_a_procesar = df["cups_id"].unique()

# Configuración de GridSearchCV 
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0] 
}

for cups in casas_a_procesar:
    df_cups = df[df["cups_id"] == cups].copy()
    ciudad_actual = df_cups["ciudad_asignada"].iloc[0]
    
    # División train / test 
    train = df_cups[df_cups["timestamp"] < fecha_corte]
    test  = df_cups[df_cups["timestamp"] >= fecha_corte]
    
    if len(train) < 100 or len(test) == 0:
        print(f" Saltando casa {cups}: Datos insuficientes.")
        continue
        
    X_train, y_train = train[features], train[target]
    X_test,  y_test  = test[features],  test[target]
    
    # Búsqueda de hiperparámetros 
    modelo_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(random_state=42))
    ])
    
    grid_search = GridSearchCV(
        estimator=modelo_pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    mejor_modelo = grid_search.best_estimator_
    mejor_alpha = grid_search.best_params_["ridge__alpha"]
    
    ruta_modelo_joblib = DIR_MODELOS_PROTOTIPO / f"ridge_{cups}.joblib"
    joblib.dump(mejor_modelo, ruta_modelo_joblib)
    
    # --- NUEVO (XAI): Extraer Coeficientes Estandarizados del Modelo Ridge ---
    coefs = mejor_modelo.named_steps["ridge"].coef_
    dict_coefs = {"cups_id": cups, "ciudad": ciudad_actual}
    for feat, coef in zip(features, coefs):
        dict_coefs[feat] = coef
    lista_coeficientes.append(dict_coefs)
    # --------------------------------------------------------------------------

    # Evaluación final
    y_pred = mejor_modelo.predict(X_test)
    y_pred = np.clip(y_pred, a_min=0, a_max=None) # Evitar consumos negativos
    
    # Guardar predicciones
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
        "mejor_alpha": mejor_alpha,
        "mae_total": mae_total, 
        "mae_sub_A": mae_A,
        "mae_sub_B": mae_B,
        "rmse": rmse_total, 
        "wape": wape_total, 
        "r2": r2_total
    })

print(f"\n--------- PASO 3: Resultados consolidados Ridge ---------")
print(f"Casas procesadas y modelos exportados: {len(lista_metricas)}")
print(f"Modelos listos para el prototipo en: {DIR_MODELOS_PROTOTIPO}")

df_metricas = pd.DataFrame(lista_metricas)

if len(lista_resultados) > 0:
    df_resultados = pd.concat(lista_resultados).reset_index(drop=True)
    ruta_predicciones = DIR_PREDICCIONES / "predicciones_ridge.csv"
    df_resultados.to_csv(ruta_predicciones, index=False)
    print(f"Predicciones de test exportadas en: {ruta_predicciones}")

df_export_metricas = df_metricas.rename(columns={
    "ciudad_asignada": "ciudad",
    "mae_total":  "MAE",
    "mae_sub_A":  "MAE_A",
    "mae_sub_B":  "MAE_B",
    "rmse": "RMSE",
    "wape": "WAPE",
    "r2":   "R2",
})

ruta_metricas = DIR_METRICAS / "metricas_ridge.csv"
df_export_metricas[["cups_id", "ciudad", "MAE", "MAE_A", "MAE_B", "RMSE", "WAPE", "R2", "mejor_alpha"]].to_csv(ruta_metricas, index=False)
print(f"Resumen de métricas de test exportado en: {ruta_metricas}")

print("\n[RENDIMIENTO GLOBAL EN TEST (1 AÑO)]")
print(f"  MAE:  {df_metricas['mae_total'].mean():.4f} ± {df_metricas['mae_total'].std():.4f} kWh")
print(f"  RMSE: {df_metricas['rmse'].mean():.4f} ± {df_metricas['rmse'].std():.4f} kWh")
print(f"  WAPE: {df_metricas['wape'].mean():.2f} ± {df_metricas['wape'].std():.2f} %")
print(f"  R²:   {df_metricas['r2'].mean()*100:.2f} ± {df_metricas['r2'].std()*100:.2f} %")

print("\n[ESTABILIDAD TEMPORAL EN TEST]")
mae_A_mean = df_metricas['mae_sub_A'].mean()
mae_B_mean = df_metricas['mae_sub_B'].mean()
degradacion = ((mae_B_mean - mae_A_mean) / mae_A_mean) * 100
print(f"  MAE Semestre A (Jun-Dic 21): {mae_A_mean:.4f} kWh")
print(f"  MAE Semestre B (Ene-Jun 22): {mae_B_mean:.4f} kWh")
signo = "+" if degradacion > 0 else ""
print(f"  Degradación del modelo:  {signo}{degradacion:.2f}% de error")

# Interpretación de coeficientes
print(f"\n--------- PASO 4: Interpretando coeficientes del modelo Ridge ---------")

df_coefs = pd.DataFrame(lista_coeficientes)
ruta_coefs = DIR_METRICAS / "coeficientes_ridge.csv"
df_coefs.to_csv(ruta_coefs, index=False)

# Media y dispersión entre las 12 viviendas 
resumen_coefs = df_coefs[features].agg(["mean", "std"]).T
resumen_coefs.columns = ["media", "desv_std"]
resumen_coefs = resumen_coefs.sort_values("media", key=abs, ascending=False)

# Variable dominante por vivienda (mayor coeficiente)
coefs_abs = df_coefs[features].abs()
df_coefs["variable_dominante"] = coefs_abs.idxmax(axis=1)
conteo_dominante = df_coefs["variable_dominante"].value_counts()

print("\n[IMPACTO PROMEDIO DE CADA VARIABLE (media ± desv. std entre 12 viviendas)]")
for var_name, row in resumen_coefs.iterrows():
    print(f"  - {var_name:20s}: {row['media']:+.4f} ± {row['desv_std']:.4f} kWh (por desv. estándar)")

print("\n[VARIABLE DOMINANTE POR VIVIENDA]")
for var_name, count in conteo_dominante.items():
    print(f"  - {var_name:20s}: dominante en {count}/{len(df_coefs)} viviendas")

resumen_coefs.to_csv(DIR_METRICAS / "resumen_coeficientes_ridge.csv")

# Gráfico con media con barra de error (desviación entre viviendas)
plt.figure(figsize=(10, 6))
orden = resumen_coefs.sort_values("media").index
medias = resumen_coefs.loc[orden, "media"]
errores = resumen_coefs.loc[orden, "desv_std"]
colores = ["#d9534f" if v < 0 else "#0275d8" for v in medias]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(orden, medias, xerr=errores, color=colores, capsize=4)
ax.axvline(0, color="black", linestyle="--", linewidth=1.2)

ax.set_title("Importancia y dirección de las variables en Ridge", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Coeficiente estandarizado (media ± desv. std entre viviendas)", fontsize=11)
ax.set_ylabel("Variables de entrada", fontsize=11)

for i, (media, err) in enumerate(zip(medias, errores)):
    offset = 0.01 if media >= 0 else -0.01
    ha = "left" if media >= 0 else "right"
    ax.annotate(f"{media:+.3f}", (media + offset, i), va="center", ha=ha,
                fontsize=10, color="black", fontweight="bold")

plt.tight_layout()
ruta_grafico = DIR_METRICAS / "coeficientes_ridge.png"
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print(f"\nGráfico exportado en: {ruta_grafico}")
print(f"CSV de coeficientes por vivienda exportado en: {ruta_coefs}")
print(f"CSV resumen (media/std/dominancia) exportado en: {DIR_METRICAS / 'resumen_coeficientes_ridge.csv'}")

 # % de viviendas con coeficiente positivo
signos = (df_coefs[features] > 0).mean() * 100 
print("\n[% DE VIVIENDAS CON COEFICIENTE POSITIVO]")
for var_name, pct in signos.sort_values(ascending=False).items():
    print(f"  - {var_name:20s}: {pct:.0f}% positivo")