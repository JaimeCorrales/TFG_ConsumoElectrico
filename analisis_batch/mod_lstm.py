import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import time 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

ARCHIVO_DATASET = BASE_DIR / "preprocesamiento" / "dataset_final_IA.csv"

if not ARCHIVO_DATASET.exists():
    print(f" Error: No se encuentra '{ARCHIVO_DATASET}'.")
    exit()

NOMBRE_MODELO = "lstm"

DIR_METRICAS = SCRIPT_DIR / "metricas" / NOMBRE_MODELO
DIR_PREDICCIONES = SCRIPT_DIR / "predicciones" / NOMBRE_MODELO

DIR_METRICAS.mkdir(parents=True, exist_ok=True)
DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)

print(f"--------- PASO 1: Cargando datos y procesando variables para {NOMBRE_MODELO.upper()} ---------")

df = pd.read_csv(ARCHIVO_DATASET)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["cups_id", "timestamp"]).reset_index(drop=True)

df["hora"]        = df["timestamp"].dt.hour
df["dia_semana"]  = df["timestamp"].dt.dayofweek
df["mes"]         = df["timestamp"].dt.month
df["es_finde"]    = (df["dia_semana"] >= 5).astype(int)
df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")

features = [
    "kWh", "hora", "dia_semana", "mes", "es_finde", "temperatura",
    "lag_168h", "rolling_7d_media"
]

LOOK_BACK = 24 
HORIZONTE = 24 

print(f" Dataset cargado. Ventana: {LOOK_BACK}h. Horizonte de predicción: {HORIZONTE}h.")

def crear_secuencias_mimo(datos, fechas, look_back, horizonte, step=1):
    X, y, target_dates = [], [], []
    
    for i in range(0, len(datos) - look_back - horizonte + 1, step):
        X.append(datos[i:(i + look_back), :])
        y.append(datos[(i + look_back):(i + look_back + horizonte), 0])
        target_dates.append(fechas[(i + look_back):(i + look_back + horizonte)])
    return np.array(X), np.array(y), np.array(target_dates)

print("\n--------- PASO 2: Entrenando LSTM (Casa por Casa) ---------")

fecha_corte = pd.to_datetime("2021-06-05")
fecha_subperiodo = pd.to_datetime("2021-12-31")

lista_resultados = []
lista_metricas = [] 

casas_a_procesar = df["cups_id"].unique()
tiempo_inicio_total = time.time()

for cups in casas_a_procesar:
    print(f" Procesando casa: {cups}...")
    
    df_cups = df[df["cups_id"] == cups].copy()
    ciudad_actual = df_cups["ciudad_asignada"].iloc[0]
    
    df_cups.set_index("timestamp", inplace=True)
    df_cups = df_cups[~df_cups.index.duplicated(keep='first')]
    df_cups = df_cups.asfreq("h").ffill()
    df_cups.reset_index(inplace=True)
    
    df_cups["lag_168h"] = df_cups["kWh"].shift(168)
    df_cups["rolling_7d_media"] = df_cups["kWh"].shift(24).rolling(window=168).mean()
    
    df_cups = df_cups.dropna(subset=features).reset_index(drop=True)
    
    if len(df_cups) < (LOOK_BACK + HORIZONTE + 100):
        continue
        
    train_df = df_cups[df_cups["timestamp"] < fecha_corte].copy()
    test_df_raw = df_cups[df_cups["timestamp"] >= fecha_corte].copy()
    
    if len(train_df) == 0 or len(test_df_raw) == 0:
        continue

    if len(train_df) >= LOOK_BACK:
        test_df = pd.concat([train_df.iloc[-LOOK_BACK:], test_df_raw]).reset_index(drop=True)
    else:
        test_df = test_df_raw

    tiempo_inicio_casa = time.time()

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    train_scaled = scaler_x.fit_transform(train_df[features].values)
    scaler_y.fit(train_df[["kWh"]].values)
    
    test_scaled = scaler_x.transform(test_df[features].values)
    
    X_train, y_train, _ = crear_secuencias_mimo(train_scaled, train_df["timestamp"].values, LOOK_BACK, HORIZONTE, step=1)
    
    X_test, y_test_scaled, dates_test = crear_secuencias_mimo(test_scaled, test_df["timestamp"].values, LOOK_BACK, HORIZONTE, step=HORIZONTE)
    
    if len(X_train) == 0 or len(X_test) == 0:
        continue

    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(LOOK_BACK, len(features))))
    model.add(Dropout(0.2))
    model.add(Dense(HORIZONTE))
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=50,             
        batch_size=32,          
        validation_split=0.1, 
        callbacks=[early_stop],
        verbose=0 
    )
    
    prediccion_scaled = model.predict(X_test, verbose=0)
    
    prediccion_real = scaler_y.inverse_transform(prediccion_scaled.reshape(-1, 1)).flatten()
    consumo_real    = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    tiempo_fin_casa = time.time()
    latencia_casa = tiempo_fin_casa - tiempo_inicio_casa

    dates_test_flat = dates_test.flatten()
    
    pred_limpia = np.clip(prediccion_real, a_min=0, a_max=None)
    
    df_res_cups = pd.DataFrame({
        "cups_id":      cups,
        "ciudad":       ciudad_actual,
        "timestamp":    dates_test_flat,
        "Consumo_Real": consumo_real,
        "Prediccion":   pred_limpia
    })
    lista_resultados.append(df_res_cups)

    sub_A = df_res_cups[df_res_cups["timestamp"] <= fecha_subperiodo]
    sub_B = df_res_cups[df_res_cups["timestamp"] > fecha_subperiodo]

    mae_total = mean_absolute_error(consumo_real, pred_limpia)
    mae_A = mean_absolute_error(sub_A["Consumo_Real"], sub_A["Prediccion"]) if len(sub_A) > 0 else np.nan
    mae_B = mean_absolute_error(sub_B["Consumo_Real"], sub_B["Prediccion"]) if len(sub_B) > 0 else np.nan
    rmse_total = np.sqrt(mean_squared_error(consumo_real, pred_limpia))
    wape_total = (np.sum(np.abs(consumo_real - pred_limpia)) / np.sum(consumo_real)) * 100
    r2_total   = r2_score(consumo_real, pred_limpia)
    
    lista_metricas.append({
        "cups_id": cups, 
        "ciudad_asignada": ciudad_actual, 
        "latencia_segundos": latencia_casa, 
        "mae_total": mae_total, 
        "mae_sub_A": mae_A,
        "mae_sub_B": mae_B,
        "rmse": rmse_total, 
        "wape": wape_total, 
        "r2": r2_total
    })
    
    clear_session()

tiempo_fin_total = time.time()
latencia_total_script = tiempo_fin_total - tiempo_inicio_total

print(f"\n--------- PASO 3: Resultados Consolidados LSTM ---------")
print(f"Casas procesadas con éxito: {len(lista_metricas)}")

df_metricas = pd.DataFrame(lista_metricas)

if len(lista_resultados) > 0:
    df_resultados = pd.concat(lista_resultados).reset_index(drop=True)
    ruta_predicciones = DIR_PREDICCIONES / "predicciones_lstm.csv"
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

ruta_metricas = DIR_METRICAS / "metricas_lstm.csv"
df_export_metricas[["cups_id", "ciudad", "MAE", "MAE_A", "MAE_B", "RMSE", "WAPE", "R2", "Latencia_s"]].to_csv(ruta_metricas, index=False)
print(f"Exportado resumen de métricas en: {ruta_metricas}")

print("\n[RENDIMIENTO GLOBAL (1 AÑO)]")
print(f"  MAE:  {df_metricas['mae_total'].mean():.4f} ± {df_metricas['mae_total'].std():.4f} kWh")
print(f"  RMSE: {df_metricas['rmse'].mean():.4f} ± {df_metricas['rmse'].std():.4f} kWh")
print(f"  WAPE: {df_metricas['wape'].mean():.2f} ± {df_metricas['wape'].std():.2f} %")
print(f"  R²:   {df_metricas['r2'].mean()*100:.2f} ± {df_metricas['r2'].std()*100:.2f} %")

print("\n[COSTE COMPUTACIONAL (LATENCIA)]")
print(f"  Latencia media por vivienda (Scaling + Secuencias + Fit + Predict): {df_metricas['latencia_segundos'].mean():.4f} segundos")
print(f"  Latencia total del proceso ({len(lista_metricas)} viviendas): {latencia_total_script:.2f} segundos")

print("\n[ESTABILIDAD TEMPORAL]")
mae_A_mean = df_metricas['mae_sub_A'].mean()
mae_B_mean = df_metricas['mae_sub_B'].mean()

degradacion = ((mae_B_mean - mae_A_mean) / mae_A_mean) * 100
print(f"  MAE Semestre A (Jun-Dic 21): {mae_A_mean:.4f} kWh")
print(f"  MAE Semestre B (Ene-Jun 22): {mae_B_mean:.4f} kWh")
signo = "+" if degradacion > 0 else ""
print(f"  Degradación del modelo:  {signo}{degradacion:.2f}% de error")

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

print("\n[CIUDAD MÁS DIFÍCIL Y MÁS FÁCIL PARA LSTM]")
mejor_ciudad = resumen_ciudad["MAE_media"].idxmin()
peor_ciudad = resumen_ciudad["MAE_media"].idxmax()
print(f"  Mejor ciudad:  {mejor_ciudad} (MAE={resumen_ciudad.loc[mejor_ciudad, 'MAE_media']:.4f} kWh)")
print(f"  Peor ciudad:   {peor_ciudad} (MAE={resumen_ciudad.loc[peor_ciudad, 'MAE_media']:.4f} kWh)")

ruta_ciudad = DIR_METRICAS / "metricas_lstm_por_ciudad.csv"
ruta_estabilidad = DIR_METRICAS / "estabilidad_lstm_por_ciudad.csv"

resumen_ciudad.reset_index().to_csv(ruta_ciudad, index=False)
est_ciudad.reset_index().to_csv(ruta_estabilidad, index=False)
print(f"\n Exportados desgloses en: {ruta_ciudad} y {ruta_estabilidad}")