import os
import pandas as pd
import math
import time
import pickle
from river import time_series 
from river import preprocessing
from river import metrics
from river import optim
from river import linear_model
from river import utils
from sklearn.metrics import r2_score
from pathlib import Path

# Configuración del modelo 
NOMBRE_MODELO = "snarimax"
FECHA_EVALUACION = pd.to_datetime("2020-01-01 00:00:00")

# Rejilla prequencial 
LEARNING_RATES = [0.0001, 0.001, 0.005, 0.01, 0.1]

DIR_RESULTADOS = os.path.join("resultados", NOMBRE_MODELO)
DIR_METRICAS = os.path.join("metricas", NOMBRE_MODELO)
DIR_MODELOS = "modelos"  

os.makedirs(DIR_RESULTADOS, exist_ok=True)
os.makedirs(DIR_METRICAS, exist_ok=True)
os.makedirs(DIR_MODELOS, exist_ok=True)

print("Cargando dataset...")
SCRIPT_DIR = Path(__file__).resolve().parent
ruta_csv = SCRIPT_DIR / "datos" / "dataset_online_15min.csv"
df = pd.read_csv(ruta_csv)
df["timestamp"] = pd.to_datetime(df["timestamp"])

casas = df["cups_id"].unique()

# Listas para guardar los resultados
resultados_predicciones = []
resultados_metricas = []

# Evaluación prequencial
for cups in casas:
    ciudad = df[df["cups_id"] == cups]["ciudad_asignada"].iloc[0]
    print(f"\n Procesando: {ciudad} ({cups[:8]}...)")
    
    # Filtrar y ordenar cronológicamente los datos de esta vivienda
    df_casa = df[df["cups_id"] == cups].sort_values("timestamp").reset_index(drop=True)
    
    # Warm-up
    modelos_grid = {
        lr: time_series.SNARIMAX(
            p=1, d=0, q=1, m=96, sp=1, sd=0, sq=0,
            regressor=linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
        ) for lr in LEARNING_RATES
    }
    metricas_grid = {lr: metrics.MAE() for lr in LEARNING_RATES}
    
    scaler = preprocessing.StandardScaler()
    
    # Variables de control del ganador
    modelo_seleccionado = None
    mejor_lr = None
    
    # Métricas definitivas para el 2020 del ganador
    mae_eval = metrics.MAE()
    rmse_eval = metrics.RMSE()
    mae_rolling = utils.Rolling(metrics.MAE(), window_size=96 * 7) # 7 días
    
    cont_warmup = 0
    cont_eval = 0
    tiempo_eval = 0.0
    
    # Listas de trazabilidad 
    ts_list = []
    y_real_list = []
    y_pred_list = []
    mae_rolling_list = []
    
    # Simulación de streaming de datos 
    for _, row in df_casa.iterrows():
        ts = row["timestamp"]        
        y = row["kWh"]
        
        # Variables de calendario y hora
        hora_frac = ts.hour + (ts.minute / 60.0)
        hora_sin = math.sin(2 * math.pi * hora_frac / 24.0)
        hora_cos = math.cos(2 * math.pi * hora_frac / 24.0)
        es_fin_de_semana = 1 if ts.weekday() >= 5 else 0
        
        x = {
            "temperatura": row["temperatura"],
            "hora_sin": hora_sin,
            "hora_cos": hora_cos,
            "es_fin_de_semana": es_fin_de_semana  
        }
        
        # Escalado en tiempo real
        scaler.learn_one(x)
        x_scaled = scaler.transform_one(x)
        
        # Fase Warm-Up: Competición de candidatos
        if ts < FECHA_EVALUACION:
            for lr, m in modelos_grid.items():
                try:
                    pred = m.forecast(horizon=1, xs=[x_scaled])[0]
                    if pred is not None and not math.isnan(pred):
                        metricas_grid[lr].update(y, max(0.0, pred))
                except Exception:
                    pass
                m.learn_one(y=y, x=x_scaled)
            cont_warmup += 1
            
        # Fase Evaluación: Selección y testeo del ganador
        else:
            if modelo_seleccionado is None:
                # Seleccionar el candidato con menor MAE acumulado durante 2019
                mejor_lr = min(metricas_grid, key=lambda k: metricas_grid[k].get())
                modelo_seleccionado = modelos_grid[mejor_lr]
                print(f"   [*] Warm-up completado. Mejor LR: {mejor_lr} (MAE: {metricas_grid[mejor_lr].get():.4f})")
                
                ruta_pkl_warmup = os.path.join(DIR_MODELOS, f"{NOMBRE_MODELO}_{cups}_warmup.pkl")
                artefactos_warmup = {
                    "model": modelo_seleccionado,
                    "scaler": scaler,
                    "mejor_lr": mejor_lr,
                    "cups_id": cups,
                    "ciudad": ciudad
                }
                with open(ruta_pkl_warmup, "wb") as f:
                    pickle.dump(artefactos_warmup, f)
                print(f"   [+] Modelo PKL (Warm-up) guardado en: {ruta_pkl_warmup}")
            
            inicio_t = time.perf_counter() 
            
            try:
                y_pred = modelo_seleccionado.forecast(horizon=1, xs=[x_scaled])[0]
            except Exception:
                y_pred = 0.0
                
            if y_pred is None or math.isnan(y_pred):
                y_pred = 0.0
            else:
                y_pred = max(0.0, y_pred)
            
            # Actualización de métricas
            mae_eval.update(y, y_pred)
            rmse_eval.update(y, y_pred)
            mae_rolling.update(y, y_pred)
            
            # Entrenamiento del ganador
            modelo_seleccionado.learn_one(y=y, x=x_scaled)
            
            tiempo_eval += (time.perf_counter() - inicio_t)
            cont_eval += 1
            
            ts_list.append(ts)
            y_real_list.append(y)
            y_pred_list.append(y_pred)
            mae_rolling_list.append(mae_rolling.get())

    ruta_pkl_final = os.path.join(DIR_MODELOS, f"{NOMBRE_MODELO}_{cups}_final.pkl")
    artefactos_final = {
        "model": modelo_seleccionado,
        "scaler": scaler,
        "mejor_lr": mejor_lr,
        "cups_id": cups,
        "ciudad": ciudad
    }
    with open(ruta_pkl_final, "wb") as f:
        pickle.dump(artefactos_final, f)
    print(f"   [+] Modelo PKL (Final) guardado en: {ruta_pkl_final}")

    # Cálculo de métricas
    y_real_arr = pd.Series(y_real_list)
    y_pred_arr = pd.Series(y_pred_list)
    
    suma_real = y_real_arr.sum()
    wape_final = (abs(y_real_arr - y_pred_arr).sum() / suma_real * 100) if suma_real > 0 else 0
    r2_final = r2_score(y_real_arr, y_pred_arr) if len(y_real_arr) > 1 else 0
    tiempo_ms = (tiempo_eval / cont_eval) * 1000 if cont_eval > 0 else 0

    # Dataframe de predicciones del candidato ganador en 2020
    df_res_cups = pd.DataFrame({
        "cups_id": cups,
        "ciudad": ciudad,
        "timestamp": ts_list,
        "Consumo_Real": y_real_list,
        "Prediccion": y_pred_list,
        "MAE_Rolling_7D": mae_rolling_list
    })
    resultados_predicciones.append(df_res_cups)

    # Métrica de concept drift
    mae_pre = df_res_cups[df_res_cups["timestamp"] < "2020-03-14"]["MAE_Rolling_7D"].mean()
    mae_drift = df_res_cups[(df_res_cups["timestamp"] >= "2020-03-14") & (df_res_cups["timestamp"] < "2020-06-21")]["MAE_Rolling_7D"].mean()

    # Guardar resultados 
    resultados_metricas.append({
        "cups_id": cups,
        "ciudad": ciudad,
        "mejor_lr": mejor_lr,
        "MAE_Global": mae_eval.get(),
        "RMSE_Global": rmse_eval.get(),
        "WAPE_Global": wape_final,
        "R2_Global": r2_final,
        "MAE_PreDrift": mae_pre,
        "MAE_Confinamiento": mae_drift,
        "Latencia (ms)": tiempo_ms
    })
    
    print(f"   [Eval: {cont_eval} iters] | Latencia: {tiempo_ms:.4f} ms")
    print(f"   WAPE: {wape_final:.2f}% | R2: {r2_final:.4f}")
    print(f"   DRIFT -> Pre: {mae_pre:.4f} | Confinamiento: {mae_drift:.4f}")

# Consolidación final
df_resultados_final = pd.concat(resultados_predicciones).reset_index(drop=True)
df_metricas_final = pd.DataFrame(resultados_metricas)

DIR_METRICAS = SCRIPT_DIR / "metricas" / "snarimax"
DIR_METRICAS.mkdir(parents=True, exist_ok=True)

DIR_RESULTADOS = SCRIPT_DIR / "resultados" / "snarimax"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

ruta_resultados = DIR_RESULTADOS / "resultados_snarimax.csv"
ruta_metricas = DIR_METRICAS / "metricas_snarimax.csv"

df_resultados_final.to_csv(ruta_resultados, index=False)
df_metricas_final.to_csv(ruta_metricas, index=False)

print("-----PROCESO COMPLETADO CON ÉXITO------")
print(f"  Predicciones -> {ruta_resultados}")
print(f"  Métricas     -> {ruta_metricas}")
print(f"  Modelos PKL  -> {DIR_MODELOS}/")
print("\nResumen de métricas:")
print(df_metricas_final[["ciudad", "mejor_lr", "WAPE_Global", "MAE_PreDrift", "MAE_Confinamiento"]].to_string(index=False))