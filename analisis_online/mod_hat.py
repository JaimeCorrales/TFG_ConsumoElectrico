import os
import pandas as pd
import numpy as np
import math
import time
from collections import deque
from river import tree, preprocessing, metrics, compose, utils
from sklearn.metrics import r2_score
from pathlib import Path

# Configuración del modelo 
NOMBRE_MODELO = "hat"
FECHA_EVALUACION = pd.to_datetime("2020-01-01 00:00:00")

# Rejilla de hiperparámetros 
GRACE_PERIODS = [10, 50, 100, 200]

print("Cargando dataset...")
SCRIPT_DIR = Path(__file__).resolve().parent
ruta_csv = SCRIPT_DIR / "datos" / "dataset_online_15min.csv"
df = pd.read_csv(ruta_csv)
df["timestamp"] = pd.to_datetime(df["timestamp"])

casas = df["cups_id"].unique()

# Listas para guardar los resultados finales
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
        gp: compose.Pipeline(
            preprocessing.StandardScaler(),
            tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=gp,
                leaf_prediction='adaptive',
                seed=42
            )
        ) for gp in GRACE_PERIODS
    }
    metricas_grid = {gp: metrics.MAE() for gp in GRACE_PERIODS}

    # Control del ganador
    modelo_seleccionado = None
    mejor_gp = None

    # Métricas definitivas para el 2020 del ganador
    mae_eval = metrics.MAE()
    rmse_eval = metrics.RMSE()
    mae_rolling = utils.Rolling(metrics.MAE(), window_size=96 * 7) # 7 días

    cont_warmup = 0
    cont_eval = 0
    tiempo_eval = 0.0
    
    # Memoria estacional (24h = 96 steps de 15 min)
    memoria_consumo = deque(maxlen=96)
    
    # Listas de trazabilidad 2020
    ts_list = []
    y_real_list = []
    y_pred_list = []
    mae_rolling_list = []
    
    # Simulación de streaming paso a paso
    for _, row in df_casa.iterrows():
        ts = row["timestamp"]
        y = row["kWh"]
        
        # Variables temporales y de calendario
        hora_frac = ts.hour + (ts.minute / 60.0)
        hora_sin = math.sin(2 * math.pi * hora_frac / 24.0)
        hora_cos = math.cos(2 * math.pi * hora_frac / 24.0)
        es_fin_semana = 1 if ts.weekday() >= 5 else 0
        
        # Lag t-96 (consumo hace 24h)
        lag_t_96 = memoria_consumo[0] if len(memoria_consumo) == 96 else 0.0
        
        # Input del modelo
        x = {
            "temperatura": row["temperatura"],
            "hora_sin": hora_sin,
            "hora_cos": hora_cos,
            "es_fin_de_semana": es_fin_semana,
            "consumo_t_96": lag_t_96
        }
        
        # Fase Warm-Up: Competiciín de candidatos
        if ts < FECHA_EVALUACION:
            for gp, m in modelos_grid.items():
                pred = m.predict_one(x)
                if pred is not None:
                    metricas_grid[gp].update(y, max(0.0, pred))
                m.learn_one(x, y)
            cont_warmup += 1
            
        # Fase Evaluación: Selección y testeo del ganador
        else:
            if modelo_seleccionado is None:
                # Seleccionar el candidato con menor MAE acumulado durante 2019
                mejor_gp = min(metricas_grid, key=lambda k: metricas_grid[k].get())
                modelo_seleccionado = modelos_grid[mejor_gp]
                print(f"   [*] Warm-up completado. Mejor GP: {mejor_gp} (MAE: {metricas_grid[mejor_gp].get():.4f})")

            inicio_t = time.perf_counter()
            
            y_pred = modelo_seleccionado.predict_one(x)
            y_pred = 0.0 if y_pred is None else max(0.0, y_pred)
            
            mae_eval.update(y, y_pred)
            rmse_eval.update(y, y_pred)
            mae_rolling.update(y, y_pred)
            
            modelo_seleccionado.learn_one(x, y)
            
            tiempo_eval += (time.perf_counter() - inicio_t)
            cont_eval += 1
            
            ts_list.append(ts)
            y_real_list.append(y)
            y_pred_list.append(y_pred)
            mae_rolling_list.append(mae_rolling.get())
            
        # Actualización de memoria histórica
        memoria_consumo.append(y)

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
        "mejor_gp": mejor_gp,
        "MAE_Global": mae_eval.get(),
        "RMSE_Global": rmse_eval.get(),
        "WAPE_Global": wape_final,
        "R2_Global": r2_final,
        "MAE_PreDrift": mae_pre,
        "MAE_Confinamiento": mae_drift,
        "Latencia (ms)": tiempo_ms
    })
    
    print(f"   [Warm-up: {cont_warmup} iters] -> [Eval: {cont_eval} iters]")
    print(f"   MAE: {mae_eval.get():.4f} | RMSE: {rmse_eval.get():.4f} | WAPE: {wape_final:.2f}% | R2: {r2_final:.4f} | Latencia: {tiempo_ms:.4f} ms")
    print(f"   DRIFT -> Pre: {mae_pre:.4f} | Confinamiento: {mae_drift:.4f}")

# Consolidación final
df_resultados_final = pd.concat(resultados_predicciones).reset_index(drop=True)
df_metricas_final = pd.DataFrame(resultados_metricas)

DIR_METRICAS = SCRIPT_DIR / "metricas" / "hat"
DIR_METRICAS.mkdir(parents=True, exist_ok=True)

DIR_RESULTADOS = SCRIPT_DIR / "resultados" / "hat"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

ruta_resultados = DIR_RESULTADOS / "resultados_hat.csv"
ruta_metricas = DIR_METRICAS / "metricas_hat.csv"

df_resultados_final.to_csv(ruta_resultados, index=False)
df_metricas_final.to_csv(ruta_metricas, index=False)

print("-----PROCESO COMPLETADO CON ÉXITO------")
print(f"  Predicciones -> {ruta_resultados}")
print(f"  Métricas     -> {ruta_metricas}")
print("\nResumen de Métricas:")
print(df_metricas_final[["ciudad", "mejor_gp", "WAPE_Global", "MAE_PreDrift", "MAE_Confinamiento"]].to_string(index=False))