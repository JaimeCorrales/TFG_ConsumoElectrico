import os
import pandas as pd
import numpy as np
import math
import time
import random
from collections import deque
from sklearn.metrics import r2_score
from river import linear_model, optim, time_series, tree, preprocessing, metrics, compose, utils
from pathlib import Path

# Semilla para reproducibilidad
random.seed(42)
np.random.seed(42)

# Tasa de aprendizaje de SGD y SNARIMAX: la seleccionada en la búsqueda prequencial de los modelos individuales
LR_SGD = 0.01
LR_SNARIMAX = 0.01


class SnarimaxEscalado:

    def __init__(self, lr=LR_SNARIMAX):
        self.scaler = preprocessing.StandardScaler()
        self.model = time_series.SNARIMAX(
            p=1, d=0, q=1, m=96, sp=1, sd=0, sq=0,
            regressor=linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
        )
        self._x_scaled = None

    def _escalar(self, x):
        x_in = {k: v for k, v in x.items() if k != "consumo_t_96"}
        self.scaler.learn_one(x_in)
        return self.scaler.transform_one(x_in)

    def predict_one(self, x):
        self._x_scaled = self._escalar(x)
        return self.model.forecast(horizon=1, xs=[self._x_scaled])[0]

    def learn_one(self, x, y):
        # En el warm-up no hay predicción previa
        x_scaled = self._x_scaled if self._x_scaled is not None else self._escalar(x)
        self.model.learn_one(y=y, x=x_scaled)
        self._x_scaled = None


# Clase para el selector Bandit (Epsilon-Greedy)
class BanditRegressor:
    def __init__(self, modelos, epsilon=0.1, alpha=0.05):
        self.modelos = modelos
        self.epsilon = epsilon
        self.alpha = alpha  # Factor para la media móvil exponencial del error
        self.errores = {k: 0.0 for k in sorted(modelos.keys())}
        self.conteos = {k: 0 for k in sorted(modelos.keys())}
        self.fallos = {k: 0 for k in sorted(modelos.keys())}  # excepciones por brazo
        self.preds = {}
        self.elegido = None

    def predict_one(self, x):
        nombres_modelos = sorted(self.modelos.keys())  # Lista ordenada determinista

        for k in nombres_modelos:
            try:
                p = self.modelos[k].predict_one(x)
                if p is None or math.isnan(p):
                    p = 0.0
                self.preds[k] = max(0.0, p)
            except Exception:
                self.fallos[k] += 1
                self.preds[k] = 0.0

        # Epsilon-Greedy con ordenación para empates
        if random.random() < self.epsilon or all(v == 0.0 for v in self.errores.values()):
            self.elegido = random.choice(nombres_modelos)  # Exploración
        else:
            self.elegido = min(nombres_modelos, key=lambda k: self.errores[k])  # Explotación

        self.conteos[self.elegido] += 1
        return self.preds[self.elegido]

    def learn_one(self, x, y):
        for k, m in self.modelos.items():
            p = self.preds.get(k, 0.0)
            err_actual = abs(y - p)

            if self.errores[k] == 0.0:
                self.errores[k] = err_actual
            else:
                self.errores[k] = (self.alpha * err_actual) + ((1 - self.alpha) * self.errores[k])

            try:
                m.learn_one(x, y)
            except Exception:
                self.fallos[k] += 1


print("Cargando dataset...")
SCRIPT_DIR = Path(__file__).resolve().parent
ruta_csv = SCRIPT_DIR / "datos" / "dataset_online_15min.csv"
df = pd.read_csv(ruta_csv)
df["timestamp"] = pd.to_datetime(df["timestamp"])

fecha_corte = pd.to_datetime("2020-01-01 00:00:00")
fecha_confinamiento = pd.to_datetime("2020-03-14 00:00:00")
fecha_fin_confinamiento = pd.to_datetime("2020-06-21 00:00:00")

casas = df["cups_id"].unique()
gp_candidates = [10, 50, 100, 200]

resultados_predicciones = []
resultados_metricas = []

for cups in casas:
    ciudad = df[df["cups_id"] == cups]["ciudad_asignada"].iloc[0]
    print(f"\n Procesando: {ciudad} ({cups[:8]}...)")
    
    df_cups = df[df["cups_id"] == cups].sort_values("timestamp").reset_index(drop=True)
    memoria_consumo = deque(maxlen=96)
    
    # Modelos base para warm-up (misma configuración que los individuales)
    sgd = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LinearRegression(optimizer=optim.SGD(lr=LR_SGD))
    )
    snarimax = SnarimaxEscalado(lr=LR_SNARIMAX)
    
    # Rejilla para seleccionar el mejor HAT durante warm-up
    hat_candidates = {
    gp: compose.Pipeline(
        preprocessing.StandardScaler(),
        tree.HoeffdingAdaptiveTreeRegressor(
            grace_period=gp, 
            leaf_prediction='adaptive',
            seed=42     
        )
    )
    for gp in gp_candidates
    }
    
    hat_mae_warmup = {gp: metrics.MAE() for gp in gp_candidates}
    n_warmup = 0
    
    # Warm-up
    for _, row in df_cups[df_cups["timestamp"] < fecha_corte].iterrows():
        ts = row["timestamp"]
        y = row["kWh"]
        
        h = ts.hour + (ts.minute / 60.0)
        hora_sin = math.sin(2 * math.pi * h / 24.0)
        hora_cos = math.cos(2 * math.pi * h / 24.0)
        es_fin_de_semana = 1 if ts.weekday() >= 5 else 0
        lag_t_96 = memoria_consumo[0] if len(memoria_consumo) == 96 else 0.0
        
        x = {
            "temperatura": row["temperatura"],
            "hora_sin": hora_sin,
            "hora_cos": hora_cos,
            "es_fin_de_semana": es_fin_de_semana,
            "consumo_t_96": lag_t_96
        }
        
        sgd.learn_one(x, y)
        snarimax.learn_one(x, y)
        
        for gp, hat_m in hat_candidates.items():
            pred_hat = hat_m.predict_one(x) or 0.0
            hat_mae_warmup[gp].update(y, max(0.0, pred_hat))
            hat_m.learn_one(x, y)
            
        memoria_consumo.append(y)
        n_warmup += 1

    best_gp = min(hat_mae_warmup, key=lambda k: hat_mae_warmup[k].get())
    hat_winner = hat_candidates[best_gp]
    best_hat_mae = hat_mae_warmup[best_gp].get()
    
    print(f"   [*] Warm-up completado. Mejor GP para HAT: {best_gp} (MAE Warm-up: {best_hat_mae:.4f})")
    
    # Configuración Bandit y evaluación
    bandit = BanditRegressor(
        modelos={"SGD": sgd, "SNARIMAX": snarimax, "HAT": hat_winner},
        epsilon=0.1,
        alpha=0.05
    )
    
    mae_eval = metrics.MAE()
    rmse_eval = metrics.RMSE()
    mae_rolling = utils.Rolling(metrics.MAE(), window_size=96 * 7)  # 7 días
    
    n_eval = 0
    t_total = 0.0
    
    ts_list = []
    y_real_list = []
    y_pred_list = []
    mae_rolling_list = []
    
    for _, row in df_cups[df_cups["timestamp"] >= fecha_corte].iterrows():
        ts = row["timestamp"]
        y = row["kWh"]
        
        h = ts.hour + (ts.minute / 60.0)
        hora_sin = math.sin(2 * math.pi * h / 24.0)
        hora_cos = math.cos(2 * math.pi * h / 24.0)
        es_fin_de_semana = 1 if ts.weekday() >= 5 else 0
        lag_t_96 = memoria_consumo[0] if len(memoria_consumo) == 96 else 0.0
        
        x = {
            "temperatura": row["temperatura"],
            "hora_sin": hora_sin,
            "hora_cos": hora_cos,
            "es_fin_de_semana": es_fin_de_semana,
            "consumo_t_96": lag_t_96
        }
        
        t0 = time.perf_counter()
        
        pred = bandit.predict_one(x)
        pred = max(0.0, pred)
        
        mae_eval.update(y, pred)
        rmse_eval.update(y, pred)
        
        mae_rolling.update(y, pred)
            
        bandit.learn_one(x, y)
        
        t_total += (time.perf_counter() - t0)
        n_eval += 1
        
        ts_list.append(ts)
        y_real_list.append(y)
        y_pred_list.append(pred)
        mae_rolling_list.append(mae_rolling.get())
        
        memoria_consumo.append(y)
        
    y_real_arr = np.array(y_real_list)
    y_pred_arr = np.array(y_pred_list)
    
    suma_real = np.sum(y_real_arr)
    wape_c = (np.sum(np.abs(y_real_arr - y_pred_arr)) / suma_real * 100) if suma_real > 0 else 0
    r2_c = r2_score(y_real_arr, y_pred_arr) if len(y_real_arr) > 1 else 0
    ms_muestra = (t_total / n_eval) * 1000 if n_eval > 0 else 0
    
    df_res_cups = pd.DataFrame({
        "cups_id": cups,
        "ciudad": ciudad,
        "timestamp": ts_list,
        "Consumo_Real": y_real_list,
        "Prediccion": y_pred_list,
        "MAE_Rolling_7D": mae_rolling_list
    })
    resultados_predicciones.append(df_res_cups)

    # Métrica de concept drift: misma definición que en los modelos individuales
    # (media del MAE móvil de 7 días antes del confinamiento y durante el mismo)
    mae_pre = df_res_cups[df_res_cups["timestamp"] < fecha_confinamiento]["MAE_Rolling_7D"].mean()
    mae_drift = df_res_cups[
        (df_res_cups["timestamp"] >= fecha_confinamiento) & (df_res_cups["timestamp"] < fecha_fin_confinamiento)
    ]["MAE_Rolling_7D"].mean()

    total_votos = sum(bandit.conteos.values())
    pct_sgd = (bandit.conteos["SGD"] / total_votos * 100) if total_votos > 0 else 0
    pct_snarimax = (bandit.conteos["SNARIMAX"] / total_votos * 100) if total_votos > 0 else 0
    pct_hat = (bandit.conteos["HAT"] / total_votos * 100) if total_votos > 0 else 0

    print(f"   [Warm-up: {n_warmup} iters] -> [Eval: {n_eval} iters]")
    print(f"   MAE: {mae_eval.get():.4f} | RMSE: {rmse_eval.get():.4f} | WAPE: {wape_c:.2f}% | R2: {r2_c:.4f} | Latencia: {ms_muestra:.4f} ms")
    print(f"   DRIFT -> Pre: {mae_pre:.4f} | Confinamiento: {mae_drift:.4f}")
    print(f"   Uso de modelos: [SGD: {pct_sgd:.1f}%] [SNARIMAX: {pct_snarimax:.1f}%] [HAT: {pct_hat:.1f}%]")
    print(f"   Excepciones por brazo: {bandit.fallos}\n")

    resultados_metricas.append({
        "ciudad": ciudad,
        "mejor_gp_hat": best_gp,
        "MAE": mae_eval.get(),
        "RMSE": rmse_eval.get(),
        "WAPE_Global": wape_c,
        "R2": r2_c,
        "Latencia_ms": ms_muestra,
        "MAE_PreDrift": mae_pre,
        "MAE_Confinamiento": mae_drift,
        "Pct_SGD": pct_sgd,
        "Pct_SNARIMAX": pct_snarimax,
        "Pct_HAT": pct_hat,
        "Fallos_SGD": bandit.fallos["SGD"],
        "Fallos_SNARIMAX": bandit.fallos["SNARIMAX"],
        "Fallos_HAT": bandit.fallos["HAT"]
    })

df_resultados_final = pd.concat(resultados_predicciones).reset_index(drop=True)
df_metricas_final = pd.DataFrame(resultados_metricas)

DIR_METRICAS = SCRIPT_DIR / "metricas" / "bandit"
DIR_METRICAS.mkdir(parents=True, exist_ok=True)

DIR_RESULTADOS = SCRIPT_DIR / "resultados" / "bandit"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

ruta_resultados = DIR_RESULTADOS / "resultados_bandit.csv"
ruta_metricas = DIR_METRICAS / "metricas_bandit.csv"

df_resultados_final.to_csv(ruta_resultados, index=False)
df_metricas_final.to_csv(ruta_metricas, index=False)

print("-------PROCESO MULTI-ARMED BANDIT COMPLETADO CON ÉXITO------")
print(f"  Predicciones -> {ruta_resultados}")
print(f"  Métricas     -> {ruta_metricas}\n")
print("Resumen de métricas:")
print(df_metricas_final.to_string(index=False))