import os
import pandas as pd
import math
import time
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from river import time_series 
from river import preprocessing
from river import metrics
from river import optim
from river import linear_model
from river import utils
from sklearn.metrics import r2_score
from pathlib import Path

# Configuración visual para las gráficas
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Configuración del modelo 
NOMBRE_MODELO = "snarimax"
FECHA_EVALUACION = pd.to_datetime("2020-01-01 00:00:00")

# Rejilla prequencial 
LEARNING_RATES = [0.0001, 0.001, 0.005, 0.01, 0.1]

# Configuración de directorios
SCRIPT_DIR = Path(__file__).resolve().parent
DIR_RESULTADOS = SCRIPT_DIR / "resultados" / NOMBRE_MODELO
DIR_METRICAS = SCRIPT_DIR / "metricas" / NOMBRE_MODELO
DIR_MODELOS = SCRIPT_DIR / "modelos"

DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
DIR_METRICAS.mkdir(parents=True, exist_ok=True)
DIR_MODELOS.mkdir(parents=True, exist_ok=True)

print("Cargando dataset...")
ruta_csv = SCRIPT_DIR / "datos" / "dataset_online_15min.csv"
df = pd.read_csv(ruta_csv)
df["timestamp"] = pd.to_datetime(df["timestamp"])

casas = df["cups_id"].unique()

# Listas para guardar los resultados
resultados_predicciones = []
resultados_metricas = []
lista_coeficientes = []  

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
        
        # Fase Warm-Up: Competición de candidatos
        if ts < FECHA_EVALUACION:
            # Escalado en tiempo real (en el warm-up no se mide la latencia)
            scaler.learn_one(x)
            x_scaled = scaler.transform_one(x)
            
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
                
                ruta_pkl_warmup = DIR_MODELOS / f"{NOMBRE_MODELO}_{cups}_warmup.pkl"
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
            
            # Escalado en tiempo real
            scaler.learn_one(x)
            x_scaled = scaler.transform_one(x)
            
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

    ruta_pkl_final = DIR_MODELOS / f"{NOMBRE_MODELO}_{cups}_final.pkl"
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

    pesos = modelo_seleccionado.regressor.weights.copy()
    dict_coefs = {"cups_id": cups, "ciudad": ciudad}
    dict_coefs.update(pesos)
    
    if hasattr(modelo_seleccionado.regressor, 'intercept'):
        dict_coefs['intercept'] = modelo_seleccionado.regressor.intercept
        
    lista_coeficientes.append(dict_coefs)
    # ---------------------------------------------------------------------------

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

# Consolidación final predictiva
df_resultados_final = pd.concat(resultados_predicciones).reset_index(drop=True)
df_metricas_final = pd.DataFrame(resultados_metricas)

ruta_resultados = DIR_RESULTADOS / "resultados_snarimax.csv"
ruta_metricas = DIR_METRICAS / "metricas_snarimax.csv"

df_resultados_final.to_csv(ruta_resultados, index=False)
df_metricas_final.to_csv(ruta_metricas, index=False)

print("\n----- PROCESO COMPLETADO CON ÉXITO ------")
print(f"  Predicciones -> {ruta_resultados}")
print(f"  Métricas     -> {ruta_metricas}")
print(f"  Modelos PKL  -> {DIR_MODELOS}/")
print("\nResumen de métricas:")
print(df_metricas_final[["ciudad", "mejor_lr", "WAPE_Global", "MAE_PreDrift", "MAE_Confinamiento"]].to_string(index=False))

print("\n--------- PASO 4: Interpretando coeficientes finales de SNARIMAX ---------")

df_coefs = pd.DataFrame(lista_coeficientes).fillna(0)
ruta_coefs = DIR_METRICAS / "coeficientes_snarimax.csv"
df_coefs.to_csv(ruta_coefs, index=False)

# Extraer solo las columnas que son variables 
features_snarimax = [c for c in df_coefs.columns if c not in ["cups_id", "ciudad"]]

# Media y dispersión entre las viviendas 
resumen_coefs = df_coefs[features_snarimax].agg(["mean", "std"]).T
resumen_coefs.columns = ["media", "desv_std"]
resumen_coefs = resumen_coefs.sort_values("media", key=abs, ascending=False)

# Variable dominante por vivienda 
coefs_abs = df_coefs[features_snarimax].abs()
df_coefs["variable_dominante"] = coefs_abs.idxmax(axis=1)
conteo_dominante = df_coefs["variable_dominante"].value_counts()

print("\n[IMPACTO PROMEDIO DE CADA VARIABLE (media ± desv. std entre viviendas)]")
for var_name, row in resumen_coefs.iterrows():
    print(f"  - {var_name:20s}: {row['media']:+.4f} ± {row['desv_std']:.4f}")

print("\n[VARIABLE DOMINANTE POR VIVIENDA]")
for var_name, count in conteo_dominante.items():
    print(f"  - {var_name:20s}: dominante en {count}/{len(df_coefs)} viviendas")

resumen_coefs.to_csv(DIR_METRICAS / "resumen_coeficientes_snarimax.csv")

# Gráfico de importancia
plt.figure(figsize=(10, 6))
orden = resumen_coefs.sort_values("media").index
medias = resumen_coefs.loc[orden, "media"]
errores = resumen_coefs.loc[orden, "desv_std"]
colores = ["#d9534f" if v < 0 else "#0275d8" for v in medias]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(orden, medias, xerr=errores, color=colores, capsize=4)
ax.axvline(0, color="black", linestyle="--", linewidth=1.2)

ax.set_title("Importancia de las variables en SNARIMAX (Pesos finales)", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Coeficiente (media ± desv. std entre viviendas)", fontsize=11)
ax.set_ylabel("Variables (Exógenas y Autorregresivas)", fontsize=11)

for i, (media, err) in enumerate(zip(medias, errores)):
    offset = (medias.max() - medias.min()) * 0.02
    offset = offset if media >= 0 else -offset
    ha = "left" if media >= 0 else "right"
    ax.annotate(f"{media:+.3f}", (media + offset, i), va="center", ha=ha,
                fontsize=10, color="black", fontweight="bold")

plt.tight_layout()
ruta_grafico = DIR_METRICAS / "coeficientes_snarimax.png"
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print(f"\nGráfico XAI exportado en: {ruta_grafico}")

# % de viviendas con coeficiente positivo
signos = (df_coefs[features_snarimax] > 0).mean() * 100 
print("\n[% DE VIVIENDAS CON COEFICIENTE POSITIVO]")
for var_name, pct in signos.sort_values(ascending=False).items():
    print(f"  - {var_name:20s}: {pct:.0f}% positivo")