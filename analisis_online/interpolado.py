import pandas as pd
import numpy as np

df = pd.read_csv("dataset_online.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["cups_id", "timestamp"]).reset_index(drop=True)

fragmentos = []
consumo_original_limpio = 0  # Mediremos el consumo real tras limpiar duplicados

for cups, grupo in df.groupby("cups_id"):
    ciudad = grupo["ciudad_asignada"].iloc[0]

    # Colapsar duplicados si existen 
    agg_dict = {"kWh": "mean", "temperatura": "mean"}
    if "imputed" in grupo.columns:
        agg_dict["imputed"] = "first"
    grupo = grupo.groupby("timestamp").agg(agg_dict)

    # Sumamos el consumo real de esta casa ya limpio de duplicados
    consumo_original_limpio += grupo["kWh"].sum()

    grupo_15min = grupo.loc[grupo.index.repeat(4)].copy()
    
    # Creamos un patrón cíclico de minutos para cada hora
    minutos_a_anadir = np.tile([0, 15, 30, 45], len(grupo))
    grupo_15min.index = grupo_15min.index + pd.to_timedelta(minutos_a_anadir, unit='m')

    #Dividir estrictamente entre 4 
    grupo_15min["kWh"] = grupo_15min["kWh"] / 4.0

    # Dejamos el valor real en el minuto 00 y ponemos NaN en el resto para que interpolate funcione
    mask_interp = grupo_15min.index.minute != 0
    grupo_15min.loc[mask_interp, "temperatura"] = np.nan
    grupo_15min["temperatura"] = grupo_15min["temperatura"].interpolate(method="time")
    # ffill de seguridad por si la última hora del dataset no tiene una hora posterior para interpolar
    grupo_15min["temperatura"] = grupo_15min["temperatura"].ffill()

    # Imputed: Forward fill si existe ---
    if "imputed" in grupo.columns:
        grupo_15min["imputed"] = grupo_15min["imputed"].ffill()

    # Asignar metadatos obligatorios
    grupo_15min["cups_id"] = cups
    grupo_15min["ciudad_asignada"] = ciudad
    grupo_15min["es_interpolado"] = mask_interp

    fragmentos.append(grupo_15min)

# Unificar todas las casas transformadas
df_15min = pd.concat(fragmentos).reset_index().rename(columns={"index": "timestamp"})

# Reordenar columnas igual que el original
cols_finales = ["timestamp", "ciudad_asignada", "cups_id", "temperatura", "kWh", "es_interpolado"]
if "imputed" in df_15min.columns:
    cols_finales.insert(5, "imputed")

df_15min = df_15min[cols_finales]

consumo_15min = df_15min["kWh"].sum()
diferencia = abs(consumo_original_limpio - consumo_15min)

print(f"Filas originales (con duplicados): {len(df)}")
print(f"Filas cuarto-horarias generadas:   {len(df_15min)}")
print(f"Ratio real de expansión:           x{len(df_15min)/len(df):.2f}")
print(f"\nConsumo base (limpio):             {consumo_original_limpio:.4f} kWh")
print(f"Consumo cuarto-horario:            {consumo_15min:.4f} kWh")
print(f"Diferencia:             {diferencia:.6f} kWh")

assert diferencia < 0.01, "La energía sigue sin conservarse"
print(" La energía total se conserva perfectamente.")

df_15min.to_csv("dataset_online_15min.csv", index=False)
print("\n Archivo guardado correctamente: dataset_online_15min.csv")