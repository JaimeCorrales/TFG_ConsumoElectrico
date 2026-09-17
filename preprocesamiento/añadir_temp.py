import pandas as pd
import os

ARCHIVO_CONSUMOS = "consumos_finales_tfg.csv"
ARCHIVO_SALIDA   = "dataset_final_IA.csv"

# Mapeamos los archivos meteorologicos ubicados en datos_auxiliares/
DIR_AUX = "datos_auxiliares"
MAPEO_CLIMA = {
    os.path.join(DIR_AUX, "open-meteo-39.47N0.37W19m.csv"): "Valencia",
    os.path.join(DIR_AUX, "open-meteo-40.39N3.68W651m.csv"): "Madrid",
    os.path.join(DIR_AUX, "open-meteo-43.27N2.96W21m.csv"): "Bilbao",
    os.path.join(DIR_AUX, "open-meteo-43.34N8.35W22m.csv"): "Coruna A"
}

print("----Inyectando las temperaturas en el dataset----")

# Cargar consumos base
print("Cargando datos de consumo")
if not os.path.exists(ARCHIVO_CONSUMOS):
    print(f"No se encuentra {ARCHIVO_CONSUMOS}")
    exit()

df_consumos = pd.read_csv(ARCHIVO_CONSUMOS)
df_consumos["timestamp"] = pd.to_datetime(df_consumos["timestamp"])

# Leer y preparar los archivos del clima
print("Procesando archivos meteorologicos de Open-Meteo")
lista_df_clima = []

for archivo, ciudad in MAPEO_CLIMA.items():
    if os.path.exists(archivo):
        df_clima = pd.read_csv(archivo, skiprows=2)
        
        df_clima = df_clima.rename(columns={
            "time": "timestamp",
            "temperature_2m (°C)": "temperatura"
        })
        
        df_clima["timestamp"] = pd.to_datetime(df_clima["timestamp"])
        df_clima["ciudad_asignada"] = ciudad
        
        lista_df_clima.append(df_clima[["timestamp", "ciudad_asignada", "temperatura"]])
    else:
        print(f"No se encontro el archivo meteorologico: {archivo}")

if not lista_df_clima:
    print("No se ha podido cargar ningun archivo de clima.")
    exit()

df_clima_total = pd.concat(lista_df_clima, ignore_index=True)

# Cruce 
print("Fusionando consumos y temperaturas")
df_final = pd.merge(df_consumos, df_clima_total, on=["timestamp", "ciudad_asignada"], how="left")

# Rellenar posibles huecos de temperatura por interpolacion lineal
df_final["temperatura"] = df_final.groupby("ciudad_asignada")["temperatura"].transform(lambda x: x.interpolate())

columnas_ordenadas = ["timestamp", "ciudad_asignada", "cups_id", "temperatura", "kWh", "imputed"]
df_final = df_final[columnas_ordenadas]

# Guardar dataset final
df_final.to_csv(ARCHIVO_SALIDA, index=False)

print(f"----Dataset completado con exito: '{ARCHIVO_SALIDA}' ({len(df_final)} filas).----")
print("\n Vista previa del dataset final:")
print(df_final.head(5))