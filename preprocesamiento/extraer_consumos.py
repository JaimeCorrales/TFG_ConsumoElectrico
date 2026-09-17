import pandas as pd
import os
import glob

ARCHIVO_SELECCION = os.path.join("datos_auxiliares", "casas_seleccionadas_global.csv")
ARCHIVO_SALIDA    = "consumos_finales_tfg.csv"

RUTA_BRUTOS = "datos_brutos"
CARPETAS_CONSUMO = [
    os.path.join(RUTA_BRUTOS, "goi4_pre", "imp_csv"),
    os.path.join(RUTA_BRUTOS, "goi4_in", "imp_csv"),
    os.path.join(RUTA_BRUTOS, "goi4_pst", "imp_csv")
]

# Cargar plantilla
print("-----PASO 1: Cargando seleccion de viviendas y mapeo de ciudades-----")

if not os.path.exists(ARCHIVO_SELECCION):
    print(f"No se encuentra el archivo '{ARCHIVO_SELECCION}'.")
    exit()

df_seleccion = pd.read_csv(ARCHIVO_SELECCION)

# Mapa para asociar directamente cada CUPS con su ciudad
mapa_ciudades = dict(zip(df_seleccion["cups_id"].astype(str), df_seleccion["ciudad_asignada"]))
lista_cups_buscados = set(mapa_ciudades.keys())

fecha_inicio_tfg = pd.to_datetime(df_seleccion["arco_inicio_comun"].iloc[0])
fecha_fin_tfg    = pd.to_datetime(df_seleccion["arco_fin_comun"].iloc[0])

print(f"[INFO] Buscando datos para {len(lista_cups_buscados)} viviendas.")
print(f"[INFO] Ventana temporal: {fecha_inicio_tfg.date()} -> {fecha_fin_tfg.date()}")

# Extraer y etiquietar consumos
print("-----PASO 2: Localizando y leyendo archivos de consumo-----")

datos_extraidos = []
casas_encontradas = set()

for carpeta in CARPETAS_CONSUMO:
    if not os.path.exists(carpeta):
        print(f"Carpeta '{carpeta}' no detectada. Saltando.")
        continue
        
    print(f"Buscando en: {carpeta}")
    archivos_csv = glob.glob(os.path.join(carpeta, "**", "*.csv"), recursive=True)
    
    for ruta_archivo in archivos_csv:
        nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]
        
        if nombre_archivo in lista_cups_buscados:
            print(f"   Casa encontrada: {nombre_archivo} ({mapa_ciudades[nombre_archivo]})")
            casas_encontradas.add(nombre_archivo)
            
            try:
                df_casa = pd.read_csv(ruta_archivo)
                df_casa["cups_id"] = nombre_archivo
                df_casa["ciudad_asignada"] = mapa_ciudades[nombre_archivo]
                datos_extraidos.append(df_casa)
            except Exception as e:
                print(f"  Error al leer {ruta_archivo}: {e}")

if not datos_extraidos:
    print(" No se han encontrado datos para las viviendas seleccionadas.")
    exit()

# Unir, filtrar y ordenar columns

print("-----PASO 3: Consolidando y ordenando dataset de consumo----")

df_total = pd.concat(datos_extraidos, ignore_index=True)
df_total["timestamp"] = pd.to_datetime(df_total["timestamp"], errors="coerce")

# Filtro por ventana temporal comun
df_final = df_total[
    (df_total["timestamp"] >= fecha_inicio_tfg) & 
    (df_total["timestamp"] <= fecha_fin_tfg)
].copy()

# Estructura ordenada de columnas
columnas_ordenadas = ["timestamp", "ciudad_asignada", "cups_id", "kWh", "imputed"]
df_final = df_final[columnas_ordenadas].sort_values(by=["cups_id", "timestamp"]).reset_index(drop=True)

df_final.to_csv(ARCHIVO_SALIDA, index=False)

print(f"----Archivo creado con exito: '{ARCHIVO_SALIDA}' ({len(df_final)} filas).-----")