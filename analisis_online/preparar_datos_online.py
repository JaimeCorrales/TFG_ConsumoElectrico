import pandas as pd

nombre_archivo_original = "dataset_final_IA.csv"
print(f"Cargando datos desde {nombre_archivo_original}...")
df = pd.read_csv(nombre_archivo_original)

# Definimos los CUPS ganadores (las casas con mayor concept drift absoluto)
# Extraídos del análisis de impacto pandémico (Enero vs Abril)
cups_elegidos = [
    "7acaed42b22a8109889ac3e22dea9792c2579317dc135e5823ce8491cd3b0db4", # A Coruña (-62.41%)
    "35efd7b3768e77ebda557ca56c4acb5732f40a41e424398dbab292422e10ffae", # Bilbao (-35.85%)
    "7cfe79b7a19094aa8137c2640e682e0fd710cf7a9fd8734d4034547dfb750f00", # Madrid (-34.68%)
    "1844d90de99c83bfabbf726a15156cba2eb58e639f088de99bdb7aebaab65bf6"  # Valencia (-51.50%)
]

# Filtrar el dataset por las casas seleccionadas
print("Filtrando las 4 casas representativas para Online Learning...")
df_online = df[df["cups_id"].isin(cups_elegidos)].copy()

# Convertir fechas y aplicar el recorte temporal (Warm-up + Evaluación)
if "timestamp" in df_online.columns:
    df_online["timestamp"] = pd.to_datetime(df_online["timestamp"])
    
    print("Recortando el periodo temporal (Octubre 2019 - Diciembre 2020)...")
    # Definimos nuestra ventana de estudio
    fecha_inicio = pd.to_datetime("2019-10-01 00:00:00")
    fecha_fin = pd.to_datetime("2020-12-31 23:59:59")
    
    # Aplicamos la máscara de fechas
    mascara_fechas = (df_online["timestamp"] >= fecha_inicio) & (df_online["timestamp"] <= fecha_fin)
    df_online = df_online[mascara_fechas]
    
    # Ordenamos cronológicamente y por casa 
    df_online = df_online.sort_values(by=["cups_id", "timestamp"])

nombre_archivo_nuevo = "dataset_online.csv"
df_online.to_csv(nombre_archivo_nuevo, index=False)

print(f"El nuevo dataset '{nombre_archivo_nuevo}' ha sido guardado.")
print(f"Total de registros: {len(df_online)}")
print(f"Viviendas incluidas: {df_online['cups_id'].nunique()}")
print(f"Fecha inicial: {df_online['timestamp'].min()}")
print(f"Fecha final: {df_online['timestamp'].max()}")
resumen = df_online.groupby('ciudad_asignada')['cups_id'].nunique().reset_index()
resumen.columns = ['Ciudad', 'Nº Viviendas']
print("\nResumen por ciudad:")
print(resumen.to_string(index=False))