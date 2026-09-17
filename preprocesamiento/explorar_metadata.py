import pandas as pd

print("Cargando metadatos...")
df_meta = pd.read_csv('datos_auxiliares/metadata.csv')

print("\n---- Análisis de casas disponibles (> 3 años) ----")

# Solo tarifa residencial (2.0TD) y más de 3 años de historial
df_filtrado = df_meta[
    (df_meta['contracted_tariff'] == '2.0TD') & 
    (df_meta['length_years'] > 3)
]

print(f"Total de hogares residenciales en España con más de 3 años: {df_filtrado.shape[0]}")

# Conteo por provincia 
print("\nNúmero de hogares con > 3 años por PROVINCIA:")
provincias_top = df_filtrado['province'].value_counts()
print(provincias_top)

# Conteo por ciudad
print("\nNúmero de hogares con > 3 años por CIUDAD (Municipios > 50.000 hab.):")
ciudades_top = df_filtrado['municipality'].value_counts()

if ciudades_top.empty:
    print("No hay ciudades visibles con más de 3 años.")
else:
    print(ciudades_top.head(20))
