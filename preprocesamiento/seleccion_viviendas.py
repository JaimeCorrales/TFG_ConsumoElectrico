import pandas as pd
import numpy as np
import itertools
import random
import os

RUTA_METADATOS     = os.path.join("datos_auxiliares", "metadata.csv")
RUTA_SALIDA        = os.path.join("datos_auxiliares", "casas_seleccionadas_global.csv")
MIN_AÑOS_SOLAPE    = 2.5
MAX_PCT_NULOS      = 5.0
N_CASAS_POR_CIUDAD = 3

CIUDADES_OBJETIVO = ["Bilbao", "Madrid", "Coruna A", "Valencia"]

# Cargar metadatos
print("-----PASO 1: Cargando metadatos-----")

if not os.path.exists(RUTA_METADATOS):
    print(f" No se encuentra el archivo '{RUTA_METADATOS}'.")
    exit()

meta = pd.read_csv(RUTA_METADATOS)
meta = meta.rename(columns={meta.columns[0]: "cups_id"})

meta["contract_start_date"] = pd.to_datetime(meta["start_date"], errors="coerce")
meta["contract_end_date"]   = pd.to_datetime(meta["end_date"],   errors="coerce").fillna(pd.Timestamp("2022-06-05"))
meta = meta[meta["contract_start_date"].notna()]

meta["municipality_clean"] = (
    meta["municipality"]
    .astype(str).str.lower()
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
)

dicc_ciudades = {}
for ciudad in CIUDADES_OBJETIVO:
    sub = meta[meta["municipality_clean"].str.contains(ciudad.lower(), na=False)].copy()
    sub_calidad = sub[sub["missing_samples_pct"] <= MAX_PCT_NULOS]

    if len(sub_calidad) >= N_CASAS_POR_CIUDAD:
        sub_calidad = sub_calidad.copy()
        sub_calidad["score_ind"] = sub_calidad["length_years"] * 10 - sub_calidad["missing_samples_pct"]

        dicc_ciudades[ciudad] = sub_calidad.nlargest(20, "score_ind").to_dict("records")
        print(f"   - {ciudad}: {len(sub_calidad)} casas de calidad disponibles.")
    else:
        print(f" {ciudad} no tiene suficientes casas de calidad.")
        exit()

# Búsqueda del bloque temporal global

print("----PASO 2: Buscando el mejor bloque temporal compartido globalmente----")

combos_por_ciudad = {
    ciu: list(itertools.combinations(lista, N_CASAS_POR_CIUDAD))
    for ciu, lista in dicc_ciudades.items()
}

mejor_solape_global = None
max_años_global     = 0.0
mejor_inicio_global = None
mejor_fin_global    = None

random.seed(42)

for _ in range(50000):
    combo_actual = []
    for ciu in CIUDADES_OBJETIVO:
        combo_actual.extend(random.choice(combos_por_ciudad[ciu]))

    inicio_comun = max(casa["contract_start_date"] for casa in combo_actual)
    fin_comun    = min(casa["contract_end_date"]   for casa in combo_actual)

    años = (fin_comun - inicio_comun).days / 365.25

    if años > max_años_global:
        max_años_global     = años
        mejor_solape_global = combo_actual
        mejor_inicio_global = inicio_comun
        mejor_fin_global    = fin_comun

# Validación y exportación
if max_años_global >= MIN_AÑOS_SOLAPE:
    print(f"Solape simultaneo encontrado: {max_años_global:.2f} años.")
    print(f"Ventana global: {mejor_inicio_global.date()} -> {mejor_fin_global.date()}")

    df_final = pd.DataFrame(mejor_solape_global)
    df_final["ciudad_asignada"] = df_final["municipality"].apply(
        lambda x: next((c for c in CIUDADES_OBJETIVO if c.lower() in str(x).lower()), "")
    )
    df_final["arco_inicio_comun"] = str(mejor_inicio_global.date())
    df_final["arco_fin_comun"]    = str(mejor_fin_global.date())
    df_final["años_solape_real"]  = round(max_años_global, 2)

    columnas_guardar = [
        "cups_id", "ciudad_asignada", "province", "length_years",
        "missing_samples_pct", "start_date", "end_date",
        "arco_inicio_comun", "arco_fin_comun"
    ]
    df_final[[c for c in columnas_guardar if c in df_final.columns]].to_csv(
        RUTA_SALIDA, index=False
    )
    print(f"\nGuardado con éxito: '{RUTA_SALIDA}'")

else:
    print(f"\n Imposible alinear las 4 ciudades durante {MIN_AÑOS_SOLAPE} años.")
    print(f"   Mejor solape encontrado: {max_años_global:.2f} años.")