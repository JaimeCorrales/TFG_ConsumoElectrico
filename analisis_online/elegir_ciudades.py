import pandas as pd

df = pd.read_csv("dataset_final_IA.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Período pre-confinamiento: enero 2020 
# Período confinamiento puro: abril 2020 
mask_pre   = (df["timestamp"].dt.year == 2020) & (df["timestamp"].dt.month == 1)
mask_post  = (df["timestamp"].dt.year == 2020) & (df["timestamp"].dt.month == 4)

df_pre  = df[mask_pre].groupby(["ciudad_asignada", "cups_id"])["kWh"].mean().rename("kwh_enero")
df_conf = df[mask_post].groupby(["ciudad_asignada", "cups_id"])["kWh"].mean().rename("kwh_abril")

# Unir y calcular impacto
pivot = pd.concat([df_pre, df_conf], axis=1).reset_index()
pivot["impacto_abs"] = (pivot["kwh_abril"] - pivot["kwh_enero"]).abs()
pivot["impacto_pct"] = ((pivot["kwh_abril"] - pivot["kwh_enero"]) / pivot["kwh_enero"] * 100).round(2)
pivot["direccion"]   = pivot["kwh_abril"].gt(pivot["kwh_enero"]).map({True: "subió", False: "bajó"})

# Ranking por ciudad
ranking = pivot.sort_values(
    by=["ciudad_asignada", "impacto_abs"],
    ascending=[True, False]
)

print("----RANKING DE CONCEPT DRIFT POR CIUDAD (enero vs abril 2020)----")
print(ranking[[
    "ciudad_asignada", "cups_id",
    "kwh_enero", "kwh_abril",
    "impacto_pct", "direccion"
]].to_string(index=False))

# Casa representativa por ciudad 
print("----CASA REPRESENTATIVA POR CIUDAD PARA LAS GRÁFICAS DEL TFG----")
representativas = ranking.loc[
    ranking.groupby("ciudad_asignada")["impacto_abs"].idxmax()
]
print(representativas[[
    "ciudad_asignada", "cups_id", "impacto_pct", "direccion"
]].to_string(index=False))

# Guardar
ranking.to_csv("drift_pandemia_por_casa.csv", index=False)
representativas.to_csv("casas_representativas_online.csv", index=False)
print("\n Guardados: drift_pandemia_por_casa.csv y casas_representativas_online.csv")