import pandas as pd
import numpy as np

def analizar_regularidad_horaria(ruta_csv):
    print("Cargando y analizando estructura de datos horarios")
    df = pd.read_csv(ruta_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Identificar columna de ciudad
    col_ciudad = "ciudad_asignada" if "ciudad_asignada" in df.columns else "ciudad"
    
    metricas_ciudades = []

    for ciudad, group in df.groupby(col_ciudad):
        # Asegurar orden cronológico
        group = group.sort_values("timestamp")
        
        kwh = group["kWh"]
        mean_val = kwh.mean()
        std_val = kwh.std()
        
        # Coeficiente de variación 
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        
        # Autocorrelación estacional (lag 24 = 24 horas en intervalos de 1 hora)
        r24 = kwh.autocorr(lag=24)
        
        # Ruido de alta frecuencia
        ruido_puntual = kwh.diff().std()
        
        metricas_ciudades.append({
            "Ciudad": ciudad,
            "Consumo Medio (kWh)": round(mean_val, 4),
            "CV (%) [Volatilidad]": round(cv, 2),
            "r_24 [Patrón Diario]": round(r24, 4),
            "Ruido Puntual (1h)": round(ruido_puntual, 4)
        })

    res_df = pd.DataFrame(metricas_ciudades)
    
    # Ordenar por regularidad (Mayor r_24 = Más regular / Menor r_24 = Más caótica)
    res_df = res_df.sort_values(by="r_24 [Patrón Diario]", ascending=False).reset_index(drop=True)
    
    return res_df

if __name__ == "__main__":
    ruta = "dataset_final_IA.csv" 
    tabla_diagnostico = analizar_regularidad_horaria(ruta)
    
    print("----Diagnóstico de regularidad de consumo por ciudad----")
    print(tabla_diagnostico.to_string(index=False))
    
    mas_regular = tabla_diagnostico.iloc[0]["Ciudad"]
    mas_caotica = tabla_diagnostico.iloc[-1]["Ciudad"]
    
    print("\nConclusiones:")
    print(f" - Ciudad más regular: {mas_regular}")
    print(f" - Ciudad más caótica: {mas_caotica}")