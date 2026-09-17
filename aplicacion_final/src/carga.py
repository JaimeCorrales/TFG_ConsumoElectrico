import os
import pandas as pd
from config import (
    ARCHIVO_DATASET,
    COL_CIUDAD,
    COL_CUPS,
    COL_KWH,
    COL_TEMP,
    COL_TIMESTAMP,
)

COLUMNAS_REQUERIDAS_BASE = [
    COL_TIMESTAMP,
    COL_KWH,
    COL_TEMP,
    COL_CIUDAD,
    COL_CUPS,
]


def cargar_dataset(
    ruta_archivo: str = ARCHIVO_DATASET, columnas_extra: list | None = None
) -> pd.DataFrame | None:
    """Carga, valida y limpia el dataset principal o de simulación."""
    if not os.path.exists(ruta_archivo):
        return None

    try:
        df = pd.read_csv(ruta_archivo)
    except Exception:
        return None

    # Validar columnas requeridas
    cols_a_validar = COLUMNAS_REQUERIDAS_BASE.copy()
    if columnas_extra:
        cols_a_validar.extend(columnas_extra)

    faltantes = [c for c in cols_a_validar if c not in df.columns]
    if faltantes:
        return None

    # Convertir formato temporal
    try:
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
    except Exception:
        return None

    # Normalizar columnas de interpolación si existen
    if "imputed" in df.columns:
        df["imputed"] = df["imputed"].astype(bool)
    if "es_interpolado" in df.columns:
        df["es_interpolado"] = df["es_interpolado"].astype(bool)

    df = df.sort_values([COL_CUPS, COL_TIMESTAMP]).reset_index(drop=True)
    return df


def filtrar_datos(
    df: pd.DataFrame,
    ciudad: str | None = None,
    cups: str | None = None,
    fecha_inicio=None,
    fecha_fin=None,
) -> pd.DataFrame:
    """Aplica los filtros espaciales y temporales sobre un DataFrame."""
    mask = pd.Series([True] * len(df), index=df.index)

    if ciudad and ciudad != "Todas":
        mask &= df[COL_CIUDAD] == ciudad
    if cups and cups != "Todos":
        mask &= df[COL_CUPS] == cups
    if fecha_inicio is not None:
        mask &= df[COL_TIMESTAMP] >= pd.Timestamp(fecha_inicio)
    if fecha_fin is not None:
        mask &= df[COL_TIMESTAMP] <= pd.Timestamp(fecha_fin)

    return df[mask].copy()