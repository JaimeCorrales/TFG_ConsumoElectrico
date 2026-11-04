"""
Relación temperatura-consumo: análisis global, por vivienda y con consumo normalizado.

Uso (desde la carpeta preprocesamiento):
    python correlacion_temperatura.py
    python correlacion_temperatura.py --ruta dataset_final_IA.csv --salida resultados_correlacion

Genera en la carpeta de salida:
    - correlaciones_por_vivienda.csv
    - curva_temperatura.csv
    - correlacion_temperatura_normalizada.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ANCHO_BIN = 2.0        # ºC por intervalo de temperatura
MIN_MUESTRAS_BIN = 100  # intervalos con menos observaciones no se dibujan


def cargar(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["timestamp"])
    df = df.dropna(subset=["kWh", "temperatura"]).copy()
    df["hora"] = df["timestamp"].dt.hour

    # Consumo relativo a la media de su vivienda (quita el efecto del nivel de consumo)
    df["kwh_rel"] = df["kWh"] / df.groupby("cups_id")["kWh"].transform("mean")

    # Consumo relativo a la media de su vivienda a esa hora (quita además el ciclo diario)
    media_hora = df.groupby(["cups_id", "hora"])["kWh"].transform("mean")
    df["kwh_rel_hora"] = np.where(media_hora > 0, df["kWh"] / media_hora, np.nan)
    return df


def correlaciones(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])[0], spearmanr(x[mask], y[mask])[0]


def r2_polinomio(x: np.ndarray, y: np.ndarray, grado: int) -> float:
    coef = np.polyfit(x, y, grado)
    pred = np.polyval(coef, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def tabla_por_vivienda(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for cups, g in df.groupby("cups_id"):
        t = g["temperatura"].to_numpy()
        y = g["kwh_rel"].to_numpy()
        r, rho = correlaciones(g["temperatura"], g["kwh_rel"])
        r_h, rho_h = correlaciones(g["temperatura"], g["kwh_rel_hora"])
        r2_lin = r2_polinomio(t, y, 1)
        r2_cuad = r2_polinomio(t, y, 2)
        coef = np.polyfit(t, y, 2)
        filas.append({
            "cups_id": cups,
            "ciudad": g["ciudad_asignada"].iloc[0],
            "n": len(g),
            "pearson": r,
            "spearman": rho,
            "pearson_sin_ciclo_diario": r_h,
            "spearman_sin_ciclo_diario": rho_h,
            "R2_lineal": r2_lin,
            "R2_cuadratico": r2_cuad,
            "forma_U": coef[0] > 0,  # coeficiente cuadrático positivo = parábola abierta hacia arriba
            "temp_minimo_curva": -coef[1] / (2 * coef[0]) if coef[0] != 0 else np.nan,
        })
    return pd.DataFrame(filas)


def curva(df: pd.DataFrame, col: str, grupo: str | None) -> pd.DataFrame:
    t_min = np.floor(df["temperatura"].min())
    t_max = np.ceil(df["temperatura"].max()) + ANCHO_BIN
    bordes = np.arange(t_min, t_max + ANCHO_BIN, ANCHO_BIN)
    d = df.copy()
    d["bin"] = pd.cut(d["temperatura"], bordes, right=False)
    claves = ["bin"] if grupo is None else [grupo, "bin"]
    res = (
        d.groupby(claves, observed=True)[col]
        .agg(media="mean", desv="std", n="count")
        .reset_index()
    )
    res["centro"] = res["bin"].apply(lambda b: b.left + ANCHO_BIN / 2).astype(float)
    if grupo is None:
        res["ciudad"] = "Global"
    else:
        res = res.rename(columns={grupo: "ciudad"})
    return res[["ciudad", "centro", "media", "desv", "n"]]


def figura(curvas: pd.DataFrame, ruta: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})

    for ciudad, c in curvas.groupby("ciudad"):
        c = c[c["n"] >= MIN_MUESTRAS_BIN]
        if ciudad == "Global":
            ax1.plot(c["centro"], c["media"], color="black", lw=2.5, label="Global")
        else:
            ax1.plot(c["centro"], c["media"], lw=1.3, alpha=0.8, label=ciudad)
    ax1.axhline(1, color="grey", ls="--", lw=0.8)
    ax1.set_xlabel("Temperatura (°C)")
    ax1.set_ylabel("Consumo relativo a la media de la vivienda")
    ax1.set_title(f"Curva consumo-temperatura (intervalos con n ≥ {MIN_MUESTRAS_BIN})")
    ax1.legend()

    g = curvas[curvas["ciudad"] == "Global"]
    ax2.bar(g["centro"], g["n"], width=ANCHO_BIN * 0.9, color="#2563EB")
    ax2.set_yscale("log")
    ax2.axhline(MIN_MUESTRAS_BIN, color="red", ls="--", lw=0.8)
    ax2.set_xlabel("Temperatura (°C)")
    ax2.set_ylabel("Observaciones por intervalo (escala log)")
    ax2.set_title("Muestras disponibles")

    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruta", default="dataset_final_IA.csv")
    ap.add_argument("--salida", default="resultados_correlacion")
    args = ap.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    df = cargar(Path(args.ruta))

    print("=== 1. Correlación global (todas las viviendas juntas) ===")
    r, rho = correlaciones(df["temperatura"], df["kWh"])
    print(f"kWh sin normalizar:             Pearson={r:+.4f}  Spearman={rho:+.4f}")
    r, rho = correlaciones(df["temperatura"], df["kwh_rel"])
    print(f"Consumo relativo a la vivienda: Pearson={r:+.4f}  Spearman={rho:+.4f}")
    r, rho = correlaciones(df["temperatura"], df["kwh_rel_hora"])
    print(f"Relativo a vivienda y hora:     Pearson={r:+.4f}  Spearman={rho:+.4f}")

    print("\n=== 2. Correlación por vivienda (consumo relativo) ===")
    tabla = tabla_por_vivienda(df)
    tabla.to_csv(salida / "correlaciones_por_vivienda.csv", index=False)
    cols = ["ciudad", "pearson", "spearman", "R2_lineal", "R2_cuadratico", "forma_U", "temp_minimo_curva"]
    print(tabla[cols].round(4).to_string(index=False))
    print("\nResumen (media ± desv. entre viviendas):")
    for c in ["pearson", "spearman", "pearson_sin_ciclo_diario", "spearman_sin_ciclo_diario", "R2_lineal", "R2_cuadratico"]:
        print(f"  {c:28s}: {tabla[c].mean():+.4f} ± {tabla[c].std():.4f}")
    print(f"  Viviendas con forma de U (coef. cuadrático > 0): {int(tabla['forma_U'].sum())}/{len(tabla)}")

    print("\n=== 3. Curva consumo-temperatura ===")
    curvas = pd.concat([
        curva(df, "kwh_rel", None),
        curva(df, "kwh_rel", "ciudad_asignada"),
    ], ignore_index=True)
    curvas.to_csv(salida / "curva_temperatura.csv", index=False)
    g = curvas[curvas["ciudad"] == "Global"]
    poco = g[g["n"] < MIN_MUESTRAS_BIN]
    print(f"Intervalos globales con n < {MIN_MUESTRAS_BIN}: {len(poco)} de {len(g)} "
          f"({poco['n'].sum()} de {g['n'].sum()} observaciones, {poco['n'].sum() / g['n'].sum() * 100:.2f} %)")
    figura(curvas, salida / "correlacion_temperatura_normalizada.png")
    print(f"\nResultados guardados en: {salida.resolve()}")


if __name__ == "__main__":
    main()