import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DIR_MODELOS = SCRIPT_DIR / "modelos"
DESTINO = SCRIPT_DIR.parent / "aplicacion_final" / "modelos" / "online" / "river_snarimax_warmup.pkl"

archivos = sorted(DIR_MODELOS.glob("snarimax_*_warmup.pkl"))
if not archivos:
    raise FileNotFoundError(f"No hay *_warmup.pkl en {DIR_MODELOS}. Ejecuta antes mod_snarimax.py.")

consolidado = {}
for ruta in archivos:
    with open(ruta, "rb") as f:
        art = pickle.load(f)
    consolidado[art["cups_id"]] = {
        "model": art["model"],
        "scaler": art["scaler"],
        "mejor_lr": art["mejor_lr"],
        "ciudad": art["ciudad"],
    }

DESTINO.parent.mkdir(parents=True, exist_ok=True)
with open(DESTINO, "wb") as f:
    pickle.dump(consolidado, f)

print(f"Consolidadas {len(consolidado)} viviendas en: {DESTINO}")