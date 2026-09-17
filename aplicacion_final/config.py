import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS       = os.path.join(BASE_DIR, "datos")
DIR_MODELOS     = os.path.join(BASE_DIR, "modelos")
DIR_BATCH       = os.path.join(DIR_MODELOS, "batch")
DIR_ONLINE      = os.path.join(DIR_MODELOS, "online")
ARCHIVO_DATASET = os.path.join(DIR_DATOS, "dataset_final_IA.csv")

MODELOS_BATCH = {
    # "SARIMAX":  "sarimax",
    # "Prophet":  "prophet",
    # "LSTM":     "lstm",
    # "XGBoost":  "xgboost",
     "Ridge":    "ridge",
}

MODELOS_ONLINE = {
    # "SGD Linear": "sgd_linear",
    "SNARIMAX":   "snarimax",
    # "HAT":        "hat",
    # "ARF":        "arf",
    # "Bandit":     "bandit",
}

METRICAS_NOMBRES = {
    "mae":  "MAE (kWh)",
    "rmse": "RMSE (kWh)",
    "wape": "WAPE (%)",
    "r2":   "R²",
}

TEMAS = {
    "Claro": {
        "bg":          "#FFFFFF",
        "bg_card":     "#F7F9FC",
        "texto":       "#000000",
        "acento":      "#2563EB",
        "acento2":     "#F59E0B",
        "real":        "#2563EB",
        "prediccion":  "#EF4444",
        "grid":        "#E5E7EB",
        "plotly":      "plotly_white",
    },
    "Oscuro": {
        "bg":          "#0F172A",
        "bg_card":     "#1E293B",
        "texto":       "#F1F5F9",
        "acento":      "#38BDF8",
        "acento2":     "#FACC15",
        "real":        "#38BDF8",
        "prediccion":  "#F87171",
        "grid":        "#334155",
        "plotly":      "plotly_dark",
    },
    "Alto Contraste": {
        "bg":          "#000000",
        "bg_card":     "#1A1A1A",
        "texto":       "#FFFFFF",
        "acento":      "#FFD700",
        "acento2":     "#00FF88",
        "real":        "#FFD700",
        "prediccion":  "#FF4444",
        "grid":        "#444444",
        "plotly":      "plotly_dark",
    },
}

# Columnas del dataset 
COL_TIMESTAMP = "timestamp"
COL_KWH       = "kWh"
COL_TEMP      = "temperatura"
COL_CIUDAD    = "ciudad_asignada"
COL_CUPS      = "cups_id"
