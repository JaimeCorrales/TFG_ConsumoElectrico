import os
import sys
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import TEMAS
from src.carga import cargar_dataset
from ui.dashboard import render_dashboard
from ui.batch_view import render_batch
from ui.online_view import render_online

# Configuración de la página
st.set_page_config(
    page_title="Predicción consumo eléctrico",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Estilos visuales
def aplicar_estilos(tema: dict) -> None:
    st.markdown(
        f"""
    <style>
        /* Fondo general */
        .stApp {{
            background-color: {tema['bg']};
            color: {tema['texto']};
        }}

        [data-testid="stSidebar"] {{
            background-color: {tema['bg_card']};
            border-right: 1px solid rgba(128, 128, 128, 0.15) !important; 
        }}

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {{
            color: {tema['texto']} !important;
        }}

        /* Métricas */
        [data-testid="stMetricLabel"] * {{
            color: {tema['texto']} !important;
        }}

        [data-testid="stMetricValue"] {{
            color: {tema['acento']} !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
        }}

        /* Tarjetas de métricas */
        [data-testid="stMetric"] {{
            background-color: {tema['bg_card']};
            border-radius: 10px;
            padding: 12px 16px;
            border-left: 3px solid {tema['acento']};
            border-top: 1px solid rgba(128, 128, 128, 0.1);
            border-right: 1px solid rgba(128, 128, 128, 0.1);
            border-bottom: 1px solid rgba(128, 128, 128, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); 
        }}

        /* Botones secundarios */
        .stButton button:not([kind="primary"]) {{
            background-color: {tema['acento']} !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }}

        .stButton button:not([kind="primary"]) * {{
            color: #FFFFFF !important;
        }}

        .stButton button:not([kind="primary"]):hover {{
            background-color: {tema['acento2']} !important;
            color: #FFFFFF !important;
            opacity: 0.9;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab"] {{
            color: {tema['texto']};
            opacity: 0.65;
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom: 3px solid {tema['acento']} !important;
            color: {tema['acento']} !important;
            opacity: 1;
            font-weight: bold;
        }}

        /* Botón de descarga */
        .stDownloadButton > button {{
            background-color: {tema['acento']};
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .stDownloadButton > button p,
        .stDownloadButton > button span {{
            color: #FFFFFF !important;
        }}

        /* Divider */
        hr {{
            border-color: rgba(128, 128, 128, 0.2); 
        }}

        /* Header */
        .tfg-header {{
            padding: 1.2rem 0 0.5rem 0;
            border-bottom: 2px solid {tema['acento']};
            margin-bottom: 1.5rem;
        }}
        .tfg-header h1 {{
            font-size: 1.8rem;
            font-weight: 800;
            color: {tema['texto']};
            margin: 0;
        }}
        .tfg-header p {{
            color: {tema['texto']};
            opacity: 0.7; 
            margin: 0.2rem 0 0 0;
            font-size: 0.85rem;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

# Barra lateral
def render_sidebar() -> tuple[str, dict]:
    with st.sidebar:
        st.markdown("## Predicción de demanda")
        st.caption("Sistema de análisis y modelos predictivos")
        st.divider()

        # Navegación principal
        pagina = st.radio(
            "Sección",
            [" Exploración", " Predicción batch", " Simulación online"],
            key="nav_pagina",
        )

        st.divider()

        # Paleta de colores
        st.markdown("### Apariencia")
        tema_nombre = st.selectbox(
            "Tema visual",
            list(TEMAS.keys()),
            index=0,
            key="tema_sel",
        )

        st.divider()

    return pagina, TEMAS[tema_nombre]


def main() -> None:
    pagina, tema = render_sidebar()
    aplicar_estilos(tema)

    # Header principal
    st.markdown(
        f"""
    <div class="tfg-header">
        <h1>Predicción de consumo eléctrico</h1>
        <p>Análisis de las predicciones de los modelos batch y online learning - Dataset Goiener Smart Meters</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Carga de datos en caché
    @st.cache_data(show_spinner="Cargando dataset principal...")
    def get_dataset_base():
        return cargar_dataset()

    @st.cache_data(show_spinner="Cargando dataset de 15 minutos...")
    def get_dataset_online():
        return cargar_dataset(
            ruta_archivo=str(BASE_DIR / "datos" / "dataset_online_15min.csv"),
            columnas_extra=["imputed", "es_interpolado"],
        )

    # Vistas
    if pagina == " Exploración":
        df = get_dataset_base()
        if df is not None:
            render_dashboard(df, tema)
        else:
            st.error(
                "No se pudo cargar el conjunto de datos histórico. Comprueba que "
                "existe 'datos/dataset_final_IA.csv' y que contiene las columnas "
                "requeridas."
            )

    elif pagina == " Predicción batch":
        render_batch(tema)

    elif pagina == " Simulación online":
        df_online = get_dataset_online()
        if df_online is not None:
            render_online(df_online, tema)
        else:
            st.warning("No se pudo iniciar la simulación por falta de datos.")


if __name__ == "__main__":
    main()