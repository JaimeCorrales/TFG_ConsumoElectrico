import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import COL_CIUDAD, COL_CUPS, MODELOS_BATCH
from src.metricas import calcular_metricas, exportar_csv
from src.prediccion_batch import cargar_dataset_features, ejecutar_inferencia_batch


def render_batch(tema: dict) -> None:
    # Usamos caché de Streamlit a nivel de vista para no sobrecargar el disco
    @st.cache_data
    def get_features_cached():
        return cargar_dataset_features()

    df_trabajo = get_features_cached()

    if df_trabajo is None:
        st.error(
            "No se encuentra el archivo 'modelos/batch/dataset_features.csv'."
        )
        return

    FECHA_CORTE = pd.to_datetime("2021-06-05")
    df_trabajo = df_trabajo[df_trabajo["timestamp"] >= FECHA_CORTE]

    if df_trabajo.empty:
        st.warning(
            "No hay datos de test disponibles a partir de 2021-06-05 en el dataset."
        )
        return

    # Filtros sidebar
    with st.sidebar:
        st.markdown("### Configuración Inicial")
        
        st.info("Modelo activo: Regression Ridge")
        
        modelo_nombre = "Regression Ridge"
        modelo_clave = MODELOS_BATCH.get("Regression Ridge", "ridge") 

        ciudades = sorted(df_trabajo[COL_CIUDAD].dropna().unique().tolist())
        ciudad_sel = st.selectbox("Ciudad", ciudades, key="batch_ciudad")

        cups_disp = (
            df_trabajo[df_trabajo[COL_CIUDAD] == ciudad_sel][COL_CUPS]
            .unique()
            .tolist()
        )
        cups_sel = st.selectbox(
            "Vivienda (CUPS)", sorted(cups_disp), key="batch_cups"
        )
    st.markdown(f"## Predicción a largo plazo — {modelo_nombre}")

    # Filtrado inicial del dataset
    df_filtrado = df_trabajo[
        (df_trabajo[COL_CIUDAD] == ciudad_sel)
        & (df_trabajo[COL_CUPS] == cups_sel)
    ]

    if df_filtrado.empty:
        st.warning(
            "No hay datos disponibles para los filtros geográficos seleccionados."
        )
        return

    # Filtro de rango temporal
    st.markdown("### Zoom temporal")
    min_date = df_filtrado["timestamp"].min().date()
    max_date = df_filtrado["timestamp"].max().date()

    rango_fechas = st.date_input(
        "Selecciona el rango de fechas a analizar",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="batch_date_range",
    )

    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        start_date, end_date = rango_fechas
    elif isinstance(rango_fechas, tuple) and len(rango_fechas) == 1:
        start_date = end_date = rango_fechas[0]
    else:
        start_date = end_date = rango_fechas

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1, seconds=-1)

    df_filtrado = df_filtrado[
        (df_filtrado["timestamp"] >= start_dt)
        & (df_filtrado["timestamp"] <= end_dt)
    ]

    if df_filtrado.empty:
        st.warning("No hay datos en el rango temporal seleccionado.")
        return

    # Inferencia usando la clave del modelo seleccionado
    with st.spinner(f"Ejecutando inferencia con {modelo_nombre}..."):
        df_plot = ejecutar_inferencia_batch(df_filtrado, clave=modelo_clave)

    if df_plot.empty:
        st.error(
            "No se pudieron cargar los modelos .joblib para las viviendas"
            " seleccionadas."
        )
        return

    # Métricas dinámicas
    metricas = calcular_metricas(
        df_plot["Consumo_Real"].values, df_plot["Prediccion"].values
    )

    st.markdown(f"### Métricas — {modelo_nombre}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{metricas['mae']} kWh")
    c2.metric("RMSE", f"{metricas['rmse']} kWh")
    c3.metric(
        "WAPE", f"{metricas['wape']} %" if metricas.get("wape") else "N/A"
    )
    c4.metric("R²", f"{metricas['r2']}")

    st.divider()

    # Gráficas
    st.markdown("### Consumo real vs predicción")
    tab1, tab2 = st.tabs(["Serie temporal", "Dispersión"])

    with tab1:
        _render_serie_temporal(df_plot, modelo_nombre, tema)

    with tab2:
        _render_dispersion(df_plot, tema)

    # Exportación
    st.divider()
    st.markdown("### Exportar resultados")
    csv_bytes = exportar_csv(df_plot, metricas)
    st.download_button(
        label="Descargar predicciones y métricas (CSV)",
        data=csv_bytes,
        file_name=f"predicciones_batch_{modelo_clave}.csv",
        mime="text/csv",
    )


def _render_serie_temporal(df_plot: pd.DataFrame, modelo_nombre: str, tema: dict) -> None:
    """Renderiza el gráfico de líneas temporal de consumo vs predicción."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["timestamp"],
            y=df_plot["Consumo_Real"],
            name="Consumo real",
            line=dict(color=tema["real"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot["timestamp"],
            y=df_plot["Prediccion"],
            name=f"Predicción {modelo_nombre}",
            line=dict(color=tema["prediccion"], width=2, dash="dot"),
        )
    )
    fig.update_layout(
        template=tema["plotly"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=tema["texto"],
        legend=dict(
            orientation="h", 
            y=-0.15,
            font=dict(color=tema["texto"]) 
        ),
        height=420,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Fecha",
        yaxis_title="kWh",
    )
    fig.update_xaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"]),
    )
    fig.update_yaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"]),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


def _render_dispersion(df_plot: pd.DataFrame, tema: dict) -> None:
    """Renderiza el scatter plot (dispersión) comparando real vs predicción."""
    muestra = df_plot.sample(n=min(3000, len(df_plot)), random_state=42)
    max_val = max(
        muestra["Consumo_Real"].max(), muestra["Prediccion"].max()
    )

    fig_disp = go.Figure()
    fig_disp.add_trace(
        go.Scatter(
            x=muestra["Consumo_Real"],
            y=muestra["Prediccion"],
            mode="markers",
            marker=dict(color=tema["acento"], opacity=0.35, size=4),
            name="Muestras",
        )
    )
    fig_disp.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color=tema["prediccion"], dash="dash", width=2),
            name="Predicción perfecta",
        )
    )
    fig_disp.update_layout(
        template=tema["plotly"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=tema["texto"],
        height=420,
        xaxis_title="Consumo real (kWh)",
        yaxis_title="Predicción (kWh)",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    fig_disp.update_xaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"]),
    )
    fig_disp.update_yaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"]),
    )
    st.plotly_chart(fig_disp, use_container_width=True, theme=None)