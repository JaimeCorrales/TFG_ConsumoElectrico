import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import COL_CIUDAD, COL_CUPS, COL_KWH, COL_TIMESTAMP
from src.metricas import calcular_metricas, exportar_csv
from src.prediccion_online import (
    cargar_modelo_snarimax_inicial,
    avanzar_simulacion,
)

DRIFT_INICIO = pd.Timestamp("2020-03-14")  # Inicio confinamiento España
DRIFT_FIN = pd.Timestamp("2020-06-21")  # Fin estado de alarma


def render_online(df_base: pd.DataFrame, tema: dict) -> None:

    st.markdown("## Simulación online learning - SNARIMAX")
    st.markdown(
        "El modelo procesa los datos muestra a muestra, predice y se actualiza internamente en memoria de forma automática, simulando un entorno de tiempo real. "
    )

    modelo_nombre = "SNARIMAX"
    modelo_clave = "snarimax"

    # Filtros sidebar
    with st.sidebar:
        st.markdown("### Configuración Online")
        st.info(f"**Modelo activo:** {modelo_nombre}")

        ciudades = sorted(df_base[COL_CIUDAD].dropna().unique().tolist())
        ciudad_sel = st.selectbox("Ciudad", ciudades, key="online_ciudad")

        df_ciudad = df_base[df_base[COL_CIUDAD] == ciudad_sel]
        cups_disp = sorted(df_ciudad[COL_CUPS].unique().tolist())

        def formato_casa(cups_id):
            return f"{ciudad_sel} (CUPS: {cups_id[:6]}...)"

        cups_sel = st.selectbox(
            "Vivienda",
            cups_disp,
            key="online_cups",
            format_func=formato_casa,
        )

        st.divider()
        ventana_dias = st.slider("Ventana MAE rolling (días)", 1, 30, 7)

    if not cups_sel:
        st.info("Selecciona una vivienda para iniciar la simulación.")
        return

    # Preparar los datos de la vivienda seleccionada
    df_casa = df_base[df_base[COL_CUPS] == cups_sel].copy()
    df_casa[COL_TIMESTAMP] = pd.to_datetime(df_casa[COL_TIMESTAMP])

    df_casa = df_casa.sort_values(COL_TIMESTAMP)
    df_casa = df_casa[df_casa[COL_TIMESTAMP] >= "2020-01-01"].reset_index(
        drop=True
    )

    max_rows = len(df_casa)
    muestras_por_dia = 96 

    if max_rows == 0:
        st.warning("No hay datos de 2020 en adelante para esta vivienda.")
        return

    # Gestión del estado
    if (
        "current_cups" not in st.session_state
        or st.session_state.current_cups != cups_sel
    ):
        st.session_state.current_cups = cups_sel
        st.session_state.sim_step = 0
        st.session_state.historial_preds = []
        st.session_state.modelo_vivo = cargar_modelo_snarimax_inicial(
            cups_sel
        )

    # Controles del simulador
    st.markdown("### Controles del simulador")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    pasos_a_avanzar = 0

    if c1.button(" Reiniciar", use_container_width=True):
        st.session_state.sim_step = 0
        st.session_state.historial_preds = []
        st.session_state.modelo_vivo = cargar_modelo_snarimax_inicial(
            cups_sel
        )

    if c2.button(" +1 Muestra", use_container_width=True):
        pasos_a_avanzar = 1

    if c3.button(
        " +1 Día", help=f"{muestras_por_dia} req", use_container_width=True
    ):
        pasos_a_avanzar = muestras_por_dia

    if c4.button(
        " +1 Sem.",
        help=f"{muestras_por_dia * 7} req",
        use_container_width=True,
    ):
        pasos_a_avanzar = muestras_por_dia * 7

    if c5.button(
        " Ir a Pandemia",
        help="Saltar al 14 de Marzo de 2020",
        use_container_width=True,
    ):
        drift_mask = df_casa[COL_TIMESTAMP] >= DRIFT_INICIO
        if drift_mask.any():
            indice_drift = int(drift_mask.idxmax()) + (muestras_por_dia * 7)
            pasos_a_avanzar = max(0, indice_drift - st.session_state.sim_step)
        else:
            st.warning("Los datos no llegan hasta la fecha del confinamiento.")

    if c6.button(
        " Procesar todo",
        help="Ejecutar todo el dataset de golpe",
        type="primary",
        use_container_width=True,
    ):
        pasos_a_avanzar = max_rows - st.session_state.sim_step

    # Bucle de inferencia precuencial
    if pasos_a_avanzar > 0 and st.session_state.modelo_vivo is not None:
        inicio = st.session_state.sim_step
        fin = min(inicio + pasos_a_avanzar, max_rows)

        with st.spinner("Procesando y aprendiendo en tiempo real..."):
            
            # Llamamos al script que procesa el modelo en River
            nuevas_preds = avanzar_simulacion(
                df_casa=df_casa, 
                inicio=inicio, 
                fin=fin, 
                obj_modelo=st.session_state.modelo_vivo
            )
            
            # Actualizamos el estado de Streamlit añadiendo la nueva lista
            st.session_state.historial_preds.extend(nuevas_preds)

        st.session_state.sim_step = fin

    progreso = st.session_state.sim_step / max_rows if max_rows > 0 else 0
    st.progress(
        progreso,
        text=f"Progreso: {st.session_state.sim_step} de {max_rows} registros procesados",
    )

    # Resultados y métricas
    if len(st.session_state.historial_preds) > 0:
        df_plot = pd.DataFrame(st.session_state.historial_preds)

        # MAE rolling dinámico
        ventana_muestras = muestras_por_dia * ventana_dias
        df_plot["error_abs"] = (
            df_plot["Consumo_Real"] - df_plot["Prediccion"]
        ).abs()
        df_plot["mae_rolling"] = (
            df_plot["error_abs"]
            .rolling(window=ventana_muestras, min_periods=1)
            .mean()
        )

        metricas = calcular_metricas(
            df_plot["Consumo_Real"].values, df_plot["Prediccion"].values
        )

        st.markdown(f"### Rendimiento acumulado — {modelo_nombre}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{metricas['mae']} kWh")
        c2.metric("RMSE", f"{metricas['rmse']} kWh")
        c3.metric(
            "WAPE",
            f"{metricas['wape']} %" if metricas.get("wape") else "N/A",
        )
        c4.metric("R²", f"{metricas['r2']}")

        st.divider()

        tab1, tab2 = st.tabs([
            "Curva de aprendizaje",
            "Predicción vs Real",
        ])

        with tab1:
            _render_curva_aprendizaje(df_plot, tema)

        with tab2:
            _render_prediccion_real(df_plot, modelo_nombre, tema)

        st.divider()
        st.markdown("### Exportar resultados")
        csv_bytes = exportar_csv(df_plot, metricas)
        st.download_button(
            label=" Descargar historial CSV",
            data=csv_bytes,
            file_name=f"online_sesion_{modelo_clave}_{cups_sel[:8]}.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Pulsa en los controles superiores para avanzar la simulación, generar predicciones y calcular las métricas."
        )


def _render_curva_aprendizaje(df_plot: pd.DataFrame, tema: dict) -> None:
    st.markdown("#### MAE progresivo en el tiempo")
    st.caption(
        "El MAE rolling muestra cómo el modelo SNARIMAX se adapta a los cambios. "
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["timestamp"],
            y=df_plot["mae_rolling"],
            name="MAE rolling",
            fill="tozeroy",
            line=dict(color=tema["acento"], width=2),
            fillcolor=f'rgba({_hex_to_rgb(tema["acento"])},0.15)',
        )
    )

    fig.add_vrect(
        x0=DRIFT_INICIO,
        x1=DRIFT_FIN,
        fillcolor=tema["prediccion"],
        opacity=0.12,
        layer="below",
        line_width=0,
        annotation_text="Confinamiento",
        annotation_position="top left",
        annotation_font_color=tema["prediccion"],
    )

    fig.update_layout(
        template=tema["plotly"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=tema["texto"], 
        height=400,
        xaxis_title="Fecha",
        yaxis_title="MAE (kWh)",
        legend=dict(
            orientation="h", 
            y=-0.15,
            font=dict(color=tema["texto"])
        ),
        margin=dict(l=0, r=0, t=20, b=0),
    )

    fig.update_xaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"])
    )
    fig.update_yaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"])
    )

    st.plotly_chart(fig, use_container_width=True, theme=None)


def _render_prediccion_real(
    df_plot: pd.DataFrame, modelo_nombre: str, tema: dict
) -> None:
    st.markdown("#### Predicción vs consumo real")

    if len(df_plot) < 2:
        st.warning(
            "Se necesitan al menos 2 registros para visualizar la serie temporal."
        )
        return

    fecha_min = df_plot["timestamp"].min().date()
    fecha_max = df_plot["timestamp"].max().date()

    if fecha_min == fecha_max:
        rango = (fecha_min, fecha_max)
    else:
        rango = st.date_input(
            "Zoom temporal",
            value=(fecha_min, fecha_max),
            min_value=fecha_min,
            max_value=fecha_max,
            key="online_zoom",
        )

    if isinstance(rango, tuple) and len(rango) == 2:
        mask = (df_plot["timestamp"].dt.date >= rango[0]) & (
            df_plot["timestamp"].dt.date <= rango[1]
        )
        df_zoom = df_plot[mask]
    else:
        df_zoom = df_plot

    dias = (
        (rango[1] - rango[0]).days
        if (isinstance(rango, tuple) and len(rango) == 2)
        else 365
    )
    if dias > 60:
        df_zoom_plot = (
            df_zoom.set_index("timestamp")[["Consumo_Real", "Prediccion"]]
            .resample("D")
            .mean()
            .reset_index()
        )
    else:
        df_zoom_plot = df_zoom

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_zoom_plot["timestamp"],
            y=df_zoom_plot["Consumo_Real"],
            name="Consumo real",
            line=dict(color=tema["real"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_zoom_plot["timestamp"],
            y=df_zoom_plot["Prediccion"],
            name=f"Predicción {modelo_nombre}",
            line=dict(color=tema["prediccion"], width=2, dash="dot"),
        )
    )

    fig.add_vrect(
        x0=DRIFT_INICIO,
        x1=DRIFT_FIN,
        fillcolor=tema["prediccion"],
        opacity=0.10,
        layer="below",
        line_width=0,
    )

    fig.update_layout(
        template=tema["plotly"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=tema["texto"], 
        height=400,
        xaxis_title="Fecha",
        yaxis_title="kWh",
        legend=dict(
            orientation="h", 
            y=-0.15,
            font=dict(color=tema["texto"])
        ),
        margin=dict(l=0, r=0, t=20, b=0),
    )

    fig.update_xaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"])
    )
    fig.update_yaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"])
    )

    st.plotly_chart(fig, use_container_width=True, theme=None)

def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"