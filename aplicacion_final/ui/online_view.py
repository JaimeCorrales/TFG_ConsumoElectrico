import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import COL_CIUDAD, COL_CUPS, COL_TIMESTAMP
from src.metricas import calcular_metricas, exportar_csv
from src.prediccion_online import (
    cargar_modelo_snarimax_inicial,
    avanzar_simulacion,
)

DRIFT_INICIO = pd.Timestamp("2020-03-14")
DRIFT_FIN = pd.Timestamp("2020-06-21")

MUESTRAS_POR_DIA = 96
DIAS_VENTANA_DINAMICA = 7  
MAX_PUNTOS_GRAFICO = 3000

OPCIONES_ZOOM = {
    "Últimas 24 h": 1,
    "Últimos 7 días": 7,
    "Últimos 30 días": 30,
    "Todo": None,
}

# Utilidades
def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"

def _fmt(valor, sufijo: str = "") -> str:
    if valor is None:
        return "N/A"
    try:
        if math.isnan(float(valor)):
            return "N/A"
    except (TypeError, ValueError):
        return str(valor)
    return f"{valor}{sufijo}"

def _submuestrear(df: pd.DataFrame, max_puntos: int = MAX_PUNTOS_GRAFICO):
    if len(df) <= max_puntos:
        return df
    paso = math.ceil(len(df) / max_puntos)
    return df.iloc[::paso]

def _metricas_seguras(df: pd.DataFrame):
    if len(df) < 2:
        return None
    try:
        return calcular_metricas(
            df["Consumo_Real"].values, df["Prediccion"].values
        )
    except Exception:
        return None

def _mostrar_metricas(m) -> None:
    c1, c2, c3, c4 = st.columns(4)
    if m is None:
        for c, nombre in zip((c1, c2, c3, c4), ("MAE", "RMSE", "WAPE", "R²")):
            c.metric(nombre, "N/A")
        return
    c1.metric("MAE", _fmt(m.get("mae"), " kWh"))
    c2.metric("RMSE", _fmt(m.get("rmse"), " kWh"))
    c3.metric("WAPE", _fmt(m.get("wape"), " %"))
    c4.metric("R²", _fmt(m.get("r2")))

def _layout_comun(fig: go.Figure, tema: dict, ytitle: str) -> None:
    fig.update_layout(
        template=tema["plotly"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=tema["texto"],
        height=400,
        xaxis_title="Fecha",
        yaxis_title=ytitle,
        legend=dict(
            orientation="h", y=-0.15, font=dict(color=tema["texto"])
        ),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    for actualizar in (fig.update_xaxes, fig.update_yaxes):
        actualizar(
            color=tema["texto"],
            tickfont=dict(color=tema["texto"]),
            title_font=dict(color=tema["texto"]),
        )

def _reiniciar_simulacion(cups_sel: str) -> None:
    st.session_state.sim_step = 0
    st.session_state.historial_preds = []
    st.session_state.modelo_vivo = cargar_modelo_snarimax_inicial(cups_sel)

def _avanzar(df_casa: pd.DataFrame, pasos: int, max_rows: int) -> None:
    if pasos <= 0 or st.session_state.modelo_vivo is None:
        return
    inicio = st.session_state.sim_step
    fin = min(inicio + pasos, max_rows)
    if fin <= inicio:
        return
    nuevas_preds = avanzar_simulacion(
        df_casa=df_casa,
        inicio=inicio,
        fin=fin,
        obj_modelo=st.session_state.modelo_vivo,
    )
    st.session_state.historial_preds.extend(nuevas_preds)
    st.session_state.sim_step = fin

# Vista principal
def render_online(df_base: pd.DataFrame, tema: dict) -> None:
    css_widgets = f"""
    <style>
        /* Solo las etiquetas de los widgets (títulos de sliders, selectboxes, etc) */
        label[data-testid="stWidgetLabel"] p, 
        label[data-testid="stWidgetLabel"] span {{
            color: {tema['texto']} !important;
        }}
        
        /* Solo los números y valores dentro de los Sliders */
        div[data-baseweb="slider"] div,
        div[data-testid="stTickBar"] div {{
            color: {tema['texto']} !important;
        }}
        
        /* Solo el texto del Toggle (interruptor) */
        div[data-testid="stCheckbox"] p,
        label[data-testid="stCheckbox"] p {{
            color: {tema['texto']} !important;
        }}
    </style>
    """
    st.markdown(css_widgets, unsafe_allow_html=True)
    st.markdown("## Simulación online learning - SNARIMAX")
    st.markdown(
        "El modelo procesa los datos muestra a muestra, predice y se "
        "actualiza de forma automática, simulando "
        "un entorno de tiempo real."
    )

    modelo_nombre = "SNARIMAX"
    modelo_clave = "snarimax"

    with st.sidebar:
        st.markdown("### Configuración Inicial")
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

    if not cups_sel:
        st.info("Selecciona una vivienda para iniciar la simulación.")
        return

    df_casa = df_base[df_base[COL_CUPS] == cups_sel].copy()
    df_casa[COL_TIMESTAMP] = pd.to_datetime(df_casa[COL_TIMESTAMP])
    df_casa = df_casa.sort_values(COL_TIMESTAMP)
    df_casa = df_casa[df_casa[COL_TIMESTAMP] >= "2020-01-01"].reset_index(
        drop=True
    )

    max_rows = len(df_casa)
    if max_rows == 0:
        st.warning("No hay datos de 2020 en adelante para esta vivienda.")
        return

    if st.session_state.pop("online_fin_autoplay", False):
        st.session_state["online_autoplay"] = False

    if (
        "current_cups" not in st.session_state
        or st.session_state.current_cups != cups_sel
    ):
        st.session_state.current_cups = cups_sel
        st.session_state["online_autoplay"] = False
        _reiniciar_simulacion(cups_sel)

    modelo_vivo = st.session_state.get("modelo_vivo")
    if isinstance(modelo_vivo, dict):
        if modelo_vivo.get("origen") == "warmup":
            st.success(
                "Modelo calentado con datos hasta 2019-12-31 "
                f"(learning rate seleccionado: {modelo_vivo.get('mejor_lr')})."
            )
        else:
            st.warning(
                "Cold start: no se ha cargado el modelo calentado. El modelo "
                "parte sin entrenar y las métricas no serán comparables con "
                f"las del capítulo 4. Motivo: {modelo_vivo.get('motivo', 'desconocido')}"
            )

    # Controles de simulación
    st.markdown("---")
    st.markdown("#### Control de simulación en tiempo real")
    st.caption(
        "Ajusta el **ritmo de actualización** visual de la pantalla y la **cantidad de datos "
        "(salto de tiempo)** que el modelo procesa de golpe en cada una de esas actualizaciones."
    )
    
    col_play, col_speed, col_batch = st.columns([1.5, 2, 2])
    
    estado_play = st.session_state.get("online_autoplay", False)
    label_play = "🟢 Tiempo Real: ACTIVO" if estado_play else "⚪ Tiempo Real: APAGADO"
    
    auto_play = col_play.toggle(
        label_play,
        key="online_autoplay",
        help="Simula la llegada de datos en streaming de manera automática.",
    )
    velocidad = col_speed.slider(
        "Segundos entre refrescos",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        disabled=not auto_play,
    )
    muestras_tick = col_batch.select_slider(
        "Muestras procesadas por salto",
        options=[1, 4, 24, 96],
        value=1,
        disabled=not auto_play,
        help="1 = 15 min, 4 = 1 h, 24 = 6 h, 96 = 1 día.",
    )

    st.markdown("#### Avance Manual")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    pasos_manual = 0

    if c1.button("Reiniciar", use_container_width=True):
        _reiniciar_simulacion(cups_sel)

    if not auto_play:
        if c2.button("+1 Muestra", use_container_width=True):
            pasos_manual = 1
        if c3.button("+1 Día", help=f"{MUESTRAS_POR_DIA} muestras", use_container_width=True):
            pasos_manual = MUESTRAS_POR_DIA
        if c4.button("+1 Semana", help=f"{MUESTRAS_POR_DIA * 7} muestras", use_container_width=True):
            pasos_manual = MUESTRAS_POR_DIA * 7
        if c5.button("Ir a Pandemia", help="Saltar al inicio del confinamiento", use_container_width=True):
            drift_mask = df_casa[COL_TIMESTAMP] >= DRIFT_INICIO
            if drift_mask.any():
                indice_drift = int(drift_mask.idxmax()) + MUESTRAS_POR_DIA * 7
                pasos_manual = max(0, indice_drift - st.session_state.sim_step)
            else:
                st.warning("Los datos no llegan hasta la fecha del confinamiento.")
        if c6.button("Procesar todo", type="primary", use_container_width=True):
            pasos_manual = max_rows - st.session_state.sim_step

    st.markdown("---")

    if pasos_manual > 0:
        with st.spinner("Procesando y aprendiendo..."):
            _avanzar(df_casa, pasos_manual, max_rows)

    @st.fragment(run_every=velocidad if auto_play else None)
    def panel_simulacion():
        if auto_play:
            if st.session_state.sim_step < max_rows:
                _avanzar(df_casa, muestras_tick, max_rows)
            if st.session_state.sim_step >= max_rows:
                st.session_state["online_fin_autoplay"] = True
                st.rerun()

        paso = st.session_state.sim_step
        st.progress(
            paso / max_rows,
            text=f"Progreso: {paso} de {max_rows} registros procesados",
        )

        if not st.session_state.historial_preds:
            st.info(
                "Inicia la simulación (tiempo real o avance manual) para visualizar datos."
            )
            return

        df_plot = pd.DataFrame(st.session_state.historial_preds)

        ventana_muestras = MUESTRAS_POR_DIA * DIAS_VENTANA_DINAMICA
        df_reciente = df_plot.tail(ventana_muestras)
        
        metricas_recientes = _metricas_seguras(df_reciente)
        metricas_globales = _metricas_seguras(df_plot)

        st.markdown(f"### Rendimiento — {modelo_nombre}")
        
        vista_metricas = st.selectbox(
            "Tipo de métricas",
            ["Dinámico (última semana)", "Acumulado (histórico total)"],
            key="online_vista_metricas",
        )
        
        if vista_metricas.startswith("Dinámico"):
            st.caption(
                f"Calculadas con las últimas {len(df_reciente)} muestras (7 días). "
                "Muestra la precisión actual del modelo."
            )
            _mostrar_metricas(metricas_recientes)
        else:
            st.caption(
                f"Calculadas sobre el total de {len(df_plot)} muestras procesadas desde el inicio."
            )
            _mostrar_metricas(metricas_globales)

        st.divider()

        _render_prediccion_real(df_plot, modelo_nombre, tema)

        if not auto_play:
            st.divider()
            st.markdown("### Exportar resultados")
            metricas_export = metricas_globales or {}
            csv_bytes = exportar_csv(df_plot, metricas_export)
            st.download_button(
                label="Descargar historial CSV",
                data=csv_bytes,
                file_name=f"online_sesion_{modelo_clave}_{cups_sel[:8]}.csv",
                mime="text/csv",
            )

    panel_simulacion()

# Gráficos
def _render_prediccion_real(
    df_plot: pd.DataFrame, modelo_nombre: str, tema: dict
) -> None:
    st.markdown("#### Predicción vs consumo real")

    if len(df_plot) < 2:
        st.warning("Se necesitan al menos 2 registros para visualizar el gráfico.")
        return

    zoom = st.selectbox(
        "Zoom temporal",
        list(OPCIONES_ZOOM.keys()),
        index=1,
        key="online_zoom_ventana",
    )
    dias_ventana = OPCIONES_ZOOM[zoom]

    max_time = df_plot["timestamp"].max()
    min_time_absoluto = df_plot["timestamp"].min()
    rango_x = None

    if dias_ventana is None:
        df_zoom = df_plot
    else:
        delta = pd.Timedelta(days=dias_ventana)
        
        if (max_time - min_time_absoluto) < delta:
            limite_izq = min_time_absoluto
            limite_der = min_time_absoluto + delta
        else:
            limite_izq = max_time - delta
            limite_der = max_time
            
        df_zoom = df_plot[(df_plot["timestamp"] >= limite_izq) & (df_plot["timestamp"] <= max_time)]
        rango_x = [limite_izq, limite_der]

    df_zoom_plot = _submuestrear(df_zoom)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_zoom_plot["timestamp"],
            y=df_zoom_plot["Consumo_Real"],
            name="Consumo real",
            line=dict(color=tema["real"], width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_zoom_plot["timestamp"],
            y=df_zoom_plot["Prediccion"],
            name=f"Predicción {modelo_nombre}",
            line=dict(color=tema["prediccion"], width=1.5, dash="dot"),
        )
    )
    
    fecha_min = df_zoom_plot["timestamp"].min() if not df_zoom_plot.empty else max_time
    
    if rango_x:
        rango_min, rango_max = rango_x[0], rango_x[1]
    else:
        rango_min, rango_max = fecha_min, max_time

    if rango_min <= DRIFT_FIN and rango_max >= DRIFT_INICIO:
        fig.add_vrect(
            x0=DRIFT_INICIO,
            x1=DRIFT_FIN,
            fillcolor=tema["prediccion"],
            opacity=0.10,
            layer="below",
            line_width=0,
        )
        
    _layout_comun(fig, tema, "kWh")

    if rango_x:
        fig.update_xaxes(range=rango_x)

    st.plotly_chart(fig, use_container_width=True, theme=None)