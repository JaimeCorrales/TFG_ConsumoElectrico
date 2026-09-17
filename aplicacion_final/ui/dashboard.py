import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import COL_CIUDAD, COL_CUPS, COL_KWH, COL_TEMP, COL_TIMESTAMP
from src.carga import filtrar_datos


def render_dashboard(df: pd.DataFrame, tema: dict) -> None:

    st.markdown("## Exploración del dataset")
    st.markdown(
        "Visualiza el consumo eléctrico histórico por ciudad y vivienda. "
        "Usa los filtros del panel lateral para acotar el análisis."
    )

    if df is None or df.empty:
        st.warning("No hay datos disponibles.")
        return

    # KPIs globales 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total registros", f"{len(df):,}")
    col2.metric("Viviendas", f"{df[COL_CUPS].nunique()}")
    col3.metric("Ciudades", f"{df[COL_CIUDAD].nunique()}")
    col4.metric(
        "Período",
        f"{df[COL_TIMESTAMP].min().year} – {df[COL_TIMESTAMP].max().year}",
    )

    st.divider()

    # Filtros 
    with st.sidebar:
        st.markdown("### Filtros de exploración")

        ciudades = ["Todas"] + sorted(
            df[COL_CIUDAD].dropna().unique().tolist()
        )
        ciudad_sel = st.selectbox("Ciudad", ciudades, key="dash_ciudad")

        cups_disp = (
            df[df[COL_CIUDAD] == ciudad_sel][COL_CUPS].unique().tolist()
            if ciudad_sel != "Todas"
            else df[COL_CUPS].unique().tolist()
        )
        cups_opciones = ["Todos"] + sorted(cups_disp)
        cups_sel = st.selectbox(
            "Vivienda (CUPS)", cups_opciones, key="dash_cups"
        )

        fecha_min = df[COL_TIMESTAMP].min().date()
        fecha_max = df[COL_TIMESTAMP].max().date()
        rango_fecha = st.date_input(
            "Rango de fechas",
            value=(fecha_min, fecha_max),
            min_value=fecha_min,
            max_value=fecha_max,
            key="dash_fechas",
        )

        granularidad = st.selectbox(
            "Granularidad del gráfico",
            ["Horaria", "Diaria", "Semanal", "Mensual"],
            index=1,
            key="dash_gran",
        )

    # Aplicar filtros
    fecha_i = rango_fecha[0] if len(rango_fecha) == 2 else fecha_min
    fecha_f = rango_fecha[1] if len(rango_fecha) == 2 else fecha_max

    df_f = filtrar_datos(df, ciudad_sel, cups_sel, fecha_i, fecha_f)
    if df_f.empty:
        return

    # Resamplear según granularidad
    freq_map = {"Horaria": "h", "Diaria": "D", "Semanal": "W", "Mensual": "ME"}
    freq = freq_map[granularidad]
    df_plot = (
        df_f.set_index(COL_TIMESTAMP)
        .groupby(COL_CUPS)[COL_KWH]
        .resample(freq)
        .sum()
        .reset_index()
    )

    # Gráfica de consumo histórico 
    st.markdown("### Consumo eléctrico histórico")

    if cups_sel == "Todos":
        # Media entre casas 
        df_agg = df_plot.groupby(COL_TIMESTAMP)[COL_KWH].mean().reset_index()
        fig = px.line(
            df_agg,
            x=COL_TIMESTAMP,
            y=COL_KWH,
            labels={COL_KWH: "kWh", COL_TIMESTAMP: "Fecha"},
            template=tema["plotly"],
            color_discrete_sequence=[tema["real"]],
        )
        fig.update_traces(line_width=1.5)
    else:
        fig = px.line(
            df_plot,
            x=COL_TIMESTAMP,
            y=COL_KWH,
            labels={COL_KWH: "kWh", COL_TIMESTAMP: "Fecha"},
            template=tema["plotly"],
            color_discrete_sequence=[tema["real"]],
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=tema["texto"],
        height=380,
        margin=dict(l=0, r=0, t=30, b=0),
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

    st.plotly_chart(fig, use_container_width=True)

    # Gráfica de temperatura
    st.markdown("### Temperatura")
    df_temp = (
        df_f.set_index(COL_TIMESTAMP)[COL_TEMP]
        .resample(freq)
        .mean()
        .reset_index()
    )
    fig_temp = px.area(
        df_temp,
        x=COL_TIMESTAMP,
        y=COL_TEMP,
        labels={COL_TEMP: "°C", COL_TIMESTAMP: "Fecha"},
        template=tema["plotly"],
        color_discrete_sequence=[tema["acento2"]],
    )
    fig_temp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=tema["texto"],
        height=260,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig_temp.update_xaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"]),
    )
    fig_temp.update_yaxes(
        color=tema["texto"],
        tickfont=dict(color=tema["texto"]),
        title_font=dict(color=tema["texto"]),
    )

    st.plotly_chart(fig_temp, use_container_width=True)

    # Estadísticas básicas
    st.markdown("### Estadísticas del período seleccionado")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consumo medio", f"{df_f[COL_KWH].mean():.3f} kWh")
    c2.metric("Consumo máximo", f"{df_f[COL_KWH].max():.3f} kWh")
    c3.metric("Consumo mínimo", f"{df_f[COL_KWH].min():.3f} kWh")
    c4.metric("Temp. media", f"{df_f[COL_TEMP].mean():.1f} °C")

    # Distribución por ciudad
    if ciudad_sel == "Todas":
        st.markdown("### Distribución del consumo por ciudad")
        df_ciudad = df_f.groupby(COL_CIUDAD)[COL_KWH].mean().reset_index()
        fig_bar = px.bar(
            df_ciudad,
            x=COL_CIUDAD,
            y=COL_KWH,
            labels={COL_KWH: "kWh medio", COL_CIUDAD: "Ciudad"},
            template=tema["plotly"],
            color=COL_CIUDAD,
            color_discrete_sequence=[
                tema["real"],
                tema["acento2"],
                tema["acento"],
                "#A78BFA",
            ],
        )
        fig_bar.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=tema["texto"],
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )

        fig_bar.update_xaxes(
            color=tema["texto"],
            tickfont=dict(color=tema["texto"]),
            title_font=dict(color=tema["texto"]),
        )
        fig_bar.update_yaxes(
            color=tema["texto"],
            tickfont=dict(color=tema["texto"]),
            title_font=dict(color=tema["texto"]),
        )

        st.plotly_chart(fig_bar, use_container_width=True)