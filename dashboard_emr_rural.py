import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Dashboard EMR Rural",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏥 Dashboard EMR Rural - Centros de Salud Michoacán")
st.markdown("---")

# Cargar datos
@st.cache_data
def cargar_datos():
    df_pacientes = pd.read_csv('pacientes_rural.csv')
    df_historias = pd.read_csv('historias_clinicas.csv')
    df_prescripciones = pd.read_csv('prescripciones.csv')
    df_inventario = pd.read_csv('inventario_medicamentos.csv')
    df_personal = pd.read_csv('personal_medico.csv')
    df_conectividad = pd.read_csv('conectividad.csv')
    df_segmentacion = pd.read_csv('pacientes_segmentados.csv')
    
    # Convertir fechas
    df_historias['fecha_consulta'] = pd.to_datetime(df_historias['fecha_consulta'])
    df_prescripciones['fecha_prescripcion'] = pd.to_datetime(df_prescripciones['fecha_prescripcion'])
    
    return df_pacientes, df_historias, df_prescripciones, df_inventario, df_personal, df_conectividad, df_segmentacion

# Cargar todos los datasets
df_pacientes, df_historias, df_prescripciones, df_inventario, df_personal, df_conectividad, df_segmentacion = cargar_datos()

# Sidebar para filtros
st.sidebar.header("🔍 Filtros")

# Filtro de centro de salud
centros = ['Todos'] + list(df_pacientes['centro_salud_nombre'].unique())
centro_seleccionado = st.sidebar.selectbox("Centro de Salud", centros)

# Filtro de fecha
fecha_inicio = st.sidebar.date_input(
    "Fecha inicio",
    value=df_historias['fecha_consulta'].min(),
    min_value=df_historias['fecha_consulta'].min(),
    max_value=df_historias['fecha_consulta'].max()
)

fecha_fin = st.sidebar.date_input(
    "Fecha fin",
    value=df_historias['fecha_consulta'].max(),
    min_value=df_historias['fecha_consulta'].min(),
    max_value=df_historias['fecha_consulta'].max()
)

# Aplicar filtros
if centro_seleccionado != 'Todos':
    df_pacientes_filtrado = df_pacientes[df_pacientes['centro_salud_nombre'] == centro_seleccionado]
    centro_id = df_pacientes_filtrado['centro_salud_id'].iloc[0]
    df_historias_filtrado = df_historias[df_historias['centro_salud_id'] == centro_id]
    df_inventario_filtrado = df_inventario[df_inventario['centro_salud_id'] == centro_id]
    df_personal_filtrado = df_personal[df_personal['centro_salud_id'] == centro_id]
else:
    df_pacientes_filtrado = df_pacientes
    df_historias_filtrado = df_historias
    df_inventario_filtrado = df_inventario
    df_personal_filtrado = df_personal

# Filtrar por fecha
df_historias_filtrado = df_historias_filtrado[
    (df_historias_filtrado['fecha_consulta'] >= pd.to_datetime(fecha_inicio)) &
    (df_historias_filtrado['fecha_consulta'] <= pd.to_datetime(fecha_fin))
]

# SECCIÓN 1: MÉTRICAS PRINCIPALES
st.header("📊 Indicadores Principales")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_pacientes = len(df_pacientes_filtrado)
    pacientes_activos = df_pacientes_filtrado['activo'].sum()
    st.metric(
        "Total Pacientes",
        f"{total_pacientes:,}",
        f"{pacientes_activos} activos"
    )

with col2:
    total_consultas = len(df_historias_filtrado)
    consultas_mes = len(df_historias_filtrado[df_historias_filtrado['fecha_consulta'] >= datetime.now() - timedelta(days=30)])
    st.metric(
        "Consultas Totales",
        f"{total_consultas:,}",
        f"{consultas_mes} último mes"
    )

with col3:
    total_personal = len(df_personal_filtrado)
    medicos = len(df_personal_filtrado[df_personal_filtrado['especialidad'] == 'Medicina General'])
    st.metric(
        "Personal Médico",
        total_personal,
        f"{medicos} médicos"
    )

with col4:
    # Medicamentos en estado crítico
    df_inventario_filtrado['meses_stock'] = df_inventario_filtrado['stock_actual'] / df_inventario_filtrado['stock_minimo']
    med_criticos = len(df_inventario_filtrado[df_inventario_filtrado['meses_stock'] < 1])
    st.metric(
        "Medicamentos Críticos",
        med_criticos,
        "⚠️ Reabastecer" if med_criticos > 0 else "✅ OK",
        delta_color="inverse"
    )

with col5:
    # Pacientes de alto riesgo
    if 'grupo_riesgo' in df_segmentacion.columns:
        alto_riesgo = len(df_segmentacion[df_segmentacion['grupo_riesgo'] == 'Alto Riesgo'])
        st.metric(
            "Pacientes Alto Riesgo",
            alto_riesgo,
            "Requieren seguimiento"
        )

st.markdown("---")

# SECCIÓN 2: ANÁLISIS EPIDEMIOLÓGICO
col1, col2 = st.columns(2)

with col1:
    st.subheader("🦠 Top Diagnósticos")
    top_diagnosticos = df_historias_filtrado['diagnostico_nombre'].value_counts().head(10)
    fig_diag = px.bar(
        x=top_diagnosticos.values,
        y=top_diagnosticos.index,
        orientation='h',
        labels={'x': 'Número de Casos', 'y': 'Diagnóstico'},
        color=top_diagnosticos.values,
        color_continuous_scale='Reds'
    )
    fig_diag.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_diag, use_container_width=True)

with col2:
    st.subheader("📈 Tendencia de Consultas")
    df_historias_filtrado['mes'] = df_historias_filtrado['fecha_consulta'].dt.to_period('M').astype(str)
    consultas_mes = df_historias_filtrado.groupby('mes').size().reset_index(name='consultas')
    fig_tendencia = px.line(
        consultas_mes,
        x='mes',
        y='consultas',
        markers=True,
        labels={'mes': 'Mes', 'consultas': 'Número de Consultas'}
    )
    fig_tendencia.update_layout(height=400)
    st.plotly_chart(fig_tendencia, use_container_width=True)

# SECCIÓN 3: GESTIÓN DE RECURSOS
st.header("💊 Gestión de Recursos")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📦 Estado del Inventario")
    # Análisis de stock
    df_stock = df_inventario_filtrado.groupby('medicamento_nombre').agg({
        'stock_actual': 'sum',
        'stock_minimo': 'sum'
    }).reset_index()
    df_stock['porcentaje'] = (df_stock['stock_actual'] / df_stock['stock_minimo'] * 100).round(0)
    df_stock['estado'] = df_stock['porcentaje'].apply(
        lambda x: '🔴 Crítico' if x < 50 else '🟡 Alerta' if x < 100 else '🟢 OK'
    )
    
    # Mostrar tabla con colores
    st.dataframe(
        df_stock[['medicamento_nombre', 'stock_actual', 'estado']].sort_values('stock_actual'),
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.subheader("👥 Carga de Trabajo")
    if centro_seleccionado != 'Todos':
        carga = len(df_historias_filtrado) / max(len(df_personal_filtrado), 1)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=carga,
            title={'text': "Consultas por Empleado"},
            gauge={
                'axis': {'range': [None, 500]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 200], 'color': "lightgreen"},
                    {'range': [200, 350], 'color': "yellow"},
                    {'range': [350, 500], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 400
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        # Mostrar por centro
        carga_centros = df_historias.groupby('centro_salud_id').size() / df_personal.groupby('centro_salud_id').size()
        fig_carga = px.bar(
            x=carga_centros.index,
            y=carga_centros.values,
            labels={'x': 'Centro', 'y': 'Consultas/Empleado'},
            color=carga_centros.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig_carga.update_layout(height=300)
        st.plotly_chart(fig_carga, use_container_width=True)

with col3:
    st.subheader("📡 Conectividad")
    if centro_seleccionado != 'Todos':
        conectividad_centro = df_conectividad[df_conectividad['centro_salud_id'] == centro_id].tail(7)
        fig_conectividad = px.line(
            conectividad_centro,
            x='fecha',
            y='horas_conectado',
            markers=True,
            labels={'fecha': 'Fecha', 'horas_conectado': 'Horas Conectado'}
        )
        fig_conectividad.add_hline(y=12, line_dash="dash", line_color="red", 
                                  annotation_text="Mínimo recomendado")
        fig_conectividad.update_layout(height=300)
        st.plotly_chart(fig_conectividad, use_container_width=True)
    else:
        promedio_conectividad = df_conectividad.groupby('centro_salud_id')['horas_conectado'].mean()
        st.bar_chart(promedio_conectividad)

# SECCIÓN 4: ANÁLISIS GEOGRÁFICO Y SEGMENTACIÓN
st.header("🗺️ Análisis Geográfico y Segmentación")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Distribución de Pacientes por Distancia")
    fig_dist = px.histogram(
        df_segmentacion,
        x='distancia_centro_km',
        nbins=30,
        labels={'distancia_centro_km': 'Distancia al Centro (km)', 'count': 'Número de Pacientes'},
        color_discrete_sequence=['#1f77b4']
    )
    fig_dist.add_vline(x=20, line_dash="dash", line_color="red", 
                       annotation_text="Accesibilidad Baja")
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.subheader("🎯 Segmentación de Pacientes")
    if 'grupo_riesgo' in df_segmentacion.columns and 'accesibilidad' in df_segmentacion.columns:
        # Crear matriz de segmentación
        matriz_seg = pd.crosstab(df_segmentacion['grupo_riesgo'], df_segmentacion['accesibilidad'])
        fig_seg = px.imshow(
            matriz_seg,
            labels=dict(x="Accesibilidad", y="Grupo de Riesgo", color="Pacientes"),
            color_continuous_scale='Reds',
            text_auto=True
        )
        fig_seg.update_layout(height=400)
        st.plotly_chart(fig_seg, use_container_width=True)

# SECCIÓN 5: ALERTAS Y RECOMENDACIONES
st.header("⚠️ Alertas y Recomendaciones")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚨 Alertas Críticas")
    alertas = []
    
    # Medicamentos críticos
    med_criticos_lista = df_inventario_filtrado[df_inventario_filtrado['meses_stock'] < 1]['medicamento_nombre'].tolist()
    if med_criticos_lista:
        for med in med_criticos_lista[:3]:
            alertas.append(f"❗ {med} - Stock crítico")
    
    # Pacientes de alto riesgo sin seguimiento
    if 'grupo_riesgo' in df_segmentacion.columns:
        alto_riesgo_count = len(df_segmentacion[
            (df_segmentacion['grupo_riesgo'] == 'Alto Riesgo') & 
            (df_segmentacion['accesibilidad'] == 'Baja')
        ])
        if alto_riesgo_count > 0:
            alertas.append(f"⚠️ {alto_riesgo_count} pacientes alto riesgo con baja accesibilidad")
    
    if alertas:
        for alerta in alertas:
            st.warning(alerta)
    else:
        st.success("✅ Sin alertas críticas")

with col2:
    st.subheader("📋 Acciones Recomendadas")
    if med_criticos > 0:
        st.info("🏪 Realizar pedido urgente de medicamentos")
    
    if centro_seleccionado != 'Todos' and len(df_personal_filtrado) > 0:
        if len(df_historias_filtrado) / len(df_personal_filtrado) > 300:
            st.info("👥 Considerar contratar más personal")
    
    st.info("🚐 Programar brigada móvil para zonas alejadas")

with col3:
    st.subheader("📊 Resumen Ejecutivo")
    # Calcular métricas resumen
    tasa_cronicos = (df_segmentacion['es_cronico'].sum() / len(df_segmentacion) * 100).round(1)
    cobertura_seguro = (df_pacientes['tiene_seguro_popular'].sum() / len(df_pacientes) * 100).round(1)
    
    st.metric("Pacientes Crónicos", f"{tasa_cronicos}%")
    st.metric("Cobertura Seguro Popular", f"{cobertura_seguro}%")
    st.metric("Distancia Promedio", f"{df_pacientes['distancia_centro_km'].mean():.1f} km")

# Footer
st.markdown("---")
st.markdown("🏥 **Sistema EMR Rural** | Actualizado: " + datetime.now().strftime("%Y-%m-%d %H:%M"))

# Instrucciones para ejecutar
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Para ejecutar el dashboard:**
    1. Instalar: `pip install streamlit plotly`
    2. Ejecutar: `streamlit run dashboard_emr_rural.py`
    3. Se abrirá en el navegador
    """
)



