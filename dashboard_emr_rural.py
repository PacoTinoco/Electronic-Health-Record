import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from deep_translator import GoogleTranslator
import time

# Configuración de la página
st.set_page_config(
    page_title="Dashboard EMR Rural",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar el estado del idioma
if 'language' not in st.session_state:
    st.session_state.language = 'es'

# Cache para traducciones (evita re-traducir)
if 'translation_cache' not in st.session_state:
    st.session_state.translation_cache = {}

# Función de traducción con cache
@st.cache_data
def translate_text(text, target_lang='en'):
    """Traduce texto usando cache para evitar re-traducciones"""
    if pd.isna(text) or text == '':
        return text
    
    # Crear clave única para el cache
    cache_key = f"{text}_{target_lang}"
    
    # Verificar si ya está traducido
    if cache_key in st.session_state.translation_cache:
        return st.session_state.translation_cache[cache_key]
    
    try:
        if target_lang == 'es':  # Si es español, no traducir
            return text
            
        translator = GoogleTranslator(source='es', target=target_lang)
        translated = translator.translate(str(text))
        
        # Guardar en cache
        st.session_state.translation_cache[cache_key] = translated
        return translated
    except:
        return text  # Si falla, devolver original

# Función helper simplificada
def t(text):
    """Traduce según el idioma seleccionado"""
    if st.session_state.language == 'es':
        return text
    return translate_text(text, st.session_state.language)

# SELECTOR DE IDIOMA EN LA PARTE SUPERIOR
col1, col2, col3 = st.columns([8, 1, 1])

with col2:
    if st.button("🇲🇽 ES", 
                 type="secondary" if st.session_state.language != 'es' else "primary",
                 use_container_width=True):
        st.session_state.language = 'es'
        st.rerun()

with col3:
    if st.button("🇺🇸 EN", 
                 type="secondary" if st.session_state.language != 'en' else "primary",
                 use_container_width=True):
        st.session_state.language = 'en'
        st.rerun()

# Título principal
st.title(t("🏥 Dashboard EMR Rural - Centros de Salud Michoacán"))
st.markdown("---")

# Cargar datos
@st.cache_data
def cargar_datos():
    df_pacientes = pd.read_csv('data/pacientes_rural.csv')
    df_historias = pd.read_csv('data/historias_clinicas.csv')
    df_prescripciones = pd.read_csv('data/prescripciones.csv')
    df_inventario = pd.read_csv('data/inventario_medicamentos.csv')
    df_personal = pd.read_csv('data/personal_medico.csv')
    df_conectividad = pd.read_csv('data/conectividad.csv')
    df_segmentacion = pd.read_csv('data/pacientes_segmentados.csv')
    
    # Convertir fechas
    df_historias['fecha_consulta'] = pd.to_datetime(df_historias['fecha_consulta'])
    df_prescripciones['fecha_prescripcion'] = pd.to_datetime(df_prescripciones['fecha_prescripcion'])
    
    return df_pacientes, df_historias, df_prescripciones, df_inventario, df_personal, df_conectividad, df_segmentacion

# Cargar todos los datasets
df_pacientes, df_historias, df_prescripciones, df_inventario, df_personal, df_conectividad, df_segmentacion = cargar_datos()

# Sidebar para filtros
st.sidebar.header(t("🔍 Filtros"))

# Filtro de centro de salud
centros = [t('Todos')] + list(df_pacientes['centro_salud_nombre'].unique())
centro_seleccionado = st.sidebar.selectbox(
    t("Centro de Salud"), 
    centros,
    format_func=lambda x: t(x) if x != t('Todos') else x
)

# Filtro de fecha
fecha_inicio = st.sidebar.date_input(
    t("Fecha inicio"),
    value=df_historias['fecha_consulta'].min(),
    min_value=df_historias['fecha_consulta'].min(),
    max_value=df_historias['fecha_consulta'].max()
)

fecha_fin = st.sidebar.date_input(
    t("Fecha fin"),
    value=df_historias['fecha_consulta'].max(),
    min_value=df_historias['fecha_consulta'].min(),
    max_value=df_historias['fecha_consulta'].max()
)

# Aplicar filtros
if centro_seleccionado not in ['Todos', 'All', t('Todos')]:
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
st.header(t("📊 Indicadores Principales"))
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_pacientes = len(df_pacientes_filtrado)
    pacientes_activos = df_pacientes_filtrado['activo'].sum()
    st.metric(
        t("Total Pacientes"),
        f"{total_pacientes:,}",
        f"{pacientes_activos} {t('activos')}"
    )

with col2:
    total_consultas = len(df_historias_filtrado)
    consultas_mes = len(df_historias_filtrado[df_historias_filtrado['fecha_consulta'] >= datetime.now() - timedelta(days=30)])
    st.metric(
        t("Consultas Totales"),
        f"{total_consultas:,}",
        f"{consultas_mes} {t('último mes')}"
    )

with col3:
    total_personal = len(df_personal_filtrado)
    medicos = len(df_personal_filtrado[df_personal_filtrado['especialidad'] == 'Medicina General'])
    st.metric(
        t("Personal Médico"),
        total_personal,
        f"{medicos} {t('médicos')}"
    )

with col4:
    # Medicamentos en estado crítico
    df_inventario_filtrado['meses_stock'] = df_inventario_filtrado['stock_actual'] / df_inventario_filtrado['stock_minimo']
    med_criticos = len(df_inventario_filtrado[df_inventario_filtrado['meses_stock'] < 1])
    st.metric(
        t("Medicamentos Críticos"),
        med_criticos,
        t("⚠️ Reabastecer") if med_criticos > 0 else t("✅ OK"),
        delta_color="inverse"
    )

with col5:
    # Pacientes de alto riesgo
    if 'grupo_riesgo' in df_segmentacion.columns:
        alto_riesgo = len(df_segmentacion[df_segmentacion['grupo_riesgo'] == 'Alto Riesgo'])
        st.metric(
            t("Pacientes Alto Riesgo"),
            alto_riesgo,
            t("Requieren seguimiento")
        )

st.markdown("---")

# SECCIÓN 2: ANÁLISIS EPIDEMIOLÓGICO
col1, col2 = st.columns(2)

with col1:
    st.subheader(t("🦠 Top Diagnósticos"))
    
    # Copiar datos para traducción
    df_historias_temp = df_historias_filtrado.copy()
    
    # Traducir diagnósticos si es necesario
    if st.session_state.language != 'es':
        df_historias_temp['diagnostico_nombre'] = df_historias_temp['diagnostico_nombre'].apply(
            lambda x: translate_text(x, st.session_state.language)
        )
    
    top_diagnosticos = df_historias_temp['diagnostico_nombre'].value_counts().head(10)
    
    fig_diag = px.bar(
        x=top_diagnosticos.values,
        y=top_diagnosticos.index,
        orientation='h',
        labels={'x': t('Número de Casos'), 'y': t('Diagnóstico')},
        color=top_diagnosticos.values,
        color_continuous_scale='Reds'
    )
    fig_diag.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_diag, use_container_width=True)

with col2:
    st.subheader(t("📈 Tendencia de Consultas"))
    df_historias_filtrado['mes'] = df_historias_filtrado['fecha_consulta'].dt.to_period('M').astype(str)
    consultas_mes = df_historias_filtrado.groupby('mes').size().reset_index(name='consultas')
    fig_tendencia = px.line(
        consultas_mes,
        x='mes',
        y='consultas',
        markers=True,
        labels={'mes': t('Mes'), 'consultas': t('Número de Consultas')}
    )
    fig_tendencia.update_layout(height=400)
    st.plotly_chart(fig_tendencia, use_container_width=True)

# SECCIÓN 3: GESTIÓN DE RECURSOS
st.header(t("💊 Gestión de Recursos"))
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(t("📦 Estado del Inventario"))
    # Análisis de stock
    df_stock = df_inventario_filtrado.groupby('medicamento_nombre').agg({
        'stock_actual': 'sum',
        'stock_minimo': 'sum'
    }).reset_index()
    df_stock['porcentaje'] = (df_stock['stock_actual'] / df_stock['stock_minimo'] * 100).round(0)
    
    # Traducir estados
    def get_estado(x):
        if x < 50:
            return t('🔴 Crítico')
        elif x < 100:
            return t('🟡 Alerta')
        else:
            return t('🟢 OK')
    
    df_stock['estado'] = df_stock['porcentaje'].apply(get_estado)
    
    # Traducir nombres de medicamentos si es necesario
    if st.session_state.language != 'es':
        df_stock['medicamento_nombre'] = df_stock['medicamento_nombre'].apply(
            lambda x: translate_text(x, st.session_state.language)
        )
    
    # Mostrar tabla
    st.dataframe(
        df_stock[['medicamento_nombre', 'stock_actual', 'estado']].sort_values('stock_actual'),
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.subheader(t("👥 Carga de Trabajo"))
    if centro_seleccionado not in ['Todos', 'All', t('Todos')]:
        carga = len(df_historias_filtrado) / max(len(df_personal_filtrado), 1)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=carga,
            title={'text': t("Consultas por Empleado")},
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

# SECCIÓN 5: ALERTAS
st.header(t("⚠️ Alertas y Recomendaciones"))
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(t("🚨 Alertas Críticas"))
    alertas = []
    
    # Medicamentos críticos
    med_criticos_lista = df_inventario_filtrado[df_inventario_filtrado['meses_stock'] < 1]['medicamento_nombre'].tolist()
    if med_criticos_lista:
        for med in med_criticos_lista[:3]:
            alertas.append(f"❗ {t(med)} - {t('Stock crítico')}")
    
    if alertas:
        for alerta in alertas:
            st.warning(alerta)
    else:
        st.success(t("✅ Sin alertas críticas"))

# Footer
st.markdown("---")
st.markdown(f"🏥 **{t('Sistema EMR Rural')}** | {t('Actualizado')}: " + datetime.now().strftime("%Y-%m-%d %H:%M"))

# Nota sobre la traducción
if st.session_state.language != 'es':
    st.sidebar.info(
        """
        🌐 **Auto-translated Dashboard**
        Using Google Translate API
        Some medical terms may require review
        """
    )
else:
    st.sidebar.info(
        """
        **Para ejecutar el dashboard:**
        1. Instalar: `pip install streamlit plotly deep-translator`
        2. Ejecutar: `streamlit run dashboard_emr_rural.py`
        3. Se abrirá en el navegador
        """
    )

