import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Predictive Health Analytics",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Predictive Health Analytics - Rural EMR")
st.markdown("---")

# Cargar datos
@st.cache_data
def load_data():
    df_patients = pd.read_csv('data/pacientes_rural.csv')
    df_history = pd.read_csv('data/historias_clinicas.csv')
    df_prescriptions = pd.read_csv('data/prescripciones.csv')
    df_inventory = pd.read_csv('data/inventario_medicamentos.csv')
    
    df_history['fecha_consulta'] = pd.to_datetime(df_history['fecha_consulta'])
    df_prescriptions['fecha_prescripcion'] = pd.to_datetime(df_prescriptions['fecha_prescripcion'])
    
    return df_patients, df_history, df_prescriptions, df_inventory

df_patients, df_history, df_prescriptions, df_inventory = load_data()

# SECCIÓN 1: PREDICCIÓN DE DEMANDA
st.header("📈 Demand Forecasting")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Consultation Demand Prediction")
    
    # Serie temporal de consultas
    daily_consultations = df_history.groupby(df_history['fecha_consulta'].dt.date).size().reset_index(name='consultations')
    daily_consultations['fecha_consulta'] = pd.to_datetime(daily_consultations['fecha_consulta'])
    
    # Features temporales
    daily_consultations['day_of_week'] = daily_consultations['fecha_consulta'].dt.dayofweek
    daily_consultations['month'] = daily_consultations['fecha_consulta'].dt.month
    daily_consultations['day'] = daily_consultations['fecha_consulta'].dt.day
    
    # Entrenar modelo
    features = ['day_of_week', 'month', 'day']
    X = daily_consultations[features]
    y = daily_consultations['consultations']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Predecir próximos 30 días
    future_dates = pd.date_range(start=daily_consultations['fecha_consulta'].max() + timedelta(days=1), periods=30)
    future_df = pd.DataFrame({
        'fecha_consulta': future_dates,
        'day_of_week': future_dates.dayofweek,
        'month': future_dates.month,
        'day': future_dates.day
    })
    
    future_df['predicted_consultations'] = model.predict(future_df[features])
    
    # Visualización
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_consultations['fecha_consulta'],
        y=daily_consultations['consultations'],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=future_df['fecha_consulta'],
        y=future_df['predicted_consultations'],
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title="30-Day Consultation Forecast",
        xaxis_title="Date",
        yaxis_title="Number of Consultations",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Medication Stock Prediction")
    
    # Análisis de consumo de medicamentos
    fecha_limite = df_prescriptions['fecha_prescripcion'].max() - timedelta(days=30)
    prescripciones_recientes = df_prescriptions[df_prescriptions['fecha_prescripcion'] >= fecha_limite]
    
    # Consumo diario
    consumo_diario = prescripciones_recientes.groupby('medicamento_nombre').size().reset_index(name='prescripciones_mes')
    consumo_diario['consumo_diario_promedio'] = consumo_diario['prescripciones_mes'] / 30
    
    # Estado del inventario
    inventory_status = df_inventory.groupby('medicamento_nombre').agg({
        'stock_actual': 'sum',
        'stock_minimo': 'sum'
    }).reset_index()
    
    # Merge y análisis
    med_analysis = pd.merge(inventory_status, consumo_diario, on='medicamento_nombre', how='left')
    med_analysis['consumo_diario_promedio'] = med_analysis['consumo_diario_promedio'].fillna(
        med_analysis['stock_minimo'] / 30
    ).clip(lower=0.1)
    
    med_analysis['dias_hasta_agotamiento'] = med_analysis['stock_actual'] / med_analysis['consumo_diario_promedio']
    med_analysis['nivel_riesgo'] = pd.cut(
        med_analysis['dias_hasta_agotamiento'], 
        bins=[-np.inf, 7, 15, 30, 90, np.inf], 
        labels=['Crítico', 'Muy Alto', 'Alto', 'Medio', 'Bajo']
    )
    
    # Visualización top 10 en riesgo
    at_risk = med_analysis[med_analysis['dias_hasta_agotamiento'] <= 30].nsmallest(10, 'dias_hasta_agotamiento')
    
    if len(at_risk) == 0:
        at_risk = med_analysis.nsmallest(10, 'dias_hasta_agotamiento')
    
    colors = {'Crítico': '#d32f2f', 'Muy Alto': '#f57c00', 'Alto': '#fbc02d', 
              'Medio': '#388e3c', 'Bajo': '#1976d2'}
    
    fig2 = go.Figure()
    
    for idx, row in at_risk.iterrows():
        fig2.add_trace(go.Bar(
            x=[row['dias_hasta_agotamiento']],
            y=[row['medicamento_nombre']],
            orientation='h',
            marker_color=colors.get(row['nivel_riesgo'], '#gray'),
            showlegend=False,
            text=f"{row['dias_hasta_agotamiento']:.0f} días",
            textposition='outside'
        ))
    
    fig2.add_vline(x=7, line_dash="dash", line_color="red", annotation_text="Crítico")
    fig2.add_vline(x=15, line_dash="dash", line_color="orange", annotation_text="Alto")
    fig2.add_vline(x=30, line_dash="dash", line_color="yellow", annotation_text="Medio")
    
    fig2.update_layout(
        title="Predicción de Agotamiento de Medicamentos",
        xaxis_title="Días hasta agotamiento",
        yaxis_title="Medicamento",
        height=400,
        xaxis=dict(range=[0, max(35, at_risk['dias_hasta_agotamiento'].max() * 1.1) if len(at_risk) > 0 else 60])
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Métricas
    col2_1, col2_2, col2_3 = st.columns(3)
    with col2_1:
        st.metric("🚨 Críticos", len(med_analysis[med_analysis['nivel_riesgo'] == 'Crítico']))
    with col2_2:
        st.metric("⚠️ Alto Riesgo", len(med_analysis[med_analysis['nivel_riesgo'].isin(['Alto', 'Muy Alto'])]))
    with col2_3:
        st.metric("📊 Consumo Diario", f"{med_analysis['consumo_diario_promedio'].sum():.0f}")

# SECCIÓN 2: PREDICCIÓN DE RIESGO DE PACIENTES
st.header("🎯 Patient Risk Prediction")

# Preparar datos de riesgo
patient_history = df_history.groupby('paciente_id').agg({
    'diagnostico_cie10': 'count',
    'fecha_consulta': ['min', 'max']
}).reset_index()
patient_history.columns = ['paciente_id', 'total_visits', 'first_visit', 'last_visit']

patient_risk = pd.merge(df_patients, patient_history, on='paciente_id', how='left')
patient_risk['days_since_last_visit'] = (datetime.now() - pd.to_datetime(patient_risk['last_visit'])).dt.days
patient_risk['days_since_last_visit'] = patient_risk['days_since_last_visit'].fillna(999)

# Crear target de riesgo
patient_risk['high_risk'] = (
    (patient_risk['es_cronico'] == 1) & 
    (patient_risk['distancia_centro_km'] > 20) & 
    (patient_risk['days_since_last_visit'] > 180)
).astype(int)

# Entrenar modelo
risk_features = ['edad', 'distancia_centro_km', 'es_cronico', 'tiene_seguro_popular', 
                 'total_visits', 'days_since_last_visit']

X_risk = patient_risk[risk_features].fillna(0)
y_risk = patient_risk['high_risk']

X_train, X_test, y_train, y_test = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
risk_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
risk_model.fit(X_train, y_train)

# Visualizaciones
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Risk Factors Importance")
    feature_importance = pd.DataFrame({
        'feature': risk_features,
        'importance': risk_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig3 = px.bar(feature_importance, x='importance', y='feature', orientation='h')
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("Risk Distribution")
    risk_counts = patient_risk['high_risk'].value_counts()
    fig4 = px.pie(values=risk_counts.values, names=['Low Risk', 'High Risk'],
                  color_discrete_map={'Low Risk': 'blue', 'High Risk': 'red'})
    st.plotly_chart(fig4, use_container_width=True)

with col3:
    st.subheader("Model Performance")
    y_pred = risk_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    st.metric("High Risk Patients", f"{patient_risk['high_risk'].sum()}")
    st.metric("Total Patients", f"{len(patient_risk):,}")

# SECCIÓN 3: DETECCIÓN DE EPIDEMIAS
st.header("🦠 Epidemic Detection & Alerts")

# Tendencias por diagnóstico
diagnosis_trends = df_history.groupby([
    df_history['fecha_consulta'].dt.to_period('W'),
    'diagnostico_nombre'
]).size().reset_index(name='cases')

top_diagnoses = df_history['diagnostico_nombre'].value_counts().head(5).index
tabs = st.tabs(top_diagnoses.tolist())

for i, diagnosis in enumerate(top_diagnoses):
    with tabs[i]:
        diagnosis_data = diagnosis_trends[diagnosis_trends['diagnostico_nombre'] == diagnosis].copy()
        diagnosis_data['fecha_consulta'] = diagnosis_data['fecha_consulta'].astype(str)
        
        # Detección de anomalías
        diagnosis_data['moving_avg'] = diagnosis_data['cases'].rolling(window=4, min_periods=1).mean()
        diagnosis_data['std'] = diagnosis_data['cases'].rolling(window=4, min_periods=1).std().fillna(1)
        diagnosis_data['upper_limit'] = diagnosis_data['moving_avg'] + 2 * diagnosis_data['std']
        diagnosis_data['is_outbreak'] = diagnosis_data['cases'] > diagnosis_data['upper_limit']
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=diagnosis_data['fecha_consulta'], y=diagnosis_data['cases'],
                                 mode='lines+markers', name='Cases', line=dict(color='blue')))
        fig5.add_trace(go.Scatter(x=diagnosis_data['fecha_consulta'], y=diagnosis_data['moving_avg'],
                                 mode='lines', name='Moving Average', line=dict(color='green', dash='dash')))
        fig5.add_trace(go.Scatter(x=diagnosis_data['fecha_consulta'], y=diagnosis_data['upper_limit'],
                                 mode='lines', name='Alert Threshold', line=dict(color='red', dash='dot')))
        
        # Marcar brotes
        outbreak_data = diagnosis_data[diagnosis_data['is_outbreak']]
        if not outbreak_data.empty:
            fig5.add_trace(go.Scatter(x=outbreak_data['fecha_consulta'], y=outbreak_data['cases'],
                                     mode='markers', name='Outbreak', 
                                     marker=dict(color='red', size=12, symbol='x')))
            st.error(f"⚠️ Potential outbreak detected in weeks: {', '.join(outbreak_data['fecha_consulta'])}")
        
        fig5.update_layout(title=f"Epidemic Monitoring: {diagnosis}",
                          xaxis_title="Week", yaxis_title="Cases", height=400)
        st.plotly_chart(fig5, use_container_width=True)

# SECCIÓN 4: RECOMENDACIONES AUTOMÁTICAS
st.header("💡 Automated Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resource Optimization")
    
    # Generar recomendaciones basadas en análisis
    recommendations = []
    
    # Personal
    workload = len(df_history) / max(df_patients['centro_salud_id'].nunique(), 1)
    if workload > 5000:
        recommendations.append({
            'priority': 'High',
            'action': 'Hire additional medical staff',
            'reason': f'Average workload is {workload:.0f} consultations per center'
        })
    
    # Inventario
    critical_meds = med_analysis[med_analysis['dias_hasta_agotamiento'] < 7]
    if len(critical_meds) > 0:
        recommendations.append({
            'priority': 'Critical',
            'action': f'Urgent restock needed for {len(critical_meds)} medications',
            'reason': 'Less than 7 days of stock remaining'
        })
    
    # Accesibilidad
    far_patients = patient_risk[patient_risk['distancia_centro_km'] > 30]
    if len(far_patients) > 100:
        recommendations.append({
            'priority': 'Medium',
            'action': 'Implement mobile health brigades',
            'reason': f'{len(far_patients)} patients live >30km from health center'
        })
    
    # Pacientes en riesgo
    high_risk_count = patient_risk['high_risk'].sum()
    if high_risk_count > 50:
        recommendations.append({
            'priority': 'High',
            'action': 'Launch targeted intervention program',
            'reason': f'{high_risk_count} high-risk patients identified'
        })
    
    # Mostrar recomendaciones
    if not recommendations:
        st.success("✅ No critical issues detected")
    else:
        for rec in recommendations:
            if rec['priority'] == 'Critical':
                st.error(f"🚨 **{rec['priority']}**: {rec['action']}")
            elif rec['priority'] == 'High':
                st.warning(f"⚠️ **{rec['priority']}**: {rec['action']}")
            else:
                st.info(f"ℹ️ **{rec['priority']}**: {rec['action']}")
            st.caption(f"Reason: {rec['reason']}")

with col2:
    st.subheader("Intervention Priorities")
    
    # Matriz de priorización
    priority_matrix = pd.DataFrame({
        'Intervention': ['Mobile Brigades', 'Telemedicine', 'Stock Management', 
                        'Staff Training', 'Prevention Programs'],
        'Impact': [85, 70, 90, 60, 75],
        'Feasibility': [60, 80, 85, 90, 70],
        'Cost': [30, 20, 15, 10, 25]
    })
    
    fig6 = px.scatter(priority_matrix, x='Feasibility', y='Impact', size='Cost',
                      text='Intervention', title="Intervention Priority Matrix",
                      size_max=60)
    fig6.update_traces(textposition='top center')
    fig6.add_vline(x=70, line_dash="dash", line_color="gray", opacity=0.5)
    fig6.add_hline(y=70, line_dash="dash", line_color="gray", opacity=0.5)
    fig6.update_layout(height=400)
    st.plotly_chart(fig6, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"🔮 **Predictive Analytics Module** | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")