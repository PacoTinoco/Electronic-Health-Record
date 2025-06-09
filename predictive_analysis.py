import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
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
    
    # Convertir fechas
    df_history['fecha_consulta'] = pd.to_datetime(df_history['fecha_consulta'])
    df_prescriptions['fecha_prescripcion'] = pd.to_datetime(df_prescriptions['fecha_prescripcion'])
    
    return df_patients, df_history, df_prescriptions, df_inventory

df_patients, df_history, df_prescriptions, df_inventory = load_data()

# SECCIÓN 1: PREDICCIÓN DE DEMANDA
st.header("📈 Demand Forecasting")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Consultation Demand Prediction")
    
    # Preparar datos para serie temporal
    daily_consultations = df_history.groupby(df_history['fecha_consulta'].dt.date).size().reset_index(name='consultations')
    daily_consultations['fecha_consulta'] = pd.to_datetime(daily_consultations['fecha_consulta'])
    
    # Crear features temporales
    daily_consultations['day_of_week'] = daily_consultations['fecha_consulta'].dt.dayofweek
    daily_consultations['month'] = daily_consultations['fecha_consulta'].dt.month
    daily_consultations['day'] = daily_consultations['fecha_consulta'].dt.day
    
    # Modelo simple de predicción
    features = ['day_of_week', 'month', 'day']
    X = daily_consultations[features]
    y = daily_consultations['consultations']
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predecir próximos 30 días
    future_dates = pd.date_range(start=daily_consultations['fecha_consulta'].max() + timedelta(days=1), periods=30)
    future_df = pd.DataFrame({
        'fecha_consulta': future_dates,
        'day_of_week': future_dates.dayofweek,
        'month': future_dates.month,
        'day': future_dates.day
    })
    
    predictions = model.predict(future_df[features])
    future_df['predicted_consultations'] = predictions
    
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
        yaxis_title="Number of Consultations"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Medication Stock Prediction")
    
    # Análisis de consumo de medicamentos
    med_consumption = df_prescriptions.groupby('medicamento_nombre').size().reset_index(name='total_prescribed')
    inventory_status = df_inventory.groupby('medicamento_nombre').agg({
        'stock_actual': 'sum',
        'stock_minimo': 'sum'
    }).reset_index()
    
    # Merge datos
    med_analysis = pd.merge(inventory_status, med_consumption, on='medicamento_nombre', how='left')
    med_analysis['days_until_stockout'] = (med_analysis['stock_actual'] / (med_analysis['total_prescribed'] / 365)).fillna(999)
    med_analysis['risk_level'] = pd.cut(med_analysis['days_until_stockout'], 
                                       bins=[0, 7, 30, 90, 999], 
                                       labels=['Critical', 'High', 'Medium', 'Low'])
    
    # Top 10 medicamentos en riesgo
    at_risk = med_analysis[med_analysis['days_until_stockout'] < 30].sort_values('days_until_stockout')
    
    fig2 = px.bar(
        at_risk.head(10),
        x='days_until_stockout',
        y='medicamento_nombre',
        orientation='h',
        color='risk_level',
        color_discrete_map={'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'},
        title="Medications at Risk of Stockout"
    )
    st.plotly_chart(fig2, use_container_width=True)

# SECCIÓN 2: PREDICCIÓN DE RIESGO DE PACIENTES
st.header("🎯 Patient Risk Prediction")

# Preparar features para predicción
patient_features = df_patients.copy()
patient_history = df_history.groupby('paciente_id').agg({
    'diagnostico_cie10': 'count',
    'fecha_consulta': ['min', 'max']
}).reset_index()
patient_history.columns = ['paciente_id', 'total_visits', 'first_visit', 'last_visit']

# Merge con datos de pacientes
patient_risk = pd.merge(patient_features, patient_history, on='paciente_id', how='left')
patient_risk['days_since_last_visit'] = (datetime.now() - pd.to_datetime(patient_risk['last_visit'])).dt.days

# Features para el modelo
risk_features = ['edad', 'distancia_centro_km', 'es_cronico', 'tiene_seguro_popular', 
                 'total_visits', 'days_since_last_visit']

# Crear target sintético (para demostración)
patient_risk['high_risk'] = (
    (patient_risk['es_cronico'] == 1) & 
    (patient_risk['distancia_centro_km'] > 20) & 
    (patient_risk['days_since_last_visit'] > 180)
).astype(int)

# Preparar datos para modelo
X_risk = patient_risk[risk_features].fillna(0)
y_risk = patient_risk['high_risk']

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': risk_features,
    'importance': risk_model.feature_importances_
}).sort_values('importance', ascending=False)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Risk Factors Importance")
    fig3 = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title="Patient Risk Factors"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("Risk Distribution")
    risk_dist = patient_risk['high_risk'].value_counts()
    fig4 = px.pie(
        values=risk_dist.values,
        names=['Low Risk', 'High Risk'],
        title="Patient Risk Distribution",
        color_discrete_map={'Low Risk': 'green', 'High Risk': 'red'}
    )
    st.plotly_chart(fig4, use_container_width=True)

with col3:
    st.subheader("Model Performance")
    y_pred = risk_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    st.metric("High Risk Patients Identified", f"{y_pred.sum()}")
    st.metric("Total Patients Analyzed", f"{len(patient_risk):,}")

# SECCIÓN 3: ANÁLISIS DE EPIDEMIAS
st.header("🦠 Epidemic Detection & Alerts")

# Detectar picos anormales en diagnósticos
diagnosis_trends = df_history.groupby([
    df_history['fecha_consulta'].dt.to_period('W'),
    'diagnostico_nombre'
]).size().reset_index(name='cases')

# Top 5 diagnósticos para monitorear
top_diagnoses = df_history['diagnostico_nombre'].value_counts().head(5).index

# Crear tabs para cada diagnóstico
tabs = st.tabs(top_diagnoses.tolist())

for i, diagnosis in enumerate(top_diagnoses):
    with tabs[i]:
        diagnosis_data = diagnosis_trends[diagnosis_trends['diagnostico_nombre'] == diagnosis].copy()
        diagnosis_data['fecha_consulta'] = diagnosis_data['fecha_consulta'].astype(str)
        
        # Calcular media móvil y desviación estándar
        diagnosis_data['moving_avg'] = diagnosis_data['cases'].rolling(window=4, min_periods=1).mean()
        diagnosis_data['std'] = diagnosis_data['cases'].rolling(window=4, min_periods=1).std()
        diagnosis_data['upper_limit'] = diagnosis_data['moving_avg'] + 2 * diagnosis_data['std']
        
        # Detectar anomalías
        diagnosis_data['is_outbreak'] = diagnosis_data['cases'] > diagnosis_data['upper_limit']
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=diagnosis_data['fecha_consulta'],
            y=diagnosis_data['cases'],
            mode='lines+markers',
            name='Cases',
            line=dict(color='blue')
        ))
        fig5.add_trace(go.Scatter(
            x=diagnosis_data['fecha_consulta'],
            y=diagnosis_data['moving_avg'],
            mode='lines',
            name='Moving Average',
            line=dict(color='green', dash='dash')
        ))
        fig5.add_trace(go.Scatter(
            x=diagnosis_data['fecha_consulta'],
            y=diagnosis_data['upper_limit'],
            mode='lines',
            name='Alert Threshold',
            line=dict(color='red', dash='dot')
        ))
        
        # Marcar outbreaks
        outbreak_data = diagnosis_data[diagnosis_data['is_outbreak']]
        if not outbreak_data.empty:
            fig5.add_trace(go.Scatter(
                x=outbreak_data['fecha_consulta'],
                y=outbreak_data['cases'],
                mode='markers',
                name='Potential Outbreak',
                marker=dict(color='red', size=12, symbol='x')
            ))
        
        fig5.update_layout(
            title=f"Epidemic Monitoring: {diagnosis}",
            xaxis_title="Week",
            yaxis_title="Number of Cases"
        )
        st.plotly_chart(fig5, use_container_width=True)
        
        if not outbreak_data.empty:
            st.error(f"⚠️ Potential outbreak detected for {diagnosis} in weeks: {', '.join(outbreak_data['fecha_consulta'].astype(str))}")

# SECCIÓN 4: RECOMENDACIONES AUTOMÁTICAS
st.header("💡 Automated Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resource Optimization")
    recommendations = []
    
    # Análisis de personal
    workload = len(df_history) / len(df_patients.groupby('centro_salud_id').first())
    if workload > 5000:
        recommendations.append({
            'priority': 'High',
            'action': 'Hire additional medical staff',
            'reason': f'Average workload is {workload:.0f} consultations per center'
        })
    
    # Análisis de inventario
    critical_meds = med_analysis[med_analysis['days_until_stockout'] < 7]
    if len(critical_meds) > 0:
        recommendations.append({
            'priority': 'Critical',
            'action': f'Urgent restock needed for {len(critical_meds)} medications',
            'reason': 'Less than 7 days of stock remaining'
        })
    
    # Análisis de accesibilidad
    far_patients = patient_risk[patient_risk['distancia_centro_km'] > 30]
    if len(far_patients) > 100:
        recommendations.append({
            'priority': 'Medium',
            'action': 'Implement mobile health brigades',
            'reason': f'{len(far_patients)} patients live >30km from health center'
        })
    
    # Mostrar recomendaciones
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
    
    # Crear matriz de priorización
    priority_matrix = pd.DataFrame({
        'Intervention': ['Mobile Brigades', 'Telemedicine', 'Stock Management', 
                        'Staff Training', 'Prevention Programs'],
        'Impact': [85, 70, 90, 60, 75],
        'Feasibility': [60, 80, 85, 90, 70],
        'Cost': [30, 20, 15, 10, 25]
    })
    
    fig6 = px.scatter(
        priority_matrix,
        x='Feasibility',
        y='Impact',
        size='Cost',
        text='Intervention',
        title="Intervention Priority Matrix",
        size_max=50
    )
    fig6.update_traces(textposition='top center')
    fig6.add_vline(x=70, line_dash="dash", line_color="gray")
    fig6.add_hline(y=70, line_dash="dash", line_color="gray")
    st.plotly_chart(fig6, use_container_width=True)

# Footer con última actualización
st.markdown("---")
st.markdown(f"🔮 **Predictive Analytics Module** | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")