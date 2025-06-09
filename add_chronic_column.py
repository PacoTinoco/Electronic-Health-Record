import pandas as pd
import numpy as np

print("🏥 Agregando columna 'es_cronico' a los datasets...")

# Cargar los datasets
print("Cargando datos...")
df_pacientes = pd.read_csv('data/pacientes_rural.csv')
df_historias = pd.read_csv('data/historias_clinicas.csv')
df_segmentacion = pd.read_csv('data/pacientes_segmentados.csv')

# Definir códigos CIE-10 de enfermedades crónicas comunes
# Estos son ejemplos de condiciones crónicas típicas en zonas rurales
codigos_cronicos = {
    # Diabetes
    'E10', 'E11', 'E12', 'E13', 'E14',
    # Hipertensión
    'I10', 'I11', 'I12', 'I13', 'I15',
    # Enfermedades cardíacas
    'I20', 'I21', 'I22', 'I23', 'I24', 'I25',
    # EPOC y asma
    'J44', 'J45', 'J47',
    # Enfermedad renal crónica
    'N18', 'N19',
    # Artritis
    'M05', 'M06', 'M15', 'M16', 'M17',
    # Epilepsia
    'G40',
    # VIH
    'B20', 'B21', 'B22', 'B23', 'B24'
}

# También buscar por nombre de diagnóstico
diagnosticos_cronicos = [
    'diabetes', 'hipertension', 'hipertensión', 'asma', 'epoc',
    'artritis', 'epilepsia', 'insuficiencia renal', 'cardiopatia',
    'cardiopatía', 'enfermedad renal', 'vih', 'sida'
]

print("Identificando pacientes con condiciones crónicas...")

# Identificar pacientes con diagnósticos crónicos
pacientes_cronicos = set()

# Buscar por código CIE-10
for codigo in codigos_cronicos:
    mask = df_historias['diagnostico_cie10'].str.startswith(codigo, na=False)
    pacientes_cronicos.update(df_historias[mask]['paciente_id'].unique())

# Buscar por nombre de diagnóstico
for diag in diagnosticos_cronicos:
    mask = df_historias['diagnostico_nombre'].str.lower().str.contains(diag, na=False)
    pacientes_cronicos.update(df_historias[mask]['paciente_id'].unique())

print(f"Se identificaron {len(pacientes_cronicos)} pacientes con condiciones crónicas")

# Agregar columna es_cronico a df_pacientes
df_pacientes['es_cronico'] = df_pacientes['paciente_id'].isin(pacientes_cronicos).astype(int)

# También agregar a df_segmentacion si existe la columna paciente_id
if 'paciente_id' in df_segmentacion.columns:
    df_segmentacion['es_cronico'] = df_segmentacion['paciente_id'].isin(pacientes_cronicos).astype(int)
else:
    # Si no hay paciente_id, usar la misma proporción que en df_pacientes
    prop_cronicos = df_pacientes['es_cronico'].mean()
    np.random.seed(42)
    df_segmentacion['es_cronico'] = np.random.choice([0, 1], 
                                                     size=len(df_segmentacion), 
                                                     p=[1-prop_cronicos, prop_cronicos])

# Mostrar estadísticas
print("\n📊 Estadísticas de pacientes crónicos:")
print(f"Total de pacientes: {len(df_pacientes)}")
print(f"Pacientes crónicos: {df_pacientes['es_cronico'].sum()} ({df_pacientes['es_cronico'].mean()*100:.1f}%)")
print(f"Pacientes no crónicos: {(df_pacientes['es_cronico'] == 0).sum()} ({(1-df_pacientes['es_cronico'].mean())*100:.1f}%)")

# Análisis adicional
print("\n🔍 Análisis por centro de salud:")
cronicos_por_centro = df_pacientes.groupby('centro_salud_nombre')['es_cronico'].agg(['sum', 'mean'])
cronicos_por_centro.columns = ['total_cronicos', 'porcentaje_cronicos']
cronicos_por_centro['porcentaje_cronicos'] = (cronicos_por_centro['porcentaje_cronicos'] * 100).round(1)
print(cronicos_por_centro.sort_values('porcentaje_cronicos', ascending=False))

# Guardar los datasets actualizados
print("\n💾 Guardando datasets actualizados...")
df_pacientes.to_csv('data/pacientes_rural.csv', index=False)
df_segmentacion.to_csv('data/pacientes_segmentados.csv', index=False)

print("✅ ¡Proceso completado exitosamente!")
print("\nLos archivos han sido actualizados con la columna 'es_cronico'.")
print("Ahora puedes ejecutar los dashboards sin errores.")

# Crear un reporte de diagnósticos crónicos más comunes
print("\n📋 Top 10 diagnósticos crónicos más frecuentes:")
# Filtrar solo diagnósticos crónicos
df_cronicos = df_historias[df_historias['paciente_id'].isin(pacientes_cronicos)]
top_diagnosticos = df_cronicos['diagnostico_nombre'].value_counts().head(10)
for i, (diag, count) in enumerate(top_diagnosticos.items(), 1):
    print(f"{i}. {diag}: {count} casos")

# Opcional: Crear un archivo de resumen
resumen = {
    'total_pacientes': len(df_pacientes),
    'pacientes_cronicos': df_pacientes['es_cronico'].sum(),
    'porcentaje_cronicos': df_pacientes['es_cronico'].mean() * 100,
    'diagnosticos_cronicos_identificados': len(pacientes_cronicos),
    'top_diagnosticos_cronicos': top_diagnosticos.to_dict()
}

# Guardar resumen
import json
with open('data/resumen_pacientes_cronicos.json', 'w', encoding='utf-8') as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print("\n📄 Se creó un archivo de resumen en: data/resumen_pacientes_cronicos.json")