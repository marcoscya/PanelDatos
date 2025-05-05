import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris, load_wine, load_diabetes

# Configuración de la página
st.set_page_config(
    page_title="Panel de Visualización de Datos",
    page_icon="📊",
    layout="wide"
)

# Título y descripción
st.title("Panel de Visualización de Datos para Ciencia de Datos/IA")
st.markdown("""
Este panel interactivo te permite explorar diferentes conjuntos de datos 
y visualizar sus características utilizando gráficos interactivos.
""")

# Barra lateral para selección de dataset
st.sidebar.header("Configuración")
dataset_name = st.sidebar.selectbox(
    "Selecciona un conjunto de datos",
    ("Iris", "Vino", "Diabetes")
)

# Función para cargar el dataset seleccionado
def get_dataset(name):
    if name == "Iris":
        data = load_iris()  # Usa la función importada de sklearn
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_names'] = [data.target_names[i] for i in data.target]
        return df, "Clasificación de flores Iris"
    elif name == "Vino":
        data = load_wine()  # Usa la función importada de sklearn
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_names'] = [data.target_names[i] for i in data.target]
        return df, "Clasificación de tipos de vino"
    else:
        data = load_diabetes()  # Usa la función importada de sklearn
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, "Predicción de diabetes"

# Cargar el dataset
df, description = get_dataset(dataset_name)

# Mostrar información del dataset
st.header(f"Dataset: {dataset_name}")
st.write(description)

# Mostrar las primeras filas del dataset
st.subheader("Primeras filas del dataset")
st.write(df.head())

# Mostrar estadísticas básicas
st.subheader("Estadísticas descriptivas")
st.write(df.describe())

# Visualizaciones
st.header("Visualizaciones")

# Selección de columnas para visualización
st.subheader("Selecciona columnas para visualizar")
col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox("Eje X", options=df.columns[:-2] if 'target_names' in df.columns else df.columns[:-1])

with col2:
    y_axis = st.selectbox(
        "Eje Y", 
        options=df.columns[:-2] if 'target_names' in df.columns else df.columns[:-1], 
        index=1 if len(df.columns) > 1 else 0)

# Gráfico de dispersión
st.subheader("Gráfico de Dispersión")
if 'target_names' in df.columns:
    fig = px.scatter(df, x=x_axis, y=y_axis, color='target_names',
                    title=f"{x_axis} vs {y_axis}",
                    labels={'target_names': 'Categoría'},
                    hover_data=['target'])
else:
    fig = px.scatter(df, x=x_axis, y=y_axis, color='target',
                    title=f"{x_axis} vs {y_axis}",
                    labels={'target': 'Valor objetivo'},
                    hover_data=['target'])

st.plotly_chart(fig, use_container_width=True)

# Histograma
st.subheader("Histograma")
hist_col = st.selectbox("Selecciona una columna para el histograma", options=df.columns[:-2] if 'target_names' in df.columns else df.columns[:-1])
hist_fig = px.histogram(
    df, x=hist_col, 
    color='target_names' if 'target_names' in df.columns else 'target', 
    title=f"Distribución de {hist_col}")
st.plotly_chart(hist_fig, use_container_width=True)

# Matriz de correlación
st.subheader("Matriz de Correlación")
corr = df.drop(columns=['target', 'target_names'] if 'target_names' in df.columns else ['target']).corr()
corr_fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlación")
st.plotly_chart(corr_fig, use_container_width=True)

# Pie chart de distribución de clases (solo para datasets de clasificación)
if 'target_names' in df.columns:
    st.subheader("Distribución de Clases")
    class_counts = df['target_names'].value_counts()
    pie_fig = px.pie(values=class_counts.values, names=class_counts.index, title="Distribución de Clases")
    st.plotly_chart(pie_fig, use_container_width=True)