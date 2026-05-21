import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración visual de la App web
st.set_page_config(page_title="Predicción de Ventas - Videojuegos", page_icon="🎮", layout="centered")

st.title("PA2 🎮 Predictor de Ventas Globales de Videojuegos | Franco Olaya Gonzaga 73244460")
st.markdown("Esta aplicación interactiva utiliza modelos de Machine Learning entrenados sobre un dataset histórico para estimar la recaudación estimada de un título a nivel global.")

# Recordatorio Importante de la rúbrica: Link al Cuaderno Colab
# Cambia el enlace de abajo por el link real de compartir de TU cuaderno de Colab público
st.markdown("[🔗 Haz clic aquí para ver el Cuaderno de Código en Google COLAB](https://colab.research.google.com/drive/1nG57rNJALiJ6ppasIfL36qbVmnvO2t3v?usp=sharing)", unsafe_allow_html=True)

# Cargar los recursos guardados usando la caché de Streamlit para optimizar velocidad
@st.cache_resource
def load_assets():
    model_lr = joblib.load('modelos/modelo_lr.pkl')
    model_rf = joblib.load('modelos/modelo_rf.pkl')
    le_platform = joblib.load('le_platform.pkl')
    le_genre = joblib.load('le_genre.pkl')
    return model_lr, model_rf, le_platform, le_genre

try:
    modelo_lr, modelo_rf, le_platform, le_genre = load_assets()
except Exception as e:
    st.error(f"Error al cargar los modelos o encoders .pkl: {e}")
    st.stop()

st.header("🛠️ Parámetros del Videojuego")
col1, col2 = st.columns(2)

with col1:
    plataforma_seleccionada = st.selectbox("Selecciona la Plataforma:", options=le_platform.classes_)
    genero_seleccionado = st.selectbox("Selecciona el Género:", options=le_genre.classes_)

with col2:
    modelo_elegido = st.radio("Modelo de Machine Learning a utilizar:", ("Regresión Lineal", "Random Forest Regressor"))

st.subheader("📊 Ventas Estimadas por Región (en Millones de unidades)")
na_sales = st.slider("Ventas en Norteamérica (NA_Sales):", min_value=0.0, max_value=45.0, value=1.0, step=0.1)
eu_sales = st.slider("Ventas en Europa (EU_Sales):", min_value=0.0, max_value=30.0, value=0.5, step=0.1)
jp_sales = st.slider("Ventas en Japón (JP_Sales):", min_value=0.0, max_value=15.0, value=0.1, step=0.1)

# Botón ejecutor
if st.button("🚀 Calcular Predicción"):
    # Transformación del input usando los mismos LabelEncoders de Colab
    platform_encoded = le_platform.transform([plataforma_seleccionada])[0]
    genre_encoded = le_genre.transform([genero_seleccionado])[0]
    
    # Construcción del vector de características para la predicción
    input_data = np.array([[platform_encoded, genre_encoded, na_sales, eu_sales, jp_sales]])
    
    if modelo_elegido == "Regresión Lineal":
        prediccion = modelo_lr.predict(input_data)[0]
    else:
        prediccion = modelo_rf.predict(input_data)[0]
        
    # Control para evitar lógicas de ventas globales negativas por desfases lineales
    prediccion_final = max(0.0, prediccion)
    
    st.success(f"### 🎉 Ventas Globales Estimadas: **{prediccion_final:.2f} Millones de unidades**")
    st.info(f"Cálculo procesado mediante: **{modelo_elegido}**.")