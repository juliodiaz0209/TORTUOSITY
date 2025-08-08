import streamlit as st
from PIL import Image
import torch
import io
import pandas as pd
import os

# Import functions from your original script
# Make sure Tortuosity.py is in the same directory or accessible in the Python path
from Tortuosity import (
    load_maskrcnn_model,
    load_unet_model,
    predict_maskrcnn_model,
    predict_unet_model,
    show_combined_result,
    resize_to_previous_multiple_of_32,
    device  # Use the device defined in Tortuosity.py
)

# --- Page Configuration (Vercel-like) ---
st.set_page_config(
    page_title="Análisis de Tortuosidad Avanzado",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Vercel-like appearance ---
st.markdown("""
    <style>
        /* Base Vercel-like dark theme */
        body {
            color: #e0e0e0; /* Slightly lighter default text color for readability */
            background-color: #0a0a0a; /* Slightly adjusted dark background */
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        }
        .stApp {
            background-color: #0a0a0a;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff; /* Pure white for main headings */
        }
        h1 {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .stButton>button {
            border: 1px solid #3a3a3a;
            border-radius: 0.375rem; /* Softer corners */
            background-color: #1c1c1c; /* Darker button */
            color: #e0e0e0;
            padding: 0.6rem 1.2rem; /* Slightly larger padding */
            font-weight: 500;
            transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        .stButton>button:active {
            background-color: #3a3a3a;
        }

        .stFileUploader {
            border: 1px dashed #3a3a3a;
            background-color: #121212;
            border-radius: 0.375rem;
            padding: 1.5rem;
        }
        .stFileUploader label {
            color: #b0b0b0; /* Lighter uploader label */
            font-weight: 500;
        }
        .stFileUploader > div > div > button { /* Target the 'Browse files' button inside uploader */
            background-color: #28a745; /* Green accent for browse button */
            color: white;
            border: none;
        }
        .stFileUploader > div > div > button:hover {
            background-color: #218838;
        }


        .stMetric {
            background-color: #161616; /* Darker metric background */
            border: 1px solid #2a2a2a; /* Subtle border for metric */
            border-radius: 0.375rem;
            padding: 1rem 1.5rem; /* Adjusted padding */
        }
        .stMetric label {
            color: #a0a0a0 !important; /* Lighter metric label */
            font-size: 0.9rem;
            font-weight: 400;
        }
        .stMetric .stMetricValue {
            color: #ffffff !important; /* White metric value */
            font-size: 1.75rem; /* Larger metric value */
            font-weight: 600;
        }

        /* Dataframe styling */
        .stDataFrame {
            background-color: #161616;
            border: 1px solid #2a2a2a;
            border-radius: 0.375rem;
        }
        .stDataFrame table {
            color: #c0c0c0;
        }
        .stDataFrame th {
            background-color: #202020;
            color: #e0e0e0;
            font-weight: 600;
        }
        .stDataFrame td, .stDataFrame th {
            border-color: #3a3a3a;
        }


        /* Chart container styling */
        .stBarChart, .stLineChart { /* Includes st.bar_chart */
            background-color: #161616;
            border: 1px solid #2a2a2a;
            border-radius: 0.375rem;
            padding: 1rem;
        }

        /* Markdown links */
        a {
            color: #3b82f6; /* Brighter blue for links */
        }
        a:hover {
            color: #60a5fa;
        }

        /* Custom container style */
        .custom-container {
            background-color: #121212; /* Background for containers */
            border: 1px solid #2a2a2a; /* Border for containers */
            border-radius: 0.5rem; /* Rounded corners for containers */
            padding: 1.5rem; /* Padding inside containers */
            margin-bottom: 1.5rem; /* Space between containers */
        }

        /* Footer styling */
        footer {
            text-align: center;
            padding: 2rem 1rem;
            color: #666; /* Softer footer text color */
            font-size: 0.875rem;
        }
        hr {
            border-top: 1px solid #2a2a2a; /* Styled horizontal rule */
        }
    </style>
""", unsafe_allow_html=True)


# --- Model Loading (Cached) ---
@st.cache_resource
def cached_load_maskrcnn_model(path):
    return load_maskrcnn_model(path)

@st.cache_resource
def cached_load_unet_model(path, device):
    return load_unet_model(path, device)

# --- Streamlit App ---
st.title("Análisis Avanzado de Tortuosidad Glandular")

# Define model paths
maskrcnn_model_path = "final_model (11).pth"
unet_model_path = "final_model_tarsus_improved.pth"

# --- File Uploader Section ---
with st.container():
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.subheader("1. Cargar Imagen del Párpado")
    uploaded_file = st.file_uploader(
        "Arrastra y suelta una imagen o haz clic para explorar (formatos: .jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        help="Sube una imagen clara del párpado para el análisis de las glándulas de Meibomio."
    )
    st.markdown('</div>', unsafe_allow_html=True)


if uploaded_file is not None:
    # --- Image Display and Analysis Button ---
    with st.container():
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.subheader("2. Imagen Original y Confirmación")
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption="Imagen Original Cargada", use_container_width=True)

        if st.button("🚀 Analizar Imagen Ahora", help="Haz clic para iniciar el procesamiento y análisis de la imagen."):
            with st.spinner("🔬 Cargando modelos y procesando imagen... Por favor espera."):
                try:
                    temp_image_path = f"temp_{uploaded_file.name}"
                    with open(temp_image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    result_image, tortuosity_data = show_combined_result(temp_image_path, maskrcnn_model_path, unet_model_path, device)

                    # --- Results Display Section ---
                    st.subheader("3. Resultados del Análisis")

                    with st.container(): # Nested container for results grouping
                        st.markdown('<div class="custom-container" style="background-color: #101010;">', unsafe_allow_html=True) # Slightly different bg for results
                        st.image(result_image, caption="Imagen Procesada con Segmentación", use_container_width=True)

                        st.markdown("---") # Visual separator

                        st.subheader("📈 Métricas Clave de Tortuosidad")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="Tortuosidad Promedio Global",
                                value=f"{tortuosity_data['avg_tortuosity']:.3f}"
                            )
                        with col2:
                            st.metric(
                                label="Número de Glándulas Detectadas",
                                value=tortuosity_data['num_glands']
                            )
                        st.markdown('</div>', unsafe_allow_html=True)


                    # --- Detailed Tortuosity Information ---
                    with st.container():
                        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
                        st.subheader("ℹ️ Información Detallada de Tortuosidad")

                        st.markdown("""
                        #### ¿Qué es la Tortuosidad?
                        La tortuosidad es una medida de cuán retorcida o curvada está una glándula de Meibomio.
                        Un valor más alto indica una glándula más tortuosa, lo que puede ser un indicador de disfunción
                        de las glándulas de Meibomio (MGD).

                        **Interpretación de los valores (referencial):**
                        - **0.0 - 0.1:** Tortuosidad baja (generalmente normal)
                        - **0.1 - 0.2:** Tortuosidad moderada (puede indicar cambios iniciales)
                        - **> 0.2:** Tortuosidad alta (sugestivo de MGD, requiere correlación clínica)

                        *Nota: Estos rangos son aproximados y la interpretación final debe ser realizada por un especialista.*
                        """, help="La tortuosidad es un parámetro morfométrico clave.")

                        if tortuosity_data['individual_tortuosities']:
                            st.markdown("---")
                            st.subheader("📊 Tortuosidad por Glándula Individual")

                            df = pd.DataFrame({
                                'ID Glándula': [f"G{i+1}" for i in range(len(tortuosity_data['individual_tortuosities']))],
                                'Valor de Tortuosidad': [f"{t:.3f}" for t in tortuosity_data['individual_tortuosities']]
                            })
                            st.dataframe(df, use_container_width=True)

                            st.subheader("Visualización Gráfica de Tortuosidad")
                            chart_data = pd.DataFrame({
                                'Glándula': [f"G{i+1}" for i in range(len(tortuosity_data['individual_tortuosities']))],
                                'Tortuosidad': tortuosity_data['individual_tortuosities']
                            })
                            st.bar_chart(
                                chart_data.set_index('Glándula'),
                                use_container_width=True,
                                height=350 # Slightly increased height
                            )
                            st.caption("Gráfico comparativo de la tortuosidad para cada glándula identificada. Valores más altos indican mayor curvatura.")
                        st.markdown('</div>', unsafe_allow_html=True)


                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

                except FileNotFoundError as e:
                    st.error(f"❌ Error Crítico: No se encontró el archivo del modelo: {e}. Asegúrate de que los archivos '.pth' estén en el directorio correcto y sean accesibles.")
                except Exception as e:
                    st.error(f"⚠️ Ocurrió un error inesperado durante el procesamiento: {e}. Intenta con otra imagen o revisa la consola para más detalles.")

        else:
            st.info("🖱️ Haz clic en el botón 'Analizar Imagen Ahora' después de cargarla para ver los resultados.")
        st.markdown('</div>', unsafe_allow_html=True) # Close Image Display and Analysis Button container

else:
    st.info("👋 ¡Bienvenido! Comienza subiendo una imagen del párpado para analizar la tortuosidad de las glándulas.")

st.markdown("---")
st.markdown("<footer>Aplicación desarrollada con Streamlit, PyTorch (Mask R-CNN & UNet). <br> Investigación y análisis de imágenes biomédicas.</footer>", unsafe_allow_html=True)