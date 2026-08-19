import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Clasificador de Objetos",
    page_icon="☁️",
    layout="centered"
)

# ==========================================
# TÍTULO
# ==========================================

st.title("☁️ Clasificador de Objetos con IA")
st.write("Examen de Computación en la Nube")
st.write("Universidad Tecnológica de Honduras - UTH")
st.write(
    "Sube una imagen o toma una fotografía para que "
    "el modelo identifique el objeto."
)


# ==========================================
# CLASES CIFAR-10
# ==========================================

class_names = [
    "Avión",
    "Automóvil",
    "Pájaro",
    "Gato",
    "Ciervo",
    "Perro",
    "Rana",
    "Caballo",
    "Barco",
    "Camión"
]

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():

    modelo = tf.keras.models.load_model(
        "clasificador_objetos.keras"
    )
    return modelo

model = cargar_modelo()
# ==========================================
# PREPARAR IMAGEN
# ==========================================
def preparar_imagen(image):
    # Convertir a RGB
    image = image.convert("RGB")

    # CIFAR-10 trabaja con imágenes 32x32
    image = image.resize(
        (32, 32)
    )

    # Convertir imagen a arreglo NumPy
    image = np.array(
        image,
        dtype=np.float32
    )
    # Normalizar valores entre 0 y 1
    image = image / 255.0

    # Agregar dimensión para el modelo
    image = np.expand_dims(
        image,
        axis=0
    )
    return image


# ==========================================
# FUNCIÓN DE PREDICCIÓN
# ==========================================

def predecir(image):
    imagen_procesada = preparar_imagen(
        image
    )
    predicciones = model.predict(
        imagen_procesada,
        verbose=0
    )[0]
    indice = np.argmax(
        predicciones
    )
    objeto = class_names[
        indice
    ]
    confianza = predicciones[
        indice
    ]
    return objeto, confianza
# ==========================================
# SUBIR IMAGEN
# ==========================================

st.subheader(
    "Subir una imagen"
)

archivo = st.file_uploader(
    "Selecciona una imagen",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

# ==========================================
# ANALIZAR IMAGEN SUBIDA
# ==========================================
if archivo is not None:
    imagen = Image.open(
        archivo
    )
    st.image(
        imagen,
        caption="Imagen seleccionada",
        use_container_width=True
    )
    if st.button(
        " Analizar imagen"
    ):
        with st.spinner(
            "Analizando imagen..."
        ):
            objeto, confianza = predecir(
                imagen
            )
        st.success(
            f"Predicción: {objeto}"
        )
        st.write(
            f"Confianza: {confianza:.2%}"
        )

        st.progress(
            int(confianza * 100)
        )

# ==========================================
# CÁMARA
# ==========================================

st.divider()
st.subheader(
    "Tomar una fotografía"
)
foto = st.camera_input(
    "Toma una fotografía"
)

# ==========================================
# ANALIZAR FOTO DE CÁMARA
# ==========================================

if foto is not None:

    imagen_camara = Image.open(
        foto
    )

    st.image(
        imagen_camara,
        caption="Fotografía tomada",
        use_container_width=True
    )

    if st.button(
        " Analizar fotografía"
    ):
        with st.spinner(
            "Analizando fotografía..."
        ):
            objeto, confianza = predecir(
                imagen_camara
            )
        st.success(
            f"Predicción: {objeto}"
        )
        st.write(
            f"Confianza: {confianza:.2%}"
        )
        st.progress(
            int(confianza * 100)
        )
# ==========================================
# INFORMACIÓN DEL MODELO
# ==========================================

st.divider()

with st.expander(
    " Información del modelo"
):
    st.write(
        "El modelo fue entrenado utilizando el dataset CIFAR-10."
    )
    st.write(
        "Puede identificar 10 categorías:"
    )
    st.write(
        """
        - Avión
        - Automóvil
        - Pájaro
        - Gato
        - Ciervo
        - Perro
        - Rana
        - Caballo
        - Barco
        - Camión
        """
    )
# ==========================================
# PIE DE PÁGINA
# ==========================================
st.divider()
st.write(
    "Desarrollado por: José Luis Domínguez"
)
st.write(
    "Universidad Tecnológica de Honduras - UTH"
)
st.write(
    "Computación en la Nube"
)
