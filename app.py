import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


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

st.set_page_config(
    page_title="Clasificador de Objetos",
    page_icon="",
    layout="centered"
)

st.title("Clasificador de Objetos con IA")
st.write(
    "Examen de Computación en la Nube - UTH"
)
st.write(
    "Sube una imagen y el modelo intentará identificar el objeto."
)