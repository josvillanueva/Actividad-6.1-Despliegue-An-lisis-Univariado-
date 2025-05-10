#################
import streamlit as st
import plotly.express as px
import pandas as pd

# Aplicar estilos personalizados en el sidebar
st.markdown("""
    <style>
        /* Estilo para el sidebar */
        [data-testid="stSidebar"] {
            background-color: turquoise !important;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data():
    df = pd.read_csv("Wuppi convertido.csv")
    Lista = ["Administrador", "botón correcto", "mini juego", "color presionado", "dificultad", "Juego", 
             "auto push", "tiempo de interacción", "tiempo de lección", "tiempo de sesión"]
    return df, Lista

df, Lista = load_data()
st.sidebar.image("WUUPI.png")

# Selección del tipo de análisis
View = st.sidebar.selectbox(
    label="Tipo de análisis", 
    options=["Extracción de características", "Regresión Lineal", "Regresión No Lineal", "Regresión Logística", "Anova", "Diccionario"]
)

# Selección de variable a analizar
Variable_Cat = st.sidebar.selectbox(
    label="Selecciona una variable",
    options=Lista
)

st.title(f"Análisis de {Variable_Cat} por usuario")

# Lista de usuarios válidos
usuarios_validos = [1, 3, 27, 8]

# Solo ejecutar si el usuario selecciona "Extracción de características"
if View == "Extracción de características":
    # 📊 Crear las cuatro gráficas, una por cada usuario
    figuras = []
    for usuario in usuarios_validos:
        df_usuario = df[df['Usuario'] == usuario]

        # **Filtrar valores 99 y, si aplica, valores 0**
        if Variable_Cat in ["tiempo de lección", "tiempo de sesión"]:
            df_usuario = df_usuario[(df_usuario[Variable_Cat] != 99) & (df_usuario[Variable_Cat] > 0)]
        else:
            df_usuario = df_usuario[df_usuario[Variable_Cat] != 99]

        # Contar frecuencia de la variable
        Tabla_frecuencias = df_usuario[Variable_Cat].value_counts().reset_index()
        Tabla_frecuencias.columns = ["categorias", "frecuencia"]

        # Asignar colores personalizados
        paleta_colores = {
            1: "lightcoral",
            3: "dodgerblue",
            27: "mediumseagreen",
            8: "mediumorchid"
        }

        # 📊 Gráfico de barras por usuario
        fig = px.bar(
            Tabla_frecuencias,
            x="categorias",
            y="frecuencia",
            title=f"Frecuencia de {Variable_Cat} - Usuario {usuario}",
            color_discrete_sequence=[paleta_colores[usuario]]
        )
        figuras.append(fig)

    # 📊 Mostrar todas las gráficas
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.plotly_chart(figuras[0], use_container_width=True)

    with col2:
        st.plotly_chart(figuras[1], use_container_width=True)

    with col3:
        st.plotly_chart(figuras[2], use_container_width=True)

    with col4:
        st.plotly_chart(figuras[3], use_container_width=True)
elif View == "Diccionario":
    st.write("Diccionario de extracción de características")
    st.write("Usuarios: 1:Leonardo, 3:Nicolas, 8:Sergio Angel, 27:Rene")
    st.write("Administrador: 1:Aleida, 2:Nicolas, 3:Leonardo, 4:Dennis, 5:Sergio Angel, 6:Carlos E, 7:Yael D, 8:Austin, 9:Valentin, 10:Erick, 11:Iker B, 12:Kytzia, 13:Benjamin")
    st.write("boton correcto: 0:Boton incorrecto 1:Boton correcto")
    st.write("Mini Juego: 1:Asteroides, 2:Restaurante, 3:Estrellas, 4:Gusanos, 5:Sonidos y animales, 6:Animales y colores, 7: Figuras y colores, 8:Partes del cuerpo, 9:Despegue, 10:Mini Game 0, 11:Mini Game 1, 12:Mini Game 2, 13: Mini Game 3")
    st.write("Color presionado: 1:Violeta, 2:Verde, 3:Amarillo, 4:Azul, 5:Rojo")
    st.write("dificultad: Episodio 1 ,Episodio 2, Episodio 3, Episodio 4")
    st.write("Juego 1:Astro, 2:Cadetes")
    st.write("Auto push: 0:No se presionó 1:Se presionó el Auto Push")
