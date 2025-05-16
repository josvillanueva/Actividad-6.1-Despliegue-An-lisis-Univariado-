#################
import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Estilo personalizado en el sidebar
st.markdown("""
    <style>
        /* Estilo para el sidebar */
        [data-testid="stSidebar"] {
            background-color: #40E0D0CC;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    df = pd.read_csv("Wuppi convertido.csv")
    Lista = ["Administrador", "botón correcto", "mini juego", "color presionado", "dificultad", "Juego", "auto push", "tiempo de interacción", "tiempo de lección", "tiempo de sesión"]
    df = df.drop(columns=["Unnamed: 0.1","Unnamed: 0","fecha"])
    return df, Lista

@st.cache_resource
def load_data_2():
    df2 = pd.read_csv("Wuppi convertido2.csv")
    Lista2 = ["Administrador", "botón correcto", "mini juego", "color presionado", "dificultad", "Juego", "auto push", "tiempo de interacción", "tiempo de lección", "tiempo de sesión"]
    df2 = df2.drop(columns=["Unnamed: 0.1","Unnamed: 0","fecha"])
    return df2, Lista2

df, Lista = load_data()
df2, Lista2 = load_data_2()

st.sidebar.image("foto2.png", width=130)
st.sidebar.markdown("---")

# Selector principal para tipo de análisis
View = st.sidebar.selectbox(
    label="Tipo de análisis",
    options=["Bienvenida 🤖", "Extracción de características 🔠", "Regresión Lineal Simple 🔧", "Regresión Lineal Multiple 🔧", "Regresión No Lineal", "Regresión Logística", "Anova", "Diccionario 📖"]
)

# Seccion de Bienvenida
if View == "Bienvenida 🤖":
    st.image("foto.png")
    st.markdown("---")
    st.write("Bienvenido a nuestro dashboard de Wuupi 🦾 ")
    st.write("Este dashboard esta enfocado en el analisis estadistico de Wuupi 📊")
    if st.checkbox("Mostrar primerer DataFrame"):
        st.dataframe(df2.head(10))
    if st.checkbox("Mostrar segundo DataFrame"):
        st.dataframe(df.head(10))
    st.image("robot.jpg", width=130)

user_dict = {
    1: "LEONARDO", 2: "ALEIDA", 3: "nicolas", 4: "JOSE JAVIER", 5: "JESUS ALEJANDRO",
    6: "ramiro isai", 7: "ADRIAN", 8: "SERGIO ANGEL", 9: "DENISSE", 10: "CARLOS ENRIQUE",
    11: "YAEL DAVID", 12: "VALENTIN", 13: "erick", 14: "IKER BENJAMIN", 15: "ERICK OSVALDO",
    16: "CONCEPCION", 17: "KYTZIA", 18: "AUSTIN", 19: "JOSE IGNACIO TADEO", 20: "JOSE IAN",
    21: "ASHLEY", 22: "JOSHUA", 23: "YEREMI YAZMIN", 24: "MA DEL ROSARIO", 25: "BENJAMIN",
    26: "INGRID", 27: "RENE", 28: "CARLOS ABEL", 29: "ARLETT", 30: "ESMERALDA",
    31: "IRVING", 32: "Jesus eduardo"
}
# Inverso para buscar ID por nombre
name_to_id = {v: k for k, v in user_dict.items()}
# Sección de Extracción de características
# Sección de Extracción de características
if View == "Extracción de características 🔠":
    st.image("foto.png")
    st.markdown("---")

    # Lista con los nombres de todos los usuarios
    todos_nombres = list(user_dict.values())

    # Selección múltiple usando nombres
    selected_nombres = st.sidebar.multiselect(
        "Selecciona hasta 6 usuarios",
        options=todos_nombres,
        default=todos_nombres[:4]
    )

    if len(selected_nombres) == 0:
        st.warning("Por favor selecciona al menos un usuario.")
    else:
        # Convertir los nombres seleccionados a IDs
        selected_ids = [name_to_id[nombre] for nombre in selected_nombres]

        # Selección variable a analizar
        Variable = st.sidebar.selectbox(
            label="Selecciona una variable",
            options=Lista
        )
        st.title(f"Análisis de {Variable} por usuario(s)")

        figuras = []

        for usuario in selected_ids:
            df_usuario = df[df['Usuario'] == usuario]
            # Filtrar valores 99 y 0 si aplica
            if Variable in ["tiempo de lección", "tiempo de sesión"]:
                df_usuario = df_usuario[(df_usuario[Variable] != 99) & (df_usuario[Variable] > 0)]
            else:
                df_usuario = df_usuario[df_usuario[Variable] != 99]
            # Contar frecuencia de la variable
            Tabla_frecuencias = df_usuario[Variable].value_counts().reset_index()
            Tabla_frecuencias.columns = ["categorias", "frecuencia"]
            # Colores (puedes personalizar)
            paleta_colores = {
                1: "lightcoral",
                3: "dodgerblue",
                27: "mediumseagreen",
                8: "mediumorchid"
            }
            color_seq = [paleta_colores.get(usuario, "gray")]

            # Crear gráfico
            fig = px.bar(
                Tabla_frecuencias,
                x="categorias",
                y="frecuencia",
                title=f"Frecuencia de {Variable} - Usuario {user_dict[usuario]}",
                color_discrete_sequence=color_seq
            )
            figuras.append(fig)

        # Mostrar gráficos en columnas
        n_cols = min(2, len(figuras))
        cols = st.columns(n_cols)
        for i, fig in enumerate(figuras):
            cols[i % n_cols].plotly_chart(fig, use_container_width=True)


#user_dict = {
#    1: "LEONARDO", 2: "ALEIDA", 3: "nicolas", 4: "JOSE JAVIER", 5: "JESUS ALEJANDRO",
#    6: "ramiro isai", 7: "ADRIAN", 8: "SERGIO ANGEL", 9: "DENISSE", 10: "CARLOS ENRIQUE",
#    11: "YAEL DAVID", 12: "VALENTIN", 13: "erick", 14: "IKER BENJAMIN", 15: "ERICK OSVALDO",
#    16: "CONCEPCION", 17: "KYTZIA", 18: "AUSTIN", 19: "JOSE IGNACIO TADEO", 20: "JOSE IAN",
#    21: "ASHLEY", 22: "JOSHUA", 23: "YEREMI YAZMIN", 24: "MA DEL ROSARIO", 25: "BENJAMIN",
#    26: "INGRID", 27: "RENE", 28: "CARLOS ABEL", 29: "ARLETT", 30: "ESMERALDA",
#    31: "IRVING", 32: "Jesus eduardo"
#}
# Inverso para buscar ID por nombre
#name_to_id = {v: k for k, v in user_dict.items()}

if View == "Regresión Lineal Simple 🔧":
    st.image("foto.png")
    st.markdown("---")
    tab_general, tab_usuario = st.tabs(["Vista General", "Por Usuario"])
    Lista_num = df2.columns
    usuarios = df2['Usuario'].unique()
    Variable_y = st.sidebar.selectbox(label= 'Variable objetivo (y)', options= Lista_num)
    Variable_X = st.sidebar.selectbox(label= 'Variable independiente del modelo simple (x)', options= Lista_num)

    with tab_general:
        with st.container():
            if st.checkbox("Mostrar mapa de calor (Vista General) ☄️", key="general_heatmap"):
                Corr_Factors2_abs = abs(df2.corr())
                plt.figure(figsize=(10,8))
                sns.heatmap(Corr_Factors2_abs, cmap='Oranges', annot=True, fmt='.2f')
                st.pyplot(plt)        
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X=df2[[Variable_X]], y = df2[Variable_y])
            y_pred = model.predict(X=df2[[Variable_X]])
            coef_deter_simple = model.score(X=df2[[Variable_X]], y = df2[Variable_y])
            coef_correl_simple = np.sqrt(coef_deter_simple)
            st.write (f'Coeficiente de correlación múltiple para Usuario 1 (R): {coef_correl_simple:.2f} 🔷')
            if coef_correl_simple >= 0.8:
                st.write("📈 La correlación es fuerte y positiva: las variables están fuertemente relacionadas.")
            elif coef_correl_simple >= 0.5:
                st.write("🔄 La correlación es moderada: existe una relación clara pero no perfecta.")
            elif coef_correl_simple > 0:
                st.write("🔎 La correlación es débil: hay cierta relación pero es débil.")
            else:
                st.write("⚠️ Correlación muy baja o inexistente: las variables no parecen estar relacionadas linealmente.")
            figure5 = px.scatter(data_frame=df2, x=y_pred, y= Variable_y, title='Modelo lineal simple')
            st.plotly_chart(figure5)

    with tab_usuario:
        # Selección de usuarios
        usuarios_ids = df2['Usuario'].unique()
        usuarios_nombres = [user_dict[u] for u in usuarios_ids if u in user_dict]

        nombre_seleccionado_1 = st.selectbox("Selecciona tu usuario 1", options=usuarios_nombres, key='usuario1')
        nombre_seleccionado_2 = st.selectbox("Selecciona tu usuario 2", options=usuarios_nombres, key='usuario2')

        usuario_seleccionado = name_to_id[nombre_seleccionado_1]
        usuario_seleccionado2 = name_to_id[nombre_seleccionado_2]

        # Variables seleccionadas específicas para cada usuario
        Variable_y_usuario = st.sidebar.selectbox('Variable objetivo (y) usuario', options=Lista_num, key='variable_y_usuario')
        Variable_X_usuario = st.sidebar.selectbox('Variable independiente del modelo simple (x) usuario', options=Lista_num, key='variable_x_usuario')

        # Perfil del usuario 1
        with st.container():
            st.markdown(f"### Análisis para Usuario 1: {nombre_seleccionado_1}")
            df_usuario1 = df2[df2['Usuario'] == usuario_seleccionado]
            st.write(f"🔢 Filas analizadas: {df_usuario1.shape[0]}")
            if not df_usuario1.empty:
                if st.checkbox(f"Mostrar mapa de calor {usuario_seleccionado} ☄️", key="general_heatmap2"):
                    Corr_Factors2_abs = abs(df_usuario1.corr())
                    plt.figure(figsize=(10,8))
                    sns.heatmap(Corr_Factors2_abs, cmap='Oranges', annot=True, fmt='.2f')
                    st.pyplot(plt)
                model = LinearRegression()
                model.fit(X=df_usuario1[[Variable_X_usuario]], y=df_usuario1[Variable_y_usuario])
                y_pred = model.predict(X=df_usuario1[[Variable_X_usuario]])
                coef_deter = model.score(X=df_usuario1[[Variable_X_usuario]], y=df_usuario1[Variable_y_usuario])
                coef_correl1 = np.sqrt(coef_deter)
                st.write(f'Coeficiente de correlación múltiple para Usuario 1 (R): {coef_correl1:.2f} 🔷')
                if coef_correl1 >= 0.8:
                    st.write("📈 La correlación es fuerte y positiva: las variables están fuertemente relacionadas.")
                elif coef_correl1 >= 0.5:
                    st.write("🔄 La correlación es moderada: existe una relación clara pero no perfecta.")
                elif coef_correl1 > 0:
                    st.write("🔎 La correlación es débil: hay cierta relación pero es débil.")
                else:
                    st.write("⚠️ Correlación muy baja o inexistente: las variables no parecen estar relacionadas linealmente.")
                fig1 = px.scatter(data_frame=df_usuario1, x=y_pred, y=Variable_y_usuario)
                st.plotly_chart(fig1)
            else:
                st.write('No hay datos para el usuario 1.')
        st.markdown("---")

        # Perfil del usuario 2
        with st.container():
            st.markdown(f"### Análisis para Usuario 2: {nombre_seleccionado_2}")
            df_usuario2 = df2[df2['Usuario'] == usuario_seleccionado2]
            st.write(f"🔢 Filas analizadas: {df_usuario2.shape[0]}")
            if not df_usuario2.empty:
                if st.checkbox(f"Mostrar mapa de calor{usuario_seleccionado2} ☄️", key="general_heatmap 3"):
                    Corr_Factors2_abs = abs(df_usuario2.corr())
                    plt.figure(figsize=(10,8))
                    sns.heatmap(Corr_Factors2_abs, cmap='Oranges', annot=True, fmt='.2f')
                    st.pyplot(plt)
                model = LinearRegression()
                model.fit(X=df_usuario2[[Variable_X_usuario]], y=df_usuario2[Variable_y_usuario])
                y_pred = model.predict(X=df_usuario2[[Variable_X_usuario]])
                coef_deter = model.score(X=df_usuario2[[Variable_X_usuario]], y=df_usuario2[Variable_y_usuario])
                coef_correl2 = np.sqrt(coef_deter)
                st.write(f'Coeficiente de correlación múltiple para Usuario 2 (R): {coef_correl2:.2f} 🔷')
                if coef_correl2 >= 0.8:
                    st.write("📈 La correlación es fuerte y positiva: las variables están fuertemente relacionadas.")
                elif coef_correl2 >= 0.5:
                    st.write("🔄 La correlación es moderada: existe una relación clara pero no perfecta.")
                elif coef_correl2 > 0:
                    st.write("🔎 La correlación es débil: hay cierta relación pero es débil.")
                else:
                    st.write("⚠️ Correlación muy baja o inexistente: las variables no parecen estar relacionadas linealmente.")
                fig2 = px.scatter(data_frame=df_usuario2, x=y_pred, y=Variable_y_usuario)
                st.plotly_chart(fig2)
            else:
                st.write('No hay datos para el usuario 2.')

        st.markdown("---")

        # Comparación final entre usuarios
        with st.container():
            st.markdown("## 📊 Comparación entre Usuarios")

            st.write(f"**{nombre_seleccionado_1}** tiene un coeficiente de correlación de **{coef_correl1:.2f}**.")
            st.write(f"**{nombre_seleccionado_2}** tiene un coeficiente de correlación de **{coef_correl2:.2f}**.")

            diferencia = coef_correl1 - coef_correl2

            if abs(diferencia) < 0.1:
                st.info("Ambos usuarios tienen un nivel de correlación muy similar entre sus variables seleccionadas.")
            elif diferencia > 0:
                st.success(f"{nombre_seleccionado_1} muestra una correlación más fuerte entre sus variables seleccionadas.🥇")
            else:
                st.success(f"{nombre_seleccionado_2} muestra una correlación más fuerte entre sus variables seleccionadas.🥇")

if View == "Regresión Lineal Multiple 🔧":
    st.image("foto.png")
    st.markdown("---")
    st.write("Regresion Lineal multiple")
    tab_general, tab_usuario = st.tabs(["Vista General", "Por Usuario"])
    Lista_num = df2.columns

    with tab_general:
        Variable_y = st.sidebar.selectbox(label= 'Variable objetivo (y)', options= Lista_num)
        Variables_x= st.sidebar.multiselect(label="Variables independientes del modelo múltiple (X)", options= Lista_num, default=["tiempo de lección"])
        if st.checkbox(f"Mostrar mapa de calor ☄️", key="general_heatmap 4"):
            Corr_Factors2_abs = abs(df2.corr())
            plt.figure(figsize=(10,8))
            sns.heatmap(Corr_Factors2_abs, cmap='Oranges', annot=True, fmt='.2f')
            st.pyplot(plt)
        from sklearn.linear_model import LinearRegression
        model_M= LinearRegression()
        model_M.fit(X=df2[Variables_x], y=df2[Variable_y])
        y_pred_M= model_M.predict(X=df2[Variables_x])
        coef_Deter_multiple=model_M.score(X=df2[Variables_x], y=df2[Variable_y])
        coef_Correl_multiple=np.sqrt(coef_Deter_multiple)
        st.write(f'Coeficiente de correlación múltiple para Usuario 1 (R): {coef_Correl_multiple:.2f} 🔷')
        if coef_Correl_multiple >= 0.8:
            st.write("📈 La correlación es fuerte y positiva: las variables están fuertemente relacionadas.")
        elif coef_Correl_multiple >= 0.5:
            st.write("🔄 La correlación es moderada: existe una relación clara pero no perfecta.")
        elif coef_Correl_multiple > 0:
            st.write("🔎 La correlación es débil: hay cierta relación pero es débil.")
        else:
            st.write("⚠️ Correlación muy baja o inexistente: las variables no parecen estar relacionadas linealmente.")
        figure6 = px.scatter(data_frame=df2, x=y_pred_M, y=Variable_y, 
                     title= 'Modelo Lineal Múltiple')
        st.plotly_chart(figure6)

    with tab_usuario:
        usuarios_ids = df2['Usuario'].unique()
        usuarios_nombres = [user_dict[u] for u in usuarios_ids if u in user_dict]

        nombre_seleccionado_1 = st.selectbox("Selecciona tu usuario 1", options=usuarios_nombres, key='usuario1')
        nombre_seleccionado_2 = st.selectbox("Selecciona tu usuario 2", options=usuarios_nombres, key='usuario2')

        usuario_seleccionado = name_to_id[nombre_seleccionado_1]
        usuario_seleccionado2 = name_to_id[nombre_seleccionado_2]


        df_usuario1 = df2[df2['Usuario'] == usuario_seleccionado]
        st.markdown(f"### Análisis para Usuario 1: {nombre_seleccionado_1}")
        st.write(f"🔢 Filas analizadas: {df_usuario1.shape[0]}")
        if not df_usuario1.empty:
            if st.checkbox(f"Mostrar mapa de calor para Usuario 1 ☄️", key='heatmap1'):
                Corr_abs1 = abs(df_usuario1.corr())
                plt.figure(figsize=(10,8))
                sns.heatmap(Corr_abs1, cmap='Oranges', annot=True, fmt='.2f')
                st.pyplot(plt)

            model1 = LinearRegression()
            model1.fit(df_usuario1[Variables_x], df_usuario1[Variable_y])
            y_pred1 = model1.predict(df_usuario1[Variables_x])
            coef_deter = model1.score(df_usuario1[Variables_x], df_usuario1[Variable_y])
            coef_correlacion1 = np.sqrt(coef_deter)
            st.write(f'Coeficiente de correlación múltiple para Usuario 1 (R): {coef_correlacion1:.2f} 🔷')
            if coef_correlacion1 >= 0.8:
                st.write("📈 La correlación es fuerte y positiva: las variables están fuertemente relacionadas.")
            elif coef_correlacion1 >= 0.5:
                st.write("🔄 La correlación es moderada: existe una relación clara pero no perfecta.")
            elif coef_correlacion1 > 0:
                st.write("🔎 La correlación es débil: hay cierta relación pero es débil.")
            else:
                st.write("⚠️ Correlación muy baja o inexistente: las variables no parecen estar relacionadas linealmente.")
            fig1 = px.scatter(x=y_pred1, y=df_usuario1[Variable_y], 
                      labels={'x':'Predicción', 'y':Variable_y},
                      title='Modelo Múltiple Predicho vs. Real - Usuario 1')
            st.plotly_chart(fig1)
        else:
            st.write("No hay datos para el Usuario 1.")

        st.markdown("---")

            # Análisis usuario 2
        df_usuario2 = df2[df2['Usuario'] == usuario_seleccionado2]
        st.markdown(f"### Análisis para Usuario 1: {nombre_seleccionado_2}")
        st.write(f"🔢 Filas analizadas: {df_usuario2.shape[0]}")
        if not df_usuario2.empty:
            if st.checkbox(f"Mostrar mapa de calor para Usuario 2 ☄️", key='heatmap2'):
                Corr_abs2 = abs(df_usuario2.corr())
                plt.figure(figsize=(10,8))
                sns.heatmap(Corr_abs2, cmap='Oranges', annot=True, fmt='.2f')
                st.pyplot(plt)

            model2 = LinearRegression()
            model2.fit(df_usuario2[Variables_x], df_usuario2[Variable_y])
            y_pred2 = model2.predict(df_usuario2[Variables_x])
            coef_deter = model2.score(df_usuario2[Variables_x], df_usuario2[Variable_y])
            coef_correlacion2 = np.sqrt(coef_deter)
            st.write(f'Coeficiente de correlación múltiple para Usuario 2 (R): {coef_correlacion2:.2f} 🔷')
            if coef_correlacion2 >= 0.8:
                st.write("📈 La correlación es fuerte y positiva: las variables están fuertemente relacionadas.")
            elif coef_correlacion2 >= 0.5:
                st.write("🔄 La correlación es moderada: existe una relación clara pero no perfecta.")
            elif coef_correlacion2 > 0:
                st.write("🔎 La correlación es débil: hay cierta relación pero es débil.")
            else:
                st.write("⚠️ Correlación muy baja o inexistente: las variables no parecen estar relacionadas linealmente.")
            fig2 = px.scatter(x=y_pred2, y=df_usuario2[Variable_y], 
                      labels={'x':'Predicción', 'y':Variable_y},
                        title='Modelo Múltiple Predicho vs. Real - Usuario 2')
            st.plotly_chart(fig2)
        else:
                st.write("No hay datos para el Usuario 2.")

        st.markdown("---")

        # Comparación final entre usuarios
        with st.container():
            st.markdown("## 📊 Comparación entre Usuarios")

            st.write(f"**{nombre_seleccionado_1}** tiene un coeficiente de correlación de **{coef_correlacion1:.2f}**.")
            st.write(f"**{nombre_seleccionado_2}** tiene un coeficiente de correlación de **{coef_correlacion2:.2f}**.")

            diferencia = coef_correlacion1 - coef_correlacion2

            if abs(diferencia) < 0.1:
                st.info("Ambos usuarios tienen un nivel de correlación muy similar entre sus variables seleccionadas.")
            elif diferencia > 0:
                st.success(f"{nombre_seleccionado_1} muestra una correlación más fuerte entre sus variables seleccionadas.🥇")
            else:
                st.success(f"{nombre_seleccionado_2} muestra una correlación más fuerte entre sus variables seleccionadas.🥇")

elif View == "Diccionario 📖":
    st.write("Diccionario de extracción de características")
    st.write("Administrador: 1:Aleida, 2:Nicolas, 3:Leonardo, 4:Dennis, 5:Sergio Angel, 6:Carlos E, 7:Yael D, 8:Austin, 9:Valentin, 10:Erick, 11:Iker B, 12:Kytzia, 13:Benjamin")
    st.write("boton correcto: 0:Boton incorrecto 1:Boton correcto")
    st.write("Mini Juego: 1:Asteroides, 2:Restaurante, 3:Estrellas, 4:Gusanos, 5:Sonidos y animales, 6:Animales y colores, 7: Figuras y colores, 8:Partes del cuerpo, 9:Despegue, 10:Mini Game 0, 11:Mini Game 1, 12:Mini Game 2, 13: Mini Game 3")
    st.write("Color presionado: 1:Violeta, 2:Verde, 3:Amarillo, 4:Azul, 5:Rojo")
    st.write("dificultad: Episodio 1 ,Episodio 2, Episodio 3, Episodio 4")
    st.write("Juego 1:Astro, 2:Cadetes")
    st.write("Auto push: 0:No se presionó 1:Se presionó el Auto Push")
