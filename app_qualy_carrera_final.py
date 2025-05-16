import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai
openai.api_key = st.secrets["openai"]["api_key"]


# --- Cargar datos y modelos ---
df_race = pd.read_csv("df2.csv", parse_dates=["RACE_DATE"])
df_qualy = pd.read_csv("df_qualy2.csv", parse_dates=["GP_DATE"])
df_drivers = pd.read_csv("df_drivers.csv")
df_constructors = pd.read_csv("df_constructors.csv")


with open("DetGrid2.pkl", "rb") as f:
    model_qualy = pickle.load(f)
with open("formula1_top88_model.pkl", "rb") as f:
    model_carrera = pickle.load(f)


# --- Inicializar estado ---
if 'poles' not in st.session_state:
    st.session_state.poles = {}  # piloto: número de poles


if 'victorias' not in st.session_state:
    st.session_state.victorias = {}  # piloto: número de victorias


if 'pilotos_config' not in st.session_state:
    st.session_state.pilotos_config = {}
if 'calendario' not in st.session_state:
    st.session_state.calendario = []
if 'resultados_mundial' not in st.session_state:
    st.session_state.resultados_mundial = {}
if 'carrera_actual' not in st.session_state:
    st.session_state.carrera_actual = 0
    
if 'historial_carreras' not in st.session_state:
    st.session_state.historial_carreras = []

if 'clasificacion_mundial' not in st.session_state:
    st.session_state.clasificacion_mundial = {}
    
    

def obtener_cronica_desde_gpt(prompt):
    return "Este es un espacio reservado para la crónica automática. Copia el siguiente prompt y pégalo en tu GPT personalizado:\n\n" + prompt




def generar_prompt_presentacion(pilotos_config):
    texto = "### Presentación del Campeonato de F1 Simulado\n\n"
    texto += "Resumen de los 8 binomios seleccionados y sus expectativas:\n\n"

    for piloto, config in pilotos_config.items():
        anio_piloto = config["anio_piloto"]
        constructor = config["constructor"]
        anio_constructor = config["anio_constructor"]
        
        nombre_formateado = piloto.replace("_", " ").title()
        texto += f"- {nombre_formateado} ({anio_piloto}) con el {constructor} del {anio_constructor}\n"

    texto += "\nEscribe una presentación con tono narrativo deportivo: destaca quiénes parten como favoritos, quiénes podrían sorprender, y qué combinaciones parecen más arriesgadas o desequilibradas. Añade emoción y expectativas para la temporada."
    return texto

   
# --- Función: generar features para clasificación (Qualy) ---
def generar_features_qualy(pilotos_config, circuito):
    rows = []
    for piloto, config in pilotos_config.items():
        if not piloto or not config['constructor']:
            continue
        try:
            driver_id = df_drivers[df_drivers['DRIVERREF'] == piloto]['DRIVER_ID'].iloc[0]
            constructor_id = df_constructors[df_constructors['NAME#'].str.lower() == config['constructor'].lower()]['CONSTRUCTOR_ID'].iloc[0]


            driver_data = df_qualy[
                (df_qualy['DRIVER_ID'] == driver_id) &
                (df_qualy['GP_DATE'].dt.year == config['anio_piloto'])
            ].sort_values(by="GP_DATE").iloc[-1]


            constructor_data = df_qualy[
                (df_qualy['CONSTRUCTOR_ID'] == constructor_id) &
                (df_qualy['GP_DATE'].dt.year == config['anio_constructor'])
            ].sort_values(by="GP_DATE").iloc[-1]


            afinidad_series = df_qualy[
                (df_qualy['DRIVER_ID'] == driver_id) &
                (df_qualy['CIRCUIT_NAME'].str.lower() == circuito.lower())
            ]['AFINIDAD_QUALY_3']


            afinidad_qualy = afinidad_series.iloc[0] if not afinidad_series.empty else 0.5


            rows.append({
                'PILOTO': piloto,
                'QUALY_PERFORMANCE': driver_data['QUALY_PERFORMANCE'],
                'CT_QUALY_PERFORMANCE': constructor_data['CT_QUALY_PERFORMANCE'],
                'AFINIDAD_QUALY_3': afinidad_qualy
            })


        except Exception as e:
            st.warning(f"⚠️ Error al procesar {piloto}: {e}")
            continue


    return pd.DataFrame(rows)

def generar_prompt_para_gpt(carrera):
    circuito = carrera["circuito"]
    lluvia = carrera["lluvia"]
    resultados = carrera["resultados"]
    clasif_antes = carrera["clasificacion_antes"]
    clasif_despues = carrera["clasificacion_despues"]

    poleman = sorted(resultados, key=lambda x: x["salida"])[0]["piloto"]
    ganador = sorted(resultados, key=lambda x: x["llegada"])[0]["piloto"]

    resumen = f"Resumen del GP de {circuito}\n\n"
    resumen += f"Condiciones: {'Lluvia' if lluvia else 'Seco'}\n"
    resumen += f"Pole position: {poleman}\n"
    resumen += f"Ganador: {ganador}\n\n"
    resumen += "Resultados:\n"

    for r in resultados:
        resumen += f"- {r['piloto']}: salía {r['salida']}º, terminó {r['llegada']}º, +{r['puntos_f1']} pts\n"

    resumen += "\nClasificación antes de la carrera:\n"
    for i, (piloto, pts) in enumerate(sorted(clasif_antes.items(), key=lambda x: x[1], reverse=True), 1):
        resumen += f"{i}. {piloto} — {pts} pts\n"

    resumen += "\nClasificación después de la carrera:\n"
    for i, (piloto, pts) in enumerate(sorted(clasif_despues.items(), key=lambda x: x[1], reverse=True), 1):
        resumen += f"{i}. {piloto} — {pts} pts\n"

    resumen += "\nEscribe una crónica deportiva con estilo narrativo profesional, destacando lo más relevante y comentando los cambios en la clasificación."

    return resumen



# --- Función: simular carrera con ajustes ---
def simular_carrera(pilotos_config, circuito, lluvia, grid_ordenado, neutro=False):
    rows = []
    debug_info = []


    for pos, piloto in enumerate(grid_ordenado, start=1):
        try:
            config = pilotos_config[piloto]
            anio_piloto = int(config['anio_piloto'])  # 🔐 asegurar tipo
            anio_constructor = int(config['anio_constructor'])


            driver_id = df_drivers[df_drivers['DRIVERREF'] == piloto]['DRIVER_ID'].iloc[0]
            constructor_id = df_constructors[df_constructors['NAME#'] == config['constructor']]['CONSTRUCTOR_ID'].iloc[0]


            driver_data = df_race[
                (df_race['DRIVER_ID'] == driver_id) &
                (df_race['RACE_DATE'].dt.year == anio_piloto)
            ].sort_values(by="RACE_DATE").iloc[-1]


            constructor_data = df_race[
                (df_race['CONSTRUCTOR_ID'] == constructor_id) &
                (df_race['RACE_DATE'].dt.year == anio_constructor)
            ].sort_values(by="RACE_DATE").iloc[-1]


            afinidad_row = df_race[
                (df_race['DRIVER_ID'] == driver_id) &
                (df_race['CIRCUIT_NAME'].str.lower() == circuito.lower())
            ]
            afinidad = afinidad_row['AFINIDAD3'].iloc[0] if not afinidad_row.empty else 0.5


            # Filtrar carreras con lluvia EN ESE AÑO
            driver_year_lluvia = df_race[
                (df_race['DRIVER_ID'] == driver_id) &
                (df_race['RACE_DATE'].dt.year == anio_piloto) &
                (df_race['LLUVIA'] == 1) &
                (df_race['AFINIDAD_LLUVIA_REAL'].notna()) &
                (df_race['AFINIDAD_LLUVIA_REAL'] > 0)
            ]


            afinidad_lluvia_real = (
                driver_year_lluvia['AFINIDAD_LLUVIA_REAL'].iloc[0]
                if not driver_year_lluvia.empty else 0
            )


            # ✅ Añadir info de debug
            debug_info.append({
                "Piloto": piloto,
                "Driver_ID": driver_id,
                "Año seleccionado": anio_piloto,
                "Carreras con lluvia en ese año": len(driver_year_lluvia),
                "AFINIDAD_LLUVIA_REAL usada": afinidad_lluvia_real,
                "¿Llueve en esta carrera?": lluvia
            })


            rows.append({
                'PILOTO': piloto,
                'GRID': pos,
                'PREV_POSITIONS': 10,
                'ADJUSTED_PERFORMANCE': driver_data['ADJUSTED_PERFORMANCE'],
                'GENERAL_PERFORMANCE': driver_data['GENERAL_PERFORMANCE'],
                'PERFORMANCE_RATIO': driver_data['PERFORMANCE_RATIO'],
                'CONSTRUCTOR_PERFORMANCE': constructor_data['CONSTRUCTOR_PERFORMANCE'],
                'AFINIDAD3': afinidad,
                'AFINIDAD_LLUVIA_REAL': afinidad_lluvia_real,
                'LLUVIA': lluvia
            })


        except Exception as e:
            st.warning(f"⚠️ Error con {piloto}: {e}")
            debug_info.append({
                "Piloto": piloto,
                "Error": str(e)
            })
            continue


    df_features = pd.DataFrame(rows)
    X = df_features.drop(columns="PILOTO")
    probs = model_carrera.predict_proba(X)
    ponderaciones = [0, 25, 18, 15, 12, 10, 8, 6, 4]


    pesos_grid_por_circuito = {
        "Circuit de Monaco": 5, "Hungaroring": 1, "Autodromo Enzo e Dino Ferrari": 0.5,
        "Circuit de Barcelona-Catalunya": 0.5, "Albert Park Grand Prix Circuit": 0.5,
        "Circuit Gilles Villeneuve": 0.5
    }
    peso_grid = pesos_grid_por_circuito.get(circuito, 0.1)


    resultados = []
    for i, row in df_features.iterrows():
        base_score = sum(ponderaciones[j] * probs[i][j] for j in range(len(ponderaciones)))
        if neutro:
            factor=1
        else:
            if row['LLUVIA']:
                factor = (
                    1 + 0.05 * row['CONSTRUCTOR_PERFORMANCE']
                    + 0.05 * row['AFINIDAD3']
                    + 0.13 * row['AFINIDAD_LLUVIA_REAL']
                    + 0.05 * row['PERFORMANCE_RATIO']
                )
            else:
                factor = (
                    1 + 0.12 * row['CONSTRUCTOR_PERFORMANCE']
                    + 0.05 * row['AFINIDAD3']
                    + 0.065 * row['PERFORMANCE_RATIO']
                    + peso_grid / row['GRID']
                )
        final_score = base_score * factor
        resultados.append((final_score, row['PILOTO']))


    resultados.sort(reverse=True)
    return resultados







# --- Interfaz de usuario ---
st.title("🏁 Simulador Mundial F1: Qualy + Carrera")


if st.button("🔄 Resetear Mundial"):
    st.session_state.pilotos_config = {}
    st.session_state.calendario = []
    st.session_state.resultados_mundial = {}
    st.session_state.carrera_actual = 0
    st.session_state.poles = {}
    st.session_state.victorias = {}
    st.success("Estado reiniciado")




st.header("1. Selecciona 8 pilotos y sus escuderías")

driverrefs = sorted(df_drivers['DRIVERREF'].unique())
constructorrefs = sorted(df_constructors['NAME#'].dropna().unique())

for i in range(8):
    st.markdown(f"### Piloto {i+1}")

    # PILOTO y AÑO
    col1, col2 = st.columns([2, 1])
    with col1:
        piloto = st.selectbox(f"Piloto {i+1}", driverrefs, key=f"piloto_select_{i}")
    with col2:
        driver_id = df_drivers[df_drivers['DRIVERREF'] == piloto]['DRIVER_ID'].iloc[0]
        years_piloto = sorted(df_race[df_race['DRIVER_ID'] == driver_id]['RACE_DATE'].dt.year.unique())
        anio_piloto = st.selectbox(f"Año piloto {i+1}", years_piloto, key=f"anio_piloto_select_{i}")

    # ESCUDERÍA y AÑO
    col3, col4 = st.columns([2, 1])
    with col3:
        constructor = st.selectbox(f"Escudería {i+1}", constructorrefs, key=f"constructor_select_{i}")
    with col4:
        constructor_id = df_constructors[df_constructors['NAME#'] == constructor]['CONSTRUCTOR_ID'].iloc[0]
        years_constructor = sorted(df_race[df_race['CONSTRUCTOR_ID'] == constructor_id]['RACE_DATE'].dt.year.unique())
        anio_constructor = st.selectbox(f"Año escudería {i+1}", years_constructor, key=f"anio_constructor_select_{i}")

    # Guardar config
    st.session_state.pilotos_config[piloto] = {
        "anio_piloto": anio_piloto,
        "constructor": constructor,
        "anio_constructor": anio_constructor
    }

    st.markdown("---")



st.header("2. Define los 10 circuitos y si llueve")
circuitos = sorted(df_race['CIRCUIT_NAME'].dropna().unique())
calendario_temporal = []
for i in range(10):
    circuito = st.selectbox(f"Circuito {i+1}", circuitos, key=f"circuito_{i}")
    lluvia = st.radio(f"¿Llueve en {circuito}?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key=f"lluvia_{i}")
    calendario_temporal.append((circuito, lluvia))


if st.button("📅 Guardar calendario"):
    st.session_state.calendario = calendario_temporal
    st.success("Calendario guardado correctamente")
    
if st.button("🎙️ Generar presentación de pilotos y equipos"):
    prompt_presentacion = generar_prompt_presentacion(st.session_state.pilotos_config)
    st.text_area("Prompt para GPT (Presentación previa al mundial)", prompt_presentacion, height=400)

   
simulacion_neutra = st.checkbox("Simular mundial sin ajustes", value=False)


if st.button("Simular siguiente carrera"):
    if not st.session_state.calendario:
        st.warning("Primero guarda el calendario con el botón '📅 Guardar calendario'")
    elif st.session_state.carrera_actual < len(st.session_state.calendario):
        circuito, lluvia = st.session_state.calendario[st.session_state.carrera_actual]
        df_qualy_features = generar_features_qualy(st.session_state.pilotos_config, circuito)
        X_qualy = df_qualy_features.drop(columns="PILOTO")
        preds = model_qualy.predict_proba(X_qualy)
        ponderaciones = [0, 25, 18, 15, 12, 10, 8, 6, 4]


# Factor de peso para escudería
        factor_ct = 0.2  # más peso => mayor efecto del rendimiento de escudería


        scores = []
        for prob, piloto, ct_perf in zip(preds, df_qualy_features["PILOTO"], df_qualy_features["CT_QUALY_PERFORMANCE"]):
            base = sum(p * w for p, w in zip(prob, ponderaciones))
            ajuste = base * (1 + factor_ct * ct_perf)
            scores.append((ajuste, piloto))


        grid = [p for _, p in sorted(scores, reverse=True)]




        st.subheader(f"🔢 Clasificación en {circuito}")

# Ordenar por puntuación (ya calculada previamente)
        scores_ordenados = sorted(scores, reverse=True)
        base_ajuste = scores_ordenados[0][0]

        grid = [p for _, p in scores_ordenados]  # lista final ordenada por rendimiento

        for i, (ajuste, piloto) in enumerate(scores_ordenados, start=1):
            tiempo_dif = (base_ajuste - ajuste) * 0.001  # cada punto = 0.1 segundos
            if i == 1:
                st.write(f"**P{i}**: {piloto} — 0.000 s")
            else:
                    st.write(f"**P{i}**: {piloto} — +{tiempo_dif:.3f} s")

# Registrar pole
        if grid:
            poleman = grid[0]
            st.session_state.poles[poleman] = st.session_state.poles.get(poleman, 0) + 1




        # ✅ Simulación y debug
        resultados = simular_carrera(st.session_state.pilotos_config, circuito, lluvia, grid,simulacion_neutra)


        #st.markdown("### 🧪 Debug de afinidad lluvia real por piloto")
        #st.dataframe(pd.DataFrame(debug_info))


        st.subheader(f"🏆 Resultado carrera en {circuito}")

        puntos_f1 = [10, 8, 6, 5, 4, 3, 2, 1]
        base_score = resultados[0][0]  # puntuación del ganador (referencia)

        for i, (score, piloto) in enumerate(resultados, start=1):
            puntos_asignados = puntos_f1[i-1] if i <= 8 else 0
            tiempo = (base_score - score) * 0.5  # conversión a segundos
            if i == 1:
                st.write(f"{i}. {piloto} — 0.00 s")
            else:
                st.write(f"{i}. {piloto} — +{tiempo:.2f} s")
            st.session_state.resultados_mundial[piloto] = st.session_state.resultados_mundial.get(piloto, 0) + puntos_asignados

        if resultados:
            ganador = resultados[0][1]
            st.session_state.victorias[ganador] = st.session_state.victorias.get(ganador, 0) + 1

        # === Guardar resultados carrera + estado del mundial ===
            resultados_dict = []
            for i, (score, piloto) in enumerate(resultados, start=1):
                salida = grid.index(piloto) + 1
                llegada = i
                puntos_f1_asignados = puntos_f1[i-1] if i <= 8 else 0
                resultados_dict.append({
                    "piloto": piloto,
                    "salida": salida,
                    "llegada": llegada,
                    "puntos_f1": puntos_f1_asignados,
                    "score_modelo": score
                })

# Clasificación antes de carrera
            clasificacion_antes = dict(st.session_state.resultados_mundial)

# Actualizar clasificación (ya actualizada arriba, usamos dict de referencia)
            clasificacion_despues = dict(st.session_state.resultados_mundial)

# Guardar datos de carrera en historial
            if 'historial_carreras' not in st.session_state:
                st.session_state.historial_carreras = []

            if 'clasificacion_mundial' not in st.session_state:
                st.session_state.clasificacion_mundial = {}

# Actualizar clasificación "oficial" paralela
            for r in resultados_dict:
                piloto = r['piloto']
                st.session_state.clasificacion_mundial[piloto] = st.session_state.clasificacion_mundial.get(piloto, 0) + r['puntos_f1']

            st.session_state.historial_carreras.append({
                "circuito": circuito,
                "lluvia": bool(lluvia),
                "resultados": resultados_dict,
                "clasificacion_antes": clasificacion_antes,
                "clasificacion_despues": clasificacion_despues
            })
            




        st.session_state.carrera_actual += 1
        
    else:
        st.success("🏁 Mundial finalizado")

#if st.button("📜 Generar crónica (copiar prompt para GPT)"):
                #ultima = st.session_state.historial_carreras[-1]

                #prompt = generar_prompt_para_gpt(ultima)
                #st.text_area("Prompt para GPT:", prompt, height=400)
#st.header("📊 Clasificación General")
#clasificacion = sorted(st.session_state.resultados_mundial.items(), key=lambda x: x[1], reverse=True)
#for i, (piloto, puntos) in enumerate(clasificacion, start=1):
    #st.write(f"{i}. {piloto} — {puntos} puntos")

if st.session_state.historial_carreras:
    ultima = st.session_state.historial_carreras[-1]
    prompt = generar_prompt_para_gpt(ultima)
    st.text_area("📝 Prompt generado", prompt, height=400)

    url_gpt = "https://chatgpt.com/g/g-6824ad0751a481918919c57574db29b3-narrador-f1"  # 🔁 usa tu link real

    st.markdown(f"""
        👉 [Abrir mi GPT personalizado y pegar el prompt]({url_gpt})
        
        ⚠️ Asegúrate de haber copiado el prompt de arriba antes de ir.
    """)


st.header("📄 Resumen del Mundial (Pilotos y Escuderías)")


resumen_mundial = []
for piloto, puntos in st.session_state.resultados_mundial.items():
    config = st.session_state.pilotos_config.get(piloto, {})
    resumen_mundial.append({
        "Piloto": piloto,
        "Año Piloto": config.get("anio_piloto", "❓"),
        "Escudería": config.get("constructor", "❓"),
        "Año Escudería": config.get("anio_constructor", "❓"),
        "Puntos": puntos,
        "Poles": st.session_state.poles.get(piloto, 0),
        "Victorias": st.session_state.victorias.get(piloto, 0)
    })
st.dataframe(pd.DataFrame(resumen_mundial).sort_values("Puntos", ascending=False))


if st.session_state.carrera_actual >= len(st.session_state.calendario):
    st.success("🏁 Mundial finalizado")

    # Ordenar clasificación final
    clasificacion_final = sorted(
        st.session_state.resultados_mundial.items(),
        key=lambda x: x[1],
        reverse=True
    )
    campeon = clasificacion_final[0][0]  # nombre del campeón (driverref)

    

from PIL import Image
import os

# Configurar info
campeon_config = st.session_state.pilotos_config.get(campeon, {})
escuderia = campeon_config.get("constructor")

ruta_piloto = os.path.join("images", f"{campeon}.jpg")
ruta_logo = os.path.join("logos", f"{escuderia}.jpg")
# Mostrar en columnas
col1, col2 = st.columns(2)

with col1:
    if os.path.exists(ruta_piloto):
        st.image(Image.open(ruta_piloto), caption=campeon.replace('_', ' ').title(), width=300)
    else:
        st.warning("No se encontró imagen del campeón.")

with col2:
    

    if os.path.exists(ruta_logo):
        st.image(Image.open(ruta_logo), caption=escuderia.replace('_', ' ').title(), width=400)
    else:
        st.warning("No se encontró logo de la escudería.")






