import streamlit as st
import pandas as pd
import numpy as np
import pickle


#st.markdown('<div class="zona_especial">', unsafe_allow_html=True)

#st.write("Este texto tendrá otra fuente")

#st.markdown('</div>', unsafe_allow_html=True)

st.set_page_config(layout="wide")

# --- Cargar datos y modelos ---
df_race = pd.read_csv("df2.csv", parse_dates=["RACE_DATE"])
df_qualy = pd.read_csv("df_qualy2.csv", parse_dates=["GP_DATE"])
df_drivers = pd.read_csv("df_drivers.csv")
df_constructors = pd.read_csv("df_constructors.csv")

import unicodedata

def arreglar_texto_mojibake(txt):
    if pd.isna(txt):
        return txt

    txt = str(txt).strip()

    # Intenta reparar textos mal decodificados tipo "AutÃ³dromo" o "NÃ¼rburgring"
    try:
        txt = txt.encode("latin1").decode("utf-8")
    except:
        pass

    # Normaliza los caracteres Unicode para evitar rarezas visuales
    txt = unicodedata.normalize("NFC", txt)

    # Reemplazos manuales de los circuitos que sabes que salen mal
    reemplazos = {
        "Autódromo Hermanos Rodríguez": "Autódromo Hermanos Rodríguez",
        "Autódromo Internacional Nelson Piquet": "Autódromo Internacional Nelson Piquet",
        "Autódromo Internacional do Algarve": "Autódromo Internacional do Algarve",
        "Autódromo José Carlos Pace": "Autódromo José Carlos Pace",
        "Autódromo Juan y Oscar Gálvez": "Autódromo Juan y Oscar Gálvez",
        "Autódromo do Estoril": "Autódromo do Estoril",
        "Nürburgring": "Nürburgring",

        # Por si alguna variante sigue viniendo rota
        "AutÃ³dromo Hermanos RodrÃ­guez": "Autódromo Hermanos Rodríguez",
        "AutÃ³dromo Internacional Nelson Piquet": "Autódromo Internacional Nelson Piquet",
        "AutÃ³dromo Internacional do Algarve": "Autódromo Internacional do Algarve",
        "AutÃ³dromo JosÃ© Carlos Pace": "Autódromo José Carlos Pace",
        "AutÃ³dromo Juan y Oscar GÃ¡lvez": "Autódromo Juan y Oscar Gálvez",
        "AutÃ³dromo do Estoril": "Autódromo do Estoril",
        "NÃ¼rburgring": "Nürburgring",
    }

    txt = reemplazos.get(txt, txt)

    # Limpieza final de espacios raros
    txt = " ".join(txt.split())

    return txt

df_race["CIRCUIT_NAME"] = df_race["CIRCUIT_NAME"].apply(arreglar_texto_mojibake)
df_qualy["CIRCUIT_NAME"] = df_qualy["CIRCUIT_NAME"].apply(arreglar_texto_mojibake)

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
def obtener_anios_piloto(driver_ref):
    try:
        driver_id = df_drivers[df_drivers['DRIVERREF'] == driver_ref]['DRIVER_ID'].iloc[0]
        years_piloto = sorted(
            df_race[df_race['DRIVER_ID'] == driver_id]['RACE_DATE'].dt.year.unique()
        )
        return years_piloto
    except:
        return []

def obtener_anios_constructor(constructor_name):
    try:
        constructor_id = df_constructors[df_constructors['NAME#'] == constructor_name]['CONSTRUCTOR_ID'].iloc[0]
        years_constructor = sorted(
            df_race[df_race['CONSTRUCTOR_ID'] == constructor_id]['RACE_DATE'].dt.year.unique()
        )
        return years_constructor
    except:
        return []

   
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

    factor_global_grid = 1.55


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
                    + factor_global_grid * (peso_grid / row['GRID'])
                )
        final_score = base_score * factor
        resultados.append((final_score, row['PILOTO']))


    resultados.sort(reverse=True)
    return resultados



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Orbitron:wght@500;700&family=Bebas+Neue&family=Rajdhani:wght@400;500;600;700&family=Oswald:wght@400;500;600;700&family=Teko:wght@400;500;600;700&family=Barlow+Condensed:wght@400;600;700&family=Exo+2:wght@400;600;700&family=Chakra+Petch:wght@400;500;600;700&family=Titillium+Web:wght@400;600;700&family=Anton&family=Russo+One&family=Montserrat:wght@400;600;700&family=Poppins:wght@400;600;700&family=Roboto+Condensed:wght@400;700&family=Oxanium:wght@400;600;700&family=Audiowide&family=Jura:wght@400;600;700&family=Kanit:wght@400;600;700&family=Manrope:wght@400;600;700&family=Archivo:wght@400;600;700&family=Saira+Condensed:wght@400;600;700&family=Prompt:wght@400;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* =========================================================
   CAMBIA SOLO ESTAS 3 LÍNEAS PARA PROBAR FUENTES
   ========================================================= */
:root {
    --font_titulos: 'Orbitron', sans-serif;
    --font_texto: 'Jura', sans-serif;
    --font_labels: 'Oxanium', sans-serif;
}

/* =========================================================
   ANCHO GENERAL DE LA APP
   ========================================================= */
.block-container {
    max-width: 76% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

/* Fondo general */
.stApp {
    background: #0f172a;
}

/* Texto base */
html, body, [class*="css"] {
    color: #e2e8f0 !important;
    font-family: var(--font_texto) !important;
}

/* Títulos principales */
h1, h2, h3 {
    color: #f1f5f9 !important;
    font-family: var(--font_titulos) !important;
    letter-spacing: 0.3px;
}

/* Texto normal */
p, li, div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li {
    color: #f8fafc !important;
    font-family: var(--font_texto) !important;
}

/* Labels de widgets */
label, .stRadio label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    color: #f8fafc !important;
    font-family: var(--font_labels) !important;
    font-weight: 600 !important;
}

/* Caption / texto secundario */
small, span, .stCaption {
    color: #cbd5e1 !important;
    font-family: var(--font_texto) !important;
}

/* Botones */
.stButton > button {
    background: linear-gradient(90deg, #ff4b4b 0%, #ff7b72 100%);
    color: #ffffff !important;
    border-radius: 10px;
    border: none !important;
    font-weight: 700 !important;
    font-family: var(--font_labels) !important;
    padding: 0.55rem 1rem !important;
}

.stButton > button:hover {
    filter: brightness(1.05);
}

/* =========================================================
   SELECTBOX CERRADO
   ========================================================= */
div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    min-height: 44px !important;
}

/* TODO lo interno del select */
div[data-baseweb="select"] *,
div[data-baseweb="select"] span,
div[data-baseweb="select"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    font-family: var(--font_texto) !important;
}

/* Texto que escribes dentro del select */
div[data-baseweb="select"] input {
    color: #ffffff !important;
    caret-color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-family: var(--font_texto) !important;
}

/* Placeholder */
div[data-baseweb="select"] input::placeholder {
    color: #cbd5e1 !important;
    opacity: 1 !important;
}

/* Single value / valor seleccionado */
div[data-baseweb="select"] [data-testid="stMarkdownContainer"],
div[data-baseweb="select"] [class*="singleValue"],
div[data-baseweb="select"] [class*="placeholder"] {
    color: #ffffff !important;
}

/* =========================================================
   DROPDOWN ABIERTO / MENÚ DE OPCIONES
   ========================================================= */

/* Popover raíz */
body [data-baseweb="popover"] {
    background: transparent !important;
}

/* Caja interna del popover */
body [data-baseweb="popover"] > div {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
}

/* Menú */
body [data-baseweb="menu"] {
    background-color: #1e293b !important;
}

/* Listbox */
body ul[role="listbox"] {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

/* Opciones */
body li[role="option"],
body ul[role="listbox"] li,
body [role="listbox"] [role="option"] {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    font-family: var(--font_texto) !important;
}

/* Todo lo de dentro de cada opción */
body li[role="option"] *,
body ul[role="listbox"] li *,
body [role="listbox"] [role="option"] * {
    color: #f8fafc !important;
    fill: #f8fafc !important;
    font-family: var(--font_texto) !important;
}

/* Hover */
body li[role="option"]:hover,
body ul[role="listbox"] li:hover,
body [role="listbox"] [role="option"]:hover {
    background-color: #334155 !important;
    color: #ffffff !important;
}

/* Seleccionado */
body li[role="option"][aria-selected="true"],
body ul[role="listbox"] li[aria-selected="true"],
body [role="listbox"] [role="option"][aria-selected="true"] {
    background-color: #ff4b4b !important;
    color: #ffffff !important;
}

/* Texto mientras filtras en el dropdown */
body [role="combobox"],
body [role="combobox"] input,
body [role="combobox"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* =========================================================
   NUMBER INPUT
   ========================================================= */
.stNumberInput input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    font-family: var(--font_texto) !important;
}

/* Text input / text area */
.stTextInput input,
.stTextArea textarea {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    font-family: var(--font_texto) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #1e293b !important;
    border-radius: 10px !important;
}

/* Alerts */
.stAlert {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    font-family: var(--font_texto) !important;
}

/* Si quieres una zona concreta con otra fuente */
.zona_especial h1,
.zona_especial h2,
.zona_especial h3 {
    font-family: 'Bebas Neue', sans-serif !important;
}

.zona_especial p,
.zona_especial span,
.zona_especial div {
    font-family: 'Space Grotesk', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* TEXTO DE RESULTADOS (CLAVE) */
div[data-testid="stMarkdownContainer"] p {
    color: #f8fafc !important;
    font-size: 16px;
}

/* LISTAS (P1, P2, etc) */
div[data-testid="stMarkdownContainer"] li {
    color: #f8fafc !important;
    font-size: 16px;
}

/* TEXTO DENTRO DE st.write */
div[data-testid="stText"] {
    color: #f8fafc !important;
}

/* Ajuste para textos secundarios (tiempos) */
small, span {
    color: #cbd5e1 !important;
}

/* Resultado destacado (ganador) */
strong {
    color: #ffffff !important;
}
div[data-testid="stSelectbox"] {
    max-width: 260px !important;
}
h2 {
    font-size: 26px !important;
}
.frase-senna {
    font-family: 'Orbitron', sans-serif;
    font-size: 37px;
    text-align: center;
    color: #f5a627;
    font-style: italic;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)



def cargar_8_pilotos_aleatorios():
    rng = np.random.default_rng()

    # Pilotos con datos válidos tanto en race como en qualy
    driver_ids_race = set(df_race['DRIVER_ID'].dropna().unique())
    driver_ids_qualy = set(df_qualy['DRIVER_ID'].dropna().unique())
    driver_ids_validos = driver_ids_race.intersection(driver_ids_qualy)

    drivers_validos = df_drivers[df_drivers['DRIVER_ID'].isin(driver_ids_validos)].copy()
    pilotos_validos = sorted(drivers_validos['DRIVERREF'].dropna().unique().tolist())

    # Escuderías con datos válidos tanto en race como en qualy
    constructor_ids_race = set(df_race['CONSTRUCTOR_ID'].dropna().unique())
    constructor_ids_qualy = set(df_qualy['CONSTRUCTOR_ID'].dropna().unique())
    constructor_ids_validos = constructor_ids_race.intersection(constructor_ids_qualy)

    constructors_validos = df_constructors[df_constructors['CONSTRUCTOR_ID'].isin(constructor_ids_validos)].copy()
    escuderias_validas = sorted(constructors_validos['NAME#'].dropna().unique().tolist())

    if len(pilotos_validos) < 8:
        st.error("No hay suficientes pilotos válidos para generar una selección aleatoria.")
        return

    pilotos_elegidos = rng.choice(pilotos_validos, size=8, replace=False)

    for i, piloto in enumerate(pilotos_elegidos):
        # Guardar piloto en el widget
        st.session_state[f"piloto_select_{i}"] = piloto

        # Año válido del piloto (intersección race + qualy)
        driver_id = df_drivers.loc[df_drivers['DRIVERREF'] == piloto, 'DRIVER_ID'].iloc[0]

        anios_piloto_race = set(
            df_race.loc[df_race['DRIVER_ID'] == driver_id, 'RACE_DATE'].dt.year.dropna().astype(int).unique()
        )
        anios_piloto_qualy = set(
            df_qualy.loc[df_qualy['DRIVER_ID'] == driver_id, 'GP_DATE'].dt.year.dropna().astype(int).unique()
        )
        anios_piloto_validos = sorted(list(anios_piloto_race.intersection(anios_piloto_qualy)))

        if not anios_piloto_validos:
            continue

        anio_piloto = int(rng.choice(anios_piloto_validos))
        st.session_state[f"anio_piloto_select_{i}"] = anio_piloto

        # Escudería aleatoria válida
        constructor = rng.choice(escuderias_validas)
        st.session_state[f"constructor_select_{i}"] = constructor

        constructor_id = df_constructors.loc[
            df_constructors['NAME#'] == constructor, 'CONSTRUCTOR_ID'
        ].iloc[0]

        anios_constructor_race = set(
            df_race.loc[df_race['CONSTRUCTOR_ID'] == constructor_id, 'RACE_DATE'].dt.year.dropna().astype(int).unique()
        )
        anios_constructor_qualy = set(
            df_qualy.loc[df_qualy['CONSTRUCTOR_ID'] == constructor_id, 'GP_DATE'].dt.year.dropna().astype(int).unique()
        )
        anios_constructor_validos = sorted(list(anios_constructor_race.intersection(anios_constructor_qualy)))

        if not anios_constructor_validos:
            continue

        anio_constructor = int(rng.choice(anios_constructor_validos))
        st.session_state[f"anio_constructor_select_{i}"] = anio_constructor




def construir_pool_equilibrado():
    """
    Pool piloto-año con métricas agregadas.
    """
    rows = []

    driver_ids_race = set(df_race['DRIVER_ID'].dropna().unique())
    driver_ids_qualy = set(df_qualy['DRIVER_ID'].dropna().unique())
    driver_ids_validos = driver_ids_race.intersection(driver_ids_qualy)

    for driver_id in driver_ids_validos:
        df_driver_info = df_drivers[df_drivers['DRIVER_ID'] == driver_id]
        if df_driver_info.empty:
            continue

        piloto = df_driver_info['DRIVERREF'].iloc[0]

        years_race = set(
            df_race.loc[df_race['DRIVER_ID'] == driver_id, 'RACE_DATE']
            .dropna().dt.year.astype(int).unique()
        )
        years_qualy = set(
            df_qualy.loc[df_qualy['DRIVER_ID'] == driver_id, 'GP_DATE']
            .dropna().dt.year.astype(int).unique()
        )
        years_validos = sorted(list(years_race.intersection(years_qualy)))

        for anio in years_validos:
            race_year = df_race[
                (df_race['DRIVER_ID'] == driver_id) &
                (df_race['RACE_DATE'].dt.year == anio)
            ].sort_values("RACE_DATE")

            qualy_year = df_qualy[
                (df_qualy['DRIVER_ID'] == driver_id) &
                (df_qualy['GP_DATE'].dt.year == anio)
            ].sort_values("GP_DATE")

            if race_year.empty or qualy_year.empty:
                continue

            race_last = race_year.iloc[-1]
            qualy_last = qualy_year.iloc[-1]

            afinidad_race_validas = race_year['AFINIDAD3'].dropna()
            afinidad_qualy_validas = qualy_year['AFINIDAD_QUALY_3'].dropna()

            afinidad_lluvia_validas = race_year.loc[
                race_year['AFINIDAD_LLUVIA_REAL'].notna() &
                (race_year['AFINIDAD_LLUVIA_REAL'] > 0),
                'AFINIDAD_LLUVIA_REAL'
            ]

            rows.append({
                "piloto": piloto,
                "driver_id": driver_id,
                "anio_piloto": int(anio),
                "GENERAL_PERFORMANCE": float(race_last['GENERAL_PERFORMANCE']),
                "ADJUSTED_PERFORMANCE": float(race_last['ADJUSTED_PERFORMANCE']),
                "PERFORMANCE_RATIO": float(race_last['PERFORMANCE_RATIO']),
                "QUALY_PERFORMANCE": float(qualy_last['QUALY_PERFORMANCE']),
                "media_afinidad_race": float(afinidad_race_validas.mean()) if not afinidad_race_validas.empty else 0.0,
                "media_afinidad_qualy": float(afinidad_qualy_validas.mean()) if not afinidad_qualy_validas.empty else 0.0,
                "media_afinidad_lluvia": float(afinidad_lluvia_validas.mean()) if not afinidad_lluvia_validas.empty else 0.0,
                "n_circuitos_race": int(race_year['CIRCUIT_NAME'].nunique()),
                "n_circuitos_qualy": int(qualy_year['CIRCUIT_NAME'].nunique())
            })

    return pd.DataFrame(rows)


def construir_pool_constructores():
    """
    Pool constructor-año con métricas agregadas.
    """
    rows = []

    constructor_ids_race = set(df_race['CONSTRUCTOR_ID'].dropna().unique())
    constructor_ids_qualy = set(df_qualy['CONSTRUCTOR_ID'].dropna().unique())
    constructor_ids_validos = constructor_ids_race.intersection(constructor_ids_qualy)

    for constructor_id in constructor_ids_validos:
        df_ct_info = df_constructors[df_constructors['CONSTRUCTOR_ID'] == constructor_id]
        if df_ct_info.empty:
            continue

        constructor = df_ct_info['NAME#'].iloc[0]

        years_race = set(
            df_race.loc[df_race['CONSTRUCTOR_ID'] == constructor_id, 'RACE_DATE']
            .dropna().dt.year.astype(int).unique()
        )
        years_qualy = set(
            df_qualy.loc[df_qualy['CONSTRUCTOR_ID'] == constructor_id, 'GP_DATE']
            .dropna().dt.year.astype(int).unique()
        )
        years_validos = sorted(list(years_race.intersection(years_qualy)))

        for anio in years_validos:
            race_year = df_race[
                (df_race['CONSTRUCTOR_ID'] == constructor_id) &
                (df_race['RACE_DATE'].dt.year == anio)
            ].sort_values("RACE_DATE")

            qualy_year = df_qualy[
                (df_qualy['CONSTRUCTOR_ID'] == constructor_id) &
                (df_qualy['GP_DATE'].dt.year == anio)
            ].sort_values("GP_DATE")

            if race_year.empty or qualy_year.empty:
                continue

            race_last = race_year.iloc[-1]
            qualy_last = qualy_year.iloc[-1]

            rows.append({
                "constructor": constructor,
                "anio_constructor": int(anio),
                "CONSTRUCTOR_PERFORMANCE": float(race_last['CONSTRUCTOR_PERFORMANCE']),
                "CT_QUALY_PERFORMANCE": float(qualy_last['CT_QUALY_PERFORMANCE'])
            })

    return pd.DataFrame(rows)


def normalizar_columna(df, col):
    min_v = df[col].min()
    max_v = df[col].max()

    if max_v == min_v:
        return pd.Series([0.5] * len(df), index=df.index)

    return (df[col] - min_v) / (max_v - min_v)


def seleccionar_bloque_compacto(df, score_col, n=8, top_k=5):
    """
    Elige un bloque de n filas consecutivas en score ordenado con poca dispersión.
    Introduce algo de aleatoriedad escogiendo entre los mejores bloques.
    """
    rng = np.random.default_rng()

    df_sorted = df.sort_values(score_col).reset_index(drop=True)

    if len(df_sorted) <= n:
        return df_sorted.copy()

    candidatos = []

    for i in range(len(df_sorted) - n + 1):
        bloque = df_sorted.iloc[i:i+n].copy()
        spread = bloque[score_col].max() - bloque[score_col].min()
        std = bloque[score_col].std()
        score_compacidad = spread + std
        candidatos.append((score_compacidad, i, bloque))

    candidatos = sorted(candidatos, key=lambda x: x[0])
    top_k = min(top_k, len(candidatos))
    elegido = candidatos[rng.integers(0, top_k)][2]

    return elegido.copy()


def sample_tier(df, n_objetivo):
    if df.empty or n_objetivo <= 0:
        return df.iloc[0:0].copy()
    return df.sample(n=min(n_objetivo, len(df)), replace=False)


def cargar_8_pilotos_equilibrados():
    rng = np.random.default_rng()

    pool_pilotos = construir_pool_equilibrado()
    pool_ct = construir_pool_constructores()

    if pool_pilotos.empty or pool_ct.empty:
        st.error("No se ha podido construir el pool equilibrado.")
        return

    # =========================================================
    # 1) FILTRO PILOTOS: ÉLITE EQUILIBRADA
    # =========================================================
    pool_pilotos = pool_pilotos.copy()

    # Capar outliers absurdos
    pool_pilotos['PERFORMANCE_RATIO_CLIP'] = pool_pilotos['PERFORMANCE_RATIO'].clip(lower=0, upper=22)

    pool_pilotos = pool_pilotos[
        (pool_pilotos['n_circuitos_race'] >= 10) &
        (pool_pilotos['n_circuitos_qualy'] >= 10) &
        (pool_pilotos['GENERAL_PERFORMANCE'] >= 2.5) &
        (pool_pilotos['GENERAL_PERFORMANCE'] <= 6.5) &
        (pool_pilotos['ADJUSTED_PERFORMANCE'] >= 0.9) &
        (pool_pilotos['ADJUSTED_PERFORMANCE'] <= 1.7) &
        (pool_pilotos['PERFORMANCE_RATIO_CLIP'] >= 10) &
        (pool_pilotos['PERFORMANCE_RATIO_CLIP'] <= 22) &
        (pool_pilotos['QUALY_PERFORMANCE'] >= 30) &
        (pool_pilotos['QUALY_PERFORMANCE'] <= 110) &
        (pool_pilotos['media_afinidad_race'] >= 1.5) &
        (pool_pilotos['media_afinidad_qualy'] >= 2.0)
    ].copy()

    if len(pool_pilotos) < 8:
        st.error("No hay suficientes pilotos en la banda élite equilibrada.")
        return

    # Normalización piloto
    pool_pilotos['gp_norm'] = normalizar_columna(pool_pilotos, 'GENERAL_PERFORMANCE')
    pool_pilotos['adj_norm'] = normalizar_columna(pool_pilotos, 'ADJUSTED_PERFORMANCE')
    pool_pilotos['pr_norm'] = normalizar_columna(pool_pilotos, 'PERFORMANCE_RATIO_CLIP')
    pool_pilotos['qp_norm'] = normalizar_columna(pool_pilotos, 'QUALY_PERFORMANCE')
    pool_pilotos['af_race_norm'] = normalizar_columna(pool_pilotos, 'media_afinidad_race')
    pool_pilotos['af_qualy_norm'] = normalizar_columna(pool_pilotos, 'media_afinidad_qualy')

    pool_pilotos['score_piloto'] = (
        0.30 * pool_pilotos['gp_norm'] +
        0.20 * pool_pilotos['qp_norm'] +
        0.15 * pool_pilotos['adj_norm'] +
        0.15 * pool_pilotos['pr_norm'] +
        0.10 * pool_pilotos['af_race_norm'] +
        0.10 * pool_pilotos['af_qualy_norm']
    )

    # Quedarnos con 1 año por piloto: el más cercano al centro
    centro_piloto = pool_pilotos['score_piloto'].median()
    pool_pilotos['dist_centro_piloto'] = (pool_pilotos['score_piloto'] - centro_piloto).abs()

    pool_pilotos = (
        pool_pilotos
        .sort_values(['dist_centro_piloto', 'score_piloto'])
        .drop_duplicates(subset=['piloto'], keep='first')
        .copy()
    )

    if len(pool_pilotos) < 8:
        st.error("No hay suficientes pilotos distintos tras filtrar.")
        return

    # =========================================================
    # 2) FILTRO COCHES: BUENOS PERO NO ROTOS
    # =========================================================
    pool_ct = pool_ct.copy()

    pool_ct = pool_ct[
        (pool_ct['CONSTRUCTOR_PERFORMANCE'] >= 8) &
        (pool_ct['CONSTRUCTOR_PERFORMANCE'] <= 22) &
        (pool_ct['CT_QUALY_PERFORMANCE'] >= 50) &
        (pool_ct['CT_QUALY_PERFORMANCE'] <= 170)
    ].copy()

    if len(pool_ct) < 8:
        st.error("No hay suficientes coches en la banda buena equilibrada.")
        return

    # Normalización coche
    pool_ct['ct_race_norm'] = normalizar_columna(pool_ct, 'CONSTRUCTOR_PERFORMANCE')
    pool_ct['ct_qualy_norm'] = normalizar_columna(pool_ct, 'CT_QUALY_PERFORMANCE')

    pool_ct['score_coche'] = (
        0.60 * pool_ct['ct_race_norm'] +
        0.40 * pool_ct['ct_qualy_norm']
    )

    # Quedarnos con 1 año por constructor: el más cercano al centro
    centro_coche = pool_ct['score_coche'].median()
    pool_ct['dist_centro_coche'] = (pool_ct['score_coche'] - centro_coche).abs()

    pool_ct = (
        pool_ct
        .sort_values(['dist_centro_coche', 'score_coche'])
        .drop_duplicates(subset=['constructor'], keep='first')
        .copy()
    )

    if len(pool_ct) < 8:
        st.error("No hay suficientes constructores distintos tras filtrar.")
        return

    # =========================================================
    # 3) SCORE DE COMBINACIÓN PILOTO + COCHE
    # =========================================================
    combinaciones = []

    for _, p_row in pool_pilotos.iterrows():
        for _, c_row in pool_ct.iterrows():
            # Score base combinación
            score_combo = (
                0.65 * p_row['score_piloto'] +
                0.35 * c_row['score_coche']
            )

            # Penalización si ambos están demasiado arriba a la vez
            exceso_top = max(0, p_row['score_piloto'] - 0.75) + max(0, c_row['score_coche'] - 0.75)
            penalizacion_top = 0.18 * exceso_top

            # Penalización si ambos están demasiado abajo a la vez
            defecto = max(0, 0.45 - p_row['score_piloto']) + max(0, 0.45 - c_row['score_coche'])
            penalizacion_bajo = 0.10 * defecto

            # Penalización por desequilibrio exagerado entre piloto y coche
            gap_pc = abs(p_row['score_piloto'] - c_row['score_coche'])
            penalizacion_gap = 0.12 * max(0, gap_pc - 0.22)

            score_final = score_combo - penalizacion_top - penalizacion_bajo - penalizacion_gap

            combinaciones.append({
                "piloto": p_row['piloto'],
                "anio_piloto": int(p_row['anio_piloto']),
                "score_piloto": float(p_row['score_piloto']),
                "constructor": c_row['constructor'],
                "anio_constructor": int(c_row['anio_constructor']),
                "score_coche": float(c_row['score_coche']),
                "score_combo": float(score_final)
            })

    df_combo = pd.DataFrame(combinaciones)

    if df_combo.empty:
        st.error("No se pudieron construir combinaciones piloto-coche.")
        return

    # =========================================================
    # 4) BANDA EQUILIBRADA DE COMBINACIONES
    # =========================================================
    # Queremos combinaciones fuertes pero compactas
    df_combo = df_combo[
        (df_combo['score_combo'] >= 0.52) &
        (df_combo['score_combo'] <= 0.68)
    ].copy()

    if len(df_combo) < 8:
        # Si quedó demasiado estrecho, abrimos un poco
        df_combo = pd.DataFrame(combinaciones)
        df_combo = df_combo[
            (df_combo['score_combo'] >= 0.50) &
            (df_combo['score_combo'] <= 0.72)
        ].copy()

    if len(df_combo) < 8:
        st.error("No hay suficientes combinaciones equilibradas tras filtrar.")
        return

    # Ordenar por cercanía al centro de equilibrio
    centro_combo = df_combo['score_combo'].median()
    df_combo['dist_centro_combo'] = (df_combo['score_combo'] - centro_combo).abs()

    # Penalizar ligeramente pilotos que aparecen demasiadas veces en combinaciones válidas
    freq_piloto = df_combo['piloto'].value_counts().to_dict()
    max_freq = max(freq_piloto.values()) if freq_piloto else 1

    df_combo['penalizacion_repeticion_piloto'] = df_combo['piloto'].map(
        lambda p: 0.025 * (freq_piloto.get(p, 1) / max_freq)
    )

    # Score ajustado: mantenemos equilibrio, pero favorecemos variedad de nombres
    df_combo['score_variedad'] = (
        df_combo['dist_centro_combo'] + df_combo['penalizacion_repeticion_piloto']
    )

    df_combo = df_combo.sort_values(['score_variedad', 'score_combo']).reset_index(drop=True)

    # =========================================================
    # 5) ELEGIR 8 BINOMIOS SIN REPETIR PILOTO NI CONSTRUCTOR
    # =========================================================
    seleccion_final = []
    pilotos_usados = set()
    constructores_usados = set()

    top_n = min(90, len(df_combo))
    candidatos = df_combo.iloc[:top_n].copy()

    # Evitar que un mismo piloto domine demasiado el pool final de candidatos
    candidatos = (
        candidatos
        .groupby('piloto', group_keys=False)
        .apply(lambda g: g.sample(n=min(3, len(g)), random_state=int(rng.integers(0, 1_000_000))))
        .reset_index(drop=True)
    )

    candidatos = candidatos.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

    for _, row in candidatos.iterrows():
        if row['piloto'] in pilotos_usados:
            continue
        if row['constructor'] in constructores_usados:
            continue

        seleccion_final.append(row)
        pilotos_usados.add(row['piloto'])
        constructores_usados.add(row['constructor'])

        if len(seleccion_final) == 8:
            break

    # Fallback si no salió suficiente
    if len(seleccion_final) < 8:
        candidatos = df_combo.reset_index(drop=True)
        for _, row in candidatos.iterrows():
            if row['piloto'] in pilotos_usados:
                continue
            if row['constructor'] in constructores_usados:
                continue

            seleccion_final.append(row)
            pilotos_usados.add(row['piloto'])
            constructores_usados.add(row['constructor'])

            if len(seleccion_final) == 8:
                break

    if len(seleccion_final) < 8:
        st.error("No se pudieron seleccionar 8 combinaciones únicas piloto-coche.")
        return

    seleccion_final = pd.DataFrame(seleccion_final).reset_index(drop=True)

    # Orden opcional: del más fuerte al menos fuerte dentro del bloque equilibrado
    seleccion_final = seleccion_final.sort_values('score_combo', ascending=False).reset_index(drop=True)

    # =========================================================
    # 6) VOLCAR A SESSION_STATE
    # =========================================================
    for i in range(8):
        row = seleccion_final.iloc[i]

        st.session_state[f"piloto_select_{i}"] = row['piloto']
        st.session_state[f"anio_piloto_select_{i}"] = int(row['anio_piloto'])

        st.session_state[f"constructor_select_{i}"] = row['constructor']
        st.session_state[f"anio_constructor_select_{i}"] = int(row['anio_constructor'])

    st.success("Mundial equilibrado generado con equilibrio real de combinación piloto-coche.")



def cargar_10_circuitos_aleatorios():
    rng = np.random.default_rng()

    circuitos_validos = sorted(df_race['CIRCUIT_NAME'].dropna().unique().tolist())

    if len(circuitos_validos) < 10:
        st.error("No hay suficientes circuitos válidos para generar un calendario aleatorio.")
        return

    circuitos_elegidos = rng.choice(circuitos_validos, size=10, replace=False)

    for i, circuito in enumerate(circuitos_elegidos):
        st.session_state[f"circuito_{i}"] = circuito
        st.session_state[f"lluvia_{i}"] = int(rng.choice([0, 1]))




# --- Interfaz de usuario ---
st.title("🏁 Simulador Histórico Mini-Mundial F1")








st.markdown("""
<p class="frase-senna">
If you no longer go for a gap that exists, you are no longer a racing driver
</p>
""", unsafe_allow_html=True)

driverrefs = sorted(df_drivers['DRIVERREF'].unique())
constructorrefs = sorted(df_constructors['NAME#'].dropna().unique())

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

st.header("1. Selección de pilotos y escuderías")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div style="display:flex; align-items:center; gap:10px;">', unsafe_allow_html=True)

    if st.button("🎲"):
        cargar_8_pilotos_aleatorios()
        st.rerun()

    st.markdown('<span style="font-size:14px; color:#cbd5e1;">Si te da pereza seleccionar pilotos o no tienes preferencias, pulsa aquí para una selección completamente aleatoria</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div style="display:flex; align-items:center; gap:10px;">', unsafe_allow_html=True)

    if st.button("⚖️"):
        cargar_8_pilotos_equilibrados()
        st.rerun()

    st.markdown('<span style="font-size:14px; color:#cbd5e1;">Esta selección también es aleatoria, es la opción RECOMENDADA si quieres un mundial equilibrado en el que cualquiera pueda ganar</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

driverrefs = sorted(df_drivers['DRIVERREF'].dropna().unique())
constructorrefs = sorted(df_constructors['NAME#'].dropna().unique())


st.session_state.pilotos_config = {}

for i in range(8):
    st.markdown(f"### Piloto {i+1}")

    # PILOTO y AÑO
    col_piloto, col_gap, col_anio, col_rest = st.columns([1.35, 0.05, 0.78, 2.10])

    with col_piloto:
        piloto = st.selectbox(f"Piloto {i+1}", driverrefs, key=f"piloto_select_{i}")

    with col_anio:
        driver_id = df_drivers[df_drivers['DRIVERREF'] == piloto]['DRIVER_ID'].iloc[0]

        years_piloto_race = set(
            df_race[df_race['DRIVER_ID'] == driver_id]['RACE_DATE'].dt.year.dropna().astype(int).unique()
        )
        years_piloto_qualy = set(
            df_qualy[df_qualy['DRIVER_ID'] == driver_id]['GP_DATE'].dt.year.dropna().astype(int).unique()
        )
        years_piloto = sorted(list(years_piloto_race.intersection(years_piloto_qualy)))

        if not years_piloto:
            years_piloto = [2012]

        anio_piloto = st.selectbox(f"Año piloto {i+1}", years_piloto, key=f"anio_piloto_select_{i}")

    # ESCUDERÍA y AÑO
    col_esc, col_gap2, col_anio_esc, col_rest2 = st.columns([1.35, 0.05, 0.78, 2.10])

    with col_esc:
        constructor = st.selectbox(f"Escudería {i+1}", constructorrefs, key=f"constructor_select_{i}")

    with col_anio_esc:
        constructor_id = df_constructors[df_constructors['NAME#'] == constructor]['CONSTRUCTOR_ID'].iloc[0]

        years_constructor_race = set(
            df_race[df_race['CONSTRUCTOR_ID'] == constructor_id]['RACE_DATE'].dt.year.dropna().astype(int).unique()
        )
        years_constructor_qualy = set(
            df_qualy[df_qualy['CONSTRUCTOR_ID'] == constructor_id]['GP_DATE'].dt.year.dropna().astype(int).unique()
        )
        years_constructor = sorted(list(years_constructor_race.intersection(years_constructor_qualy)))

        if not years_constructor:
            years_constructor = [2012]

        anio_constructor = st.selectbox(f"Año escudería {i+1}", years_constructor, key=f"anio_constructor_select_{i}")

    st.session_state.pilotos_config[piloto] = {
        "anio_piloto": int(anio_piloto),
        "constructor": constructor,
        "anio_constructor": int(anio_constructor)
    }

    st.markdown("---")




st.header("2. Define los 10 circuitos y si llueve")

col_cal1, col_cal2 = st.columns([1, 4])

with col_cal1:
    if st.button("🎲 Circuitos y lluvia aleatorios"):
        cargar_10_circuitos_aleatorios()
        st.rerun()

with col_cal2:
    st.caption("Rellena automáticamente los 10 circuitos y asigna lluvia aleatoria a cada carrera.")

circuitos = sorted(df_race['CIRCUIT_NAME'].dropna().unique())
calendario_temporal = []

for i in range(10):
    st.markdown(f"### Carrera {i+1}")

    col_circuito, col_gap, col_lluvia, col_rest = st.columns([1.35, 0.04, 0.60, 2.01])

    with col_circuito:
        circuito = st.selectbox(
            f"Circuito {i+1}",
            circuitos,
            key=f"circuito_{i}"
        )

    with col_lluvia:
        lluvia = st.radio(
            f"¿Llueve?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
            horizontal=True,
            key=f"lluvia_{i}"
        )

    calendario_temporal.append((circuito, lluvia))
    st.markdown("---")


col_guardar, col_reset = st.columns([3, 1])

with col_guardar:
    guardar_calendario = st.button("📅 Guardar calendario")

with col_reset:
    resetear_mundial = st.button("🔄 Resetear Mundial")

if guardar_calendario:
    st.session_state.calendario = calendario_temporal
    st.success("Calendario guardado correctamente")

if resetear_mundial:
    st.session_state.pilotos_config = {}
    st.session_state.calendario = []
    st.session_state.resultados_mundial = {}
    st.session_state.carrera_actual = 0
    st.session_state.poles = {}
    st.session_state.victorias = {}
    st.session_state.historial_carreras = []  # 🔁 limpia prompts anteriores
    st.session_state.clasificacion_mundial = {}
    st.success("Estado reiniciado correctamente")

if 'modo_seleccion' not in st.session_state:
    st.session_state.modo_seleccion = "Manual"

if 'seed_aleatoria' not in st.session_state:
    st.session_state.seed_aleatoria = 42
    
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

# No mostrar tabla si aún no se ha simulado ninguna carrera
if st.session_state.carrera_actual == 0 or not st.session_state.resultados_mundial:
    st.info("ℹ️ El resumen del mundial aparecerá cuando se simule la primera carrera.")
else:
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

    if resumen_mundial:
        st.dataframe(pd.DataFrame(resumen_mundial).sort_values("Puntos", ascending=False))
    else:
        st.info("ℹ️ Aún no hay resultados para mostrar.")





if st.session_state.carrera_actual >= len(st.session_state.calendario):
    st.success("🏁 Mundial finalizado")

    clasificacion_final = sorted(
        st.session_state.resultados_mundial.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if clasificacion_final:  # Solo si hay pilotos clasificados
        campeon = clasificacion_final[0][0]
        campeon_config = st.session_state.pilotos_config.get(campeon, {})
        escuderia = campeon_config.get("constructor", "unknown")

        st.markdown(f"🎉 **¡El campeón del mundial es {campeon.replace('_', ' ').title()} con {escuderia.title()}!** 🏆")

        import os
        from PIL import Image

        ruta_piloto = os.path.join("images", f"{campeon}.jpg")
        ruta_logo = os.path.join("logos", f"{escuderia}.jpg")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(ruta_piloto):
                st.image(Image.open(ruta_piloto), caption=campeon.replace('_', ' ').title(), width=300)
            else:
                st.info("📷 Imagen del campeón no disponible.")

        with col2:
            if os.path.exists(ruta_logo):
                st.image(Image.open(ruta_logo), caption=escuderia.title(), width=400)
            else:
                st.info("🏎️ Logo de la escudería no disponible.")












