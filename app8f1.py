import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Cargar datos y modelo ---
df = pd.read_csv("df.csv", parse_dates=["RACE_DATE"])
df_drivers = pd.read_csv("df_drivers.csv")
df_constructors = pd.read_csv("df_constructors.csv")

with open("formula1_top8_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Estado de la aplicación ---
if 'tabla_mundial' not in st.session_state:
    st.session_state.tabla_mundial = {}
if 'historial_carreras' not in st.session_state:
    st.session_state.historial_carreras = []
if 'resultados_previos' not in st.session_state:
    st.session_state.resultados_previos = {}

# --- Funciones ---
def generar_features_para_pilotos(pilotos_seleccionados, grids_por_piloto, constructores_por_piloto, constructor_years_por_piloto, circuit_name, lluvia):
    rows = []
    for piloto in pilotos_seleccionados:
        try:
            driver_name, year = piloto.rsplit(' ', 1)
            year = int(year)
            grid = grids_por_piloto.get(piloto, 5)
            constructor_name = constructores_por_piloto.get(piloto, "")
            constructor_year = constructor_years_por_piloto.get(piloto, 2012)

            driver_id = df_drivers.loc[df_drivers['DRIVERREF'] == driver_name, 'DRIVER_ID'].iloc[0]

            constructor_id = df[(df['RACE_DATE'].dt.year == constructor_year) &
                (df['CONSTRUCTOR_ID'].isin(
                    df_constructors[df_constructors['NAME#'].str.lower() == constructor_name.lower()]['CONSTRUCTOR_ID']
                ))]['CONSTRUCTOR_ID'].iloc[0]

            driver_rows = df[(df['DRIVER_ID'] == driver_id) & (df['RACE_DATE'].dt.year == year)].sort_values(by="RACE_DATE")
            if driver_rows.empty:
                raise ValueError("No hay datos para el piloto en ese año.")
            driver_data = driver_rows.iloc[-1]

            constructor_rows = df[(df['CONSTRUCTOR_ID'] == constructor_id) & (df['RACE_DATE'].dt.year == constructor_year)].sort_values(by="RACE_DATE")
            if constructor_rows.empty:
                raise ValueError("No hay datos para el constructor en ese año.")
            constructor_data = constructor_rows.iloc[-1]

            afinidad_series = df[(df['DRIVER_ID'] == driver_id) & (df['CIRCUIT_NAME'].str.lower() == circuit_name.lower())]['AFINIDAD']
            afinidad = afinidad_series.iloc[0] if not afinidad_series.empty else 0.5

            prev_pos = st.session_state.resultados_previos.get(piloto, 10)

            rows.append({
                'PILOTO': piloto,
                'GRID': grid,
                'PREV_POSITIONS': prev_pos,
                'ADJUSTED_PERFORMANCE': driver_data['ADJUSTED_PERFORMANCE'],
                'GENERAL_PERFORMANCE': driver_data['GENERAL_PERFORMANCE'],
                'CONSTRUCTOR_PERFORMANCE': constructor_data['CONSTRUCTOR_PERFORMANCE'],
                'AFINIDAD': afinidad,
                'LLUVIA': lluvia
            })
        except Exception as e:
            print(f"❌ Error en {piloto}: {e}")
            continue

    return pd.DataFrame(rows)

def simular_carrera_8_pilotos(pilotos, model, grids_por_piloto, constructores_por_piloto, constructor_years_por_piloto, circuit_name, lluvia):
    df_features = generar_features_para_pilotos(pilotos, grids_por_piloto, constructores_por_piloto, constructor_years_por_piloto, circuit_name, lluvia)
    if df_features.empty:
        return None

    X = df_features.drop(columns="PILOTO")
    probs = model.predict_proba(X)
    ponderaciones = [0, 25, 18, 15, 12, 10, 8, 6, 4]  # clase 0 = fuera top 8
    resultados = []
    for i, piloto in enumerate(df_features["PILOTO"]):
        puntuacion = sum(ponderaciones[j] * probs[i][j] for j in range(len(ponderaciones)))
        resultados.append((puntuacion, piloto, probs[i]))
    resultados.sort(reverse=True)
    return resultados

def simular_carrera_y_actualizar(pilotos, model, circuito, lluvia, grids_por_piloto, constructores_por_piloto, constructor_years_por_piloto):
    resultados = simular_carrera_8_pilotos(pilotos, model, grids_por_piloto, constructores_por_piloto, constructor_years_por_piloto, circuito, lluvia)
    if resultados is None:
        return None

    puntos_f1 = [10, 8, 6, 5, 4, 3, 2, 1]
    for i, (_, piloto, _) in enumerate(resultados):
        puntos = puntos_f1[i]
        st.session_state.tabla_mundial[piloto] = st.session_state.tabla_mundial.get(piloto, 0) + puntos
        st.session_state.resultados_previos[piloto] = i + 1

    st.session_state.historial_carreras.append((circuito, lluvia, resultados))
    return resultados

# --- Interfaz Streamlit ---
st.title("🏆 Simulador de Mundial F1 (8 pilotos / 10 carreras)")

st.markdown("---")
st.subheader(f"Carrera {len(st.session_state.historial_carreras) + 1}")

# --- Selección de pilotos dinámicos ---
driverrefs = sorted(df_drivers['DRIVERREF'].unique())
pilotos_seleccionados = []
grids_por_piloto = {}
constructores_por_piloto = {}
constructor_years_por_piloto = {}
constructor_names = sorted(df_constructors['NAME#'].unique())

for i in range(8):
    st.markdown(f"### Piloto {i + 1}")
    piloto = st.selectbox(f"Selecciona piloto {i + 1}", driverrefs, key=f"piloto_{i}")

    try:
        driver_id = df_drivers[df_drivers['DRIVERREF'] == piloto]['DRIVER_ID'].iloc[0]
        years_piloto = sorted(df[df['DRIVER_ID'] == driver_id]['RACE_DATE'].dt.year.unique())
        years_piloto = [y for y in years_piloto if y <= 2023]
    except:
        years_piloto = [2012]

    year = st.selectbox(f"Año del piloto {piloto}", years_piloto, index=len(years_piloto) - 1, key=f"year_piloto_{i}")
    piloto_completo = f"{piloto} {year}"
    pilotos_seleccionados.append(piloto_completo)

    grids_por_piloto[piloto_completo] = st.number_input(f"Grid de salida para {piloto_completo}", min_value=1, max_value=20, value=5, step=1, key=f"grid_{i}")

    # Escuderías del piloto en ese año
    try:
        constructor_ids = df[
            (df['DRIVER_ID'] == driver_id) &
            (df['RACE_DATE'].dt.year == year)
        ]['CONSTRUCTOR_ID'].unique()
        constructor_options = df_constructors[df_constructors['CONSTRUCTOR_ID'].isin(constructor_ids)]['NAME#'].unique()
        constructor_options = sorted(constructor_options)
    except:
        constructor_options = sorted(df_constructors['NAME#'].unique())

    constructor_name = st.selectbox(f"Escudería para {piloto_completo}", constructor_options, key=f"constructor_{i}")
    constructores_por_piloto[piloto_completo] = constructor_name

    # Filtrar años en los que esa escudería participó
    try:
        constructor_id = df_constructors[df_constructors['NAME#'].str.lower() == constructor_name.lower()]['CONSTRUCTOR_ID'].iloc[0]
        years_constructor = sorted(df[df['CONSTRUCTOR_ID'] == constructor_id]['RACE_DATE'].dt.year.unique())
        years_constructor = [y for y in years_constructor if y <= 2023]
    except:
        years_constructor = [2012]

    constructor_years_por_piloto[piloto_completo] = st.selectbox(f"Año escudería para {piloto_completo}", years_constructor, index=len(years_constructor) - 1, key=f"year_constructor_{i}")


# --- Selección de circuito y clima ---
circuit_names = sorted(df['CIRCUIT_NAME'].dropna().unique())
circuito = st.selectbox("Selecciona el circuito para esta carrera:", circuit_names)
lluvia = st.radio("¿Llueve en esta carrera?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")

# --- Botón de simulación ---
if st.button("Simular esta carrera"):
    resultado = simular_carrera_y_actualizar(
        pilotos=pilotos_seleccionados,
        model=model,
        circuito=circuito,
        lluvia=lluvia,
        grids_por_piloto=grids_por_piloto,
        constructores_por_piloto=constructores_por_piloto,
        constructor_years_por_piloto=constructor_years_por_piloto
    )

    if resultado:
        st.success("Carrera simulada correctamente")
        st.markdown("### Clasificación de esta carrera:")
        primer_puntaje = resultado[0][0]
        for i, (puntos_prob, piloto, proba) in enumerate(resultado, start=1):
            if i == 1:
                st.write(f"**P{i}**: {piloto} — 🏁 Ganador")
            else:
                diferencia = primer_puntaje - puntos_prob
                tiempo = diferencia * 5  # 1 punto = 5 segundos
                st.write(f"**P{i}**: {piloto} — +{tiempo:.1f}s")
    else:
        st.error("No se pudo simular la carrera.")

# --- Clasificación acumulada ---
st.markdown("### 🏁 Clasificación General del Mundial")
tabla = sorted(st.session_state.tabla_mundial.items(), key=lambda x: x[1], reverse=True)
for i, (piloto, puntos) in enumerate(tabla, start=1):
    st.write(f"{i}. {piloto} — {puntos} puntos")

# --- Fin del mundial ---
if len(st.session_state.historial_carreras) == 10:
    st.success("🏁 ¡El mundial ha terminado!")
