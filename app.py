import json
import os
import urllib.parse
import joblib
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="Predictor Top Billboard", page_icon="🎵", layout="centered")

st.title("🎵 Predictor de Entrada al Top Billboard")
st.write("Ingresa un artista y una canción para consultar sus características musicales y predecir si entrará al top de Billboard.")

api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif os.environ.get("GEMINI_API_KEY"):
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("Configura `GEMINI_API_KEY` en los Secrets de Streamlit o en tu entorno para usar el agente.")
    st.stop()

client = genai.Client(api_key=api_key)


class InfoCancion(BaseModel):
    durationSeg: float = Field(description="Duración aproximada de la canción en segundos")
    anioNacimiento: float = Field(description="Año de nacimiento del artista o año de formación del grupo")
    mood: str = Field(description="Estado de ánimo predominante de la canción")
    tempo: str = Field(description="Tempo de la canción, por ejemplo: Fast Tempo, Medium Tempo, Slow Tempo")
    genre: str = Field(description="Género musical principal")
    artist_type: str = Field(description="Tipo de artista: Male, Female, Mixed o Duo/Group")


@st.cache_resource
def cargar_recursos():
    modelo = joblib.load("modelo_random_forest.joblib")
    columnas = joblib.load("columnas_modelo.joblib")
    return modelo, columnas


def consultar_cancion(artista: str, cancion: str) -> dict:
    prompt = f"""
    Busca información sobre la canción '{cancion}' del artista '{artista}'.
    Devuelve los datos más precisos posibles para:
    - durationSeg (duración en segundos)
    - anioNacimiento (año de nacimiento del artista o formación de la banda)
    - mood (estado de ánimo en inglés, por ejemplo: Brooding, Energizing, Melancholy, Cool, Romantic)
    - tempo (por ejemplo: Fast Tempo, Medium Tempo, Slow Tempo)
    - genre (género musical)
    - artist_type (Male, Female, Mixed o Duo/Group)
    """
    response = client.models.generate_content(
       model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            response_mime_type="application/json",
            response_schema=InfoCancion,
            temperature=0.1,
        ),
    )
    return json.loads(response.text)


def predecir_top(datos: dict, artista: str, cancion: str, modelo, columnas_modelo):
    df_nuevo = pd.DataFrame([{
        "chart_date": 20140628,
        "durationSeg": float(datos.get("durationSeg", 210.0)),
        "anioNacimiento": float(datos.get("anioNacimiento", 1990.0)),
        "mood": str(datos.get("mood", "Brooding")),
        "tempo": str(datos.get("tempo", "Medium Tempo")),
        "genre": str(datos.get("genre", "Pop")),
        "artist_type": str(datos.get("artist_type", "Male")),
        "artist": artista,
        "title": cancion,
    }])

    df_encoded = pd.get_dummies(
        df_nuevo,
        columns=["mood", "tempo", "genre", "artist_type", "artist", "title"]
    )
    df_alineado = df_encoded.reindex(columns=columnas_modelo, fill_value=False)
    prediccion = modelo.predict(df_alineado)[0]
    return int(prediccion)


with st.form("form_cancion"):
    artista = st.text_input("Nombre del artista", placeholder="Ej. Coldplay")
    cancion = st.text_input("Nombre de la canción", placeholder="Ej. Yellow")
    submitted = st.form_submit_button("Predecir")

if submitted:
    if not artista.strip() or not cancion.strip():
        st.error("Por favor escribe tanto el artista como la canción.")
    else:
        try:
            modelo, columnas_modelo = cargar_recursos()
            with st.spinner("Consultando información de la canción y ejecutando el modelo..."):
                datos_cancion = consultar_cancion(artista.strip(), cancion.strip())
                resultado = predecir_top(
                    datos_cancion,
                    artista.strip(),
                    cancion.strip(),
                    modelo,
                    columnas_modelo
                )

            st.subheader("Resultado de la predicción")
            if resultado == 1:
                st.success(f"La canción **'{cancion.strip()}'** de **{artista.strip()}** **SÍ** entraría al top de Billboard.")
            else:
                st.info(f"La canción **'{cancion.strip()}'** de **{artista.strip()}** **NO** entraría al top de Billboard.")

            st.subheader("Datos consultados de la canción")
            st.json(datos_cancion)

            st.subheader("Preview de la canción")
            query = urllib.parse.quote(f"{artista.strip()} {cancion.strip()} official audio")
            youtube_search_url = f"https://www.youtube.com/results?search_query={query}"
            spotify_search_url = f"https://open.spotify.com/search/{urllib.parse.quote(f'{artista.strip()} {cancion.strip()}')}"

            st.markdown(f"**{cancion.strip()}** — *{artista.strip()}*")
            st.markdown(f"- [Buscar y escuchar en YouTube]({youtube_search_url})")
            st.markdown(f"- [Buscar en Spotify]({spotify_search_url})")

        except Exception as e:
            st.error(f"Ocurrió un error al procesar la solicitud: {e}")
