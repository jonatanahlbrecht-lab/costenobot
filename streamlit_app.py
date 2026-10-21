import os
import time

import streamlit as st
from google import genai
from google.genai import types

MODELO = "gemini-3.8-flash"
MAX_MENSAJES = 20  # Cuántos mensajes recuerda el bot

SYSTEM_PROMPT = """
Eres CosteñoBot, un barranquillero de pura cepa, criao en el Caribe colombiano.
Hablas golpeao, rápido, charlatán y gracioso.
Te comes las eses y las des: "pa'", "to'", "e'", "nojodá", "pue'".
Muletillas: ajá, nojoda, eche, erda, mondá, vaina, mi llave, compa, mi vale, burda de, qué e' lo que.
Referencias costeñas: la fría, el carnaval, la butifarra, el Junior, el calor, la terraza.
Respuestas CORTAS (2 a 4 frases), como en la terraza tomando una fría.
Bromeas con cariño, nunca ofendes en serio. La info que das tiene que ser correcta.
No hables como robot ni uses listas.
"""

st.set_page_config(page_title="CosteñoBot", page_icon="🌴")

# La key NO va escrita en el código: se lee de los "secrets" (web) o de una variable (local)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Falta la GEMINI_API_KEY. Agrégala en los Secrets de la app.")
    st.stop()


@st.cache_resource
def get_client(key):
    return genai.Client(api_key=key)


client = get_client(api_key)


def generar(mensajes):
    """Llama a Gemini con la conversación y reintenta si hay mucha demanda."""
    contenido = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in mensajes[-MAX_MENSAJES:]
    ]
    for intento in range(4):
        try:
            r = client.models.generate_content(
                model=MODELO,
                contents=contenido,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=1.0,
                    max_output_tokens=800,
                ),
            )
            return r.text or "Ajá, me quedé en blanco, mi llave. Pregúntame otra vez."
        except Exception as e:
            error = str(e)
            if "503" in error or "UNAVAILABLE" in error:
                time.sleep(3 * (intento + 1))
                continue
            if "429" in error:
                return "Eche, me tienen hablando mucho. Espérate tantico y me preguntas de nuevo."
            return "Erda, se dañó esta vaina. Inténtalo otra vez en un ratico."
    return "Nojoda mi llave, Google tiene el servicio full de gente. Pregúntame en un ratico."


st.title("CosteñoBot 🌴")
st.caption("Un barranquillero pa' charlar un rato. Escríbele, mi vale.")

# Memoria de la conversación (una por cada persona que abre la página)
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

with st.sidebar:
    if st.button("Olvidar la charla"):
        st.session_state.mensajes = []
        st.rerun()

for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if pregunta := st.chat_input("Escríbele al costeño..."):
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando, mi llave..."):
            respuesta = generar(st.session_state.mensajes)
        st.markdown(respuesta)
    st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
