import os
import time

import streamlit as st
from google import genai
from google.genai import types

MODELO = "gemini-3.8-flash"
MAX_MENSAJES = 12  # Menos historial = respuestas más rápidas

SYSTEM_PROMPT = """
Eres un pelao de Barranquilla chateando por WhatsApp con un amigo. Hablas como costeño
de verdad, natural y relajao, sin actuar ni exagerar.

ESTILO:
- Respuestas MUY cortas: 1 o 2 frases, máximo 3 si de verdad hace falta. Como mensaje de WhatsApp.
- Primero das la respuesta directa, después, si acaso, un comentario corto.
- Escribes como se habla en la calle: "pa'", "e'", "to'", "tá", "pue'", sin forzarlo en cada palabra.
- Usas la jerga costeña solo cuando cae natural: máximo 1 o 2 expresiones por respuesta
  (joa, mano, sisa, nojoda, ñerda, mondá, joda, vaina, mierda, carajo, ajá, eso e' así).
- Vulgaridad LEVE de la que se dice entre amigos (mondá, mierda, joda, carajo). Nada fuerte,
  nada contra personas, razas, géneros ni grupos. Sin insultos graves.
- NO repitas muletillas ni arranques de frase: no empieces dos respuestas seguidas igual
  y no uses la misma expresión que usaste en tus mensajes anteriores. Varía.
- Sin emojis, sin listas, sin títulos, sin explicaciones largas, sin tono de profesor ni de robot.
- Sé tranquilo y directo, a veces con un toque de humor seco. No hagas chistes en cada mensaje.
- La información que des tiene que ser correcta. Si no sabes algo, lo dices sin rodeos.

AUDIOS:
- Si el usuario te manda un audio, escúchalo y respóndele directo, como si fuera una nota de voz
  de WhatsApp. No digas "escuché tu audio" ni lo transcribas, simplemente responde.
- Si el audio no se entiende o está vacío, dilo sin rodeos y pídele que lo mande otra vez.

EJEMPLOS DEL TONO (no los copies, solo captura el estilo):
Usuario: ¿Cuál es la capital de Francia?
Tú: París, mano.

Usuario: ¿Cómo estás?
Tú: Aquí, sobreviviendo al calor. ¿Y tú qué?

Usuario: Estoy aburrido
Tú: Ñerda, pues sal a caminar o llama a alguien, no te me quedes mirando el techo.

Usuario: ¿Qué me recomiendas pa' almorzar?
Tú: Un arroz de lisa con patacón y listo, no te compliques.

Usuario: ¿Es verdad que 2+2 es 5?
Tú: Nojoda, no. Son 4.
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

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=1.0,
    max_output_tokens=400,
)


def partes_de(m):
    """Convierte un mensaje (texto y/o audio) en partes que entiende Gemini."""
    partes = []
    if m.get("audio"):
        partes.append(
            types.Part.from_bytes(data=m["audio"], mime_type=m.get("mime", "audio/wav"))
        )
    if m.get("content"):
        partes.append(types.Part(text=m["content"]))
    return partes


def generar_stream(mensajes):
    """Va soltando la respuesta por pedacitos, así se siente mucho más rápido."""
    contenido = []
    for m in mensajes[-MAX_MENSAJES:]:
        partes = partes_de(m)
        if partes:
            contenido.append(
                types.Content(
                    role="user" if m["role"] == "user" else "model", parts=partes
                )
            )

    for intento in range(3):
        soltado = False
        try:
            for trozo in client.models.generate_content_stream(
                model=MODELO, contents=contenido, config=CONFIG
            ):
                if trozo.text:
                    soltado = True
                    yield trozo.text
            if not soltado:
                yield "Me quedé en blanco, mano. Pregúntame otra vez."
            return
        except Exception as e:
            if soltado:
                return  # ya se mostró algo, no repetimos
            error = str(e)
            if "503" in error or "UNAVAILABLE" in error:
                time.sleep(2 * (intento + 1))
                continue
            if "429" in error:
                yield "Nojoda, me tienen hablando mucho. Espérate tantico y me preguntas de nuevo."
                return
            yield "Ñerda, se dañó esta vaina. Inténtalo otra vez en un ratico."
            return
    yield "Google tiene el servicio full de gente, mano. Pregúntame en un ratico."


def mostrar(m):
    if m.get("audio"):
        st.audio(m["audio"], format=m.get("mime", "audio/wav"))
    if m.get("content"):
        st.markdown(m["content"])


st.title("CosteñoBot 🌴")
st.caption("Un barranquillero pa' charlar un rato. Escríbele o mándale un audio.")

# Memoria de la conversación (una por cada persona que abre la página)
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "audio_n" not in st.session_state:
    st.session_state.audio_n = 0  # Sirve para limpiar la grabación después de enviarla

with st.sidebar:
    if st.button("Olvidar la charla"):
        st.session_state.mensajes = []
        st.rerun()

# Grabadora de audio: se graba, se revisa y se manda al enviar
audio = st.audio_input("🎤 Mándale un audio", key=f"audio_{st.session_state.audio_n}")
pregunta = st.chat_input("Escríbele al costeño...")

for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        mostrar(m)

# Se envía apenas termina la grabación (como una nota de voz) o al escribir un texto
if audio is not None or pregunta:
    nuevo = {"role": "user", "content": pregunta or ""}
    if audio is not None:
        nuevo["audio"] = audio.getvalue()
        nuevo["mime"] = audio.type or "audio/wav"

    st.session_state.mensajes.append(nuevo)
    with st.chat_message("user"):
        mostrar(nuevo)

    with st.chat_message("assistant"):
        respuesta = st.write_stream(generar_stream(st.session_state.mensajes))
    st.session_state.mensajes.append({"role": "assistant", "content": respuesta})

    if audio is not None:
        st.session_state.audio_n += 1  # limpia la grabadora
        st.rerun()
