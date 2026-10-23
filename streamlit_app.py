import os
import time

import streamlit as st
from google import genai
from google.genai import types

MODELO = "gemini-3.8-flash"
MAX_MENSAJES = 16  # Cuántos mensajes recuerda el bot en cada chat

SYSTEM_PROMPT = """
Eres un asistente con la personalidad de un pelao de Barranquilla: costeño de verdad, relajao
y buena gente. Hablas como se habla en la calle, pero lo MÁS importante es que SIEMPRE respondes
lo que te preguntan, con información real y correcta. La jerga es solo el tono, nunca una excusa
pa' no responder.

REGLA #1: RESPONDER DE VERDAD
- Cuando te pregunten algo (un tema, una explicación, datos, cómo se hace algo, noticias, ciencia,
  historia, salud, tareas, tecnología...), contéstalo completo y bien explicado, con datos concretos.
  Nada de evadir la pregunta, nada de "esa vaina es compleja", nada de contestar con puro chiste.
- Si necesitas datos actuales o no estás seguro de algo, busca información antes de responder.
  No inventes nada. Si no encuentras algo, dilo sin rodeos.
- Si te piden explicar un tema (por ejemplo, el cuerpo humano), explícalo en serio: organizado, con
  las partes importantes, en párrafos o listas cortas, con ejemplos si ayudan.
- En temas de salud das la información útil y, si es algo serio, recomiendas ir al médico,
  sin sermonear.

REGLA #2: LARGO DE LA RESPUESTA (varía según lo que pidan)
- Charla casual, saludos, opiniones rápidas, datos simples: CORTO, 1 o 2 frases, como un mensaje
  de WhatsApp.
- Preguntas que piden explicación, información, pasos, comparaciones o "cuéntame de...":
  LARGO y completo, el que haga falta (varios párrafos si toca).
- Si el usuario pide "resumido" o "corto", hazlo corto. Si pide "detallado" o "a fondo", hazlo largo.

ESTILO COSTEÑO
- Natural, no actuado. Hablas "pa'", "e'", "to'", "tá", "pue'" sin forzarlo en cada palabra.
- Jerga costeña solo cuando cae natural (joa, mano, sisa, nojoda, ñerda, mondá, joda, vaina,
  mierda, carajo, ajá, eso e' así). Pocas por respuesta, no en cada frase.
- Vulgaridad LEVE de la que se dice entre amigos. Nada fuerte, nada contra personas, razas,
  géneros ni grupos. Sin insultos graves.
- NO repitas muletillas ni arranques de frase respecto a tus mensajes anteriores. Varía.
- Sin emojis. Sin tono de profesor ni de robot. Humor seco de vez en cuando, no en cada mensaje.

AUDIOS
- Si el usuario te manda un audio, escúchalo y respóndele directo, como si fuera una nota de voz
  de WhatsApp. No digas "escuché tu audio" ni lo transcribas, simplemente responde.
- Si el audio no se entiende o está vacío, dilo sin rodeos y pídele que lo mande otra vez.

EJEMPLOS DEL TONO (no los copies, solo captura el estilo)
Usuario: ¿Cuál es la capital de Francia?
Tú: París, mano.

Usuario: ¿Cómo estás?
Tú: Aquí, sobreviviendo al calor. ¿Y tú qué?

Usuario: Explícame cómo funciona el corazón
Tú: Mira, el corazón es una bomba muscular, más o menos del tamaño de un puño, que no para nunca.
Tiene cuatro cámaras: arriba las aurículas, que reciben la sangre, y abajo los ventrículos, que la
empujan pa' to'a parte. El lado derecho manda la sangre sin oxígeno a los pulmones, y el izquierdo
agarra la que ya viene oxigenada y la lanza al resto del cuerpo por la aorta. Unas válvulas se abren
y se cierran pa' que la sangre vaya en un solo sentido, y de ahí sale el "lub-dub" del latido.
En reposo late entre 60 y 100 veces por minuto, o sea, más de cien mil veces al día, nojoda.
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

# Con búsqueda en Google (para que consulte información real) y sin búsqueda (plan B)
CONFIG_BUSQUEDA = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=1.0,
    max_output_tokens=3000,
    tools=[types.Tool(google_search=types.GoogleSearch())],
)
CONFIG_BASE = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=1.0,
    max_output_tokens=3000,
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


def recoger_fuentes(trozo, fuentes):
    """Guarda los links de donde Gemini sacó la información (si buscó en internet)."""
    try:
        meta = trozo.candidates[0].grounding_metadata
        for g in meta.grounding_chunks or []:
            if g.web and g.web.uri and g.web.uri not in [u for _, u in fuentes]:
                fuentes.append((g.web.title or "fuente", g.web.uri))
    except Exception:
        pass


def generar_stream(mensajes, fuentes):
    """Va soltando la respuesta por pedacitos. Primero intenta con búsqueda en Google."""
    contenido = []
    for m in mensajes[-MAX_MENSAJES:]:
        partes = partes_de(m)
        if partes:
            contenido.append(
                types.Content(
                    role="user" if m["role"] == "user" else "model", parts=partes
                )
            )

    ultimo_error = ""
    for config in (CONFIG_BUSQUEDA, CONFIG_BASE):
        for intento in range(3):
            soltado = False
            try:
                for trozo in client.models.generate_content_stream(
                    model=MODELO, contents=contenido, config=config
                ):
                    recoger_fuentes(trozo, fuentes)
                    if trozo.text:
                        soltado = True
                        yield trozo.text
                if not soltado:
                    yield "Me quedé en blanco, mano. Pregúntame otra vez."
                return
            except Exception as e:
                if soltado:
                    return  # ya se mostró algo, no repetimos
                ultimo_error = str(e)
                if "503" in ultimo_error or "UNAVAILABLE" in ultimo_error:
                    time.sleep(2 * (intento + 1))
                    continue
                break  # otro error: probamos sin búsqueda

    if "503" in ultimo_error or "UNAVAILABLE" in ultimo_error:
        yield "Google tiene el servicio full de gente, mano. Pregúntame en un ratico."
    elif "429" in ultimo_error:
        yield "Nojoda, me tienen hablando mucho. Espérate tantico y me preguntas de nuevo."
    else:
        yield "Ñerda, se dañó esta vaina. Inténtalo otra vez en un ratico."


def mostrar(m):
    if m.get("audio"):
        st.audio(m["audio"], format=m.get("mime", "audio/wav"))
    if m.get("content"):
        st.markdown(m["content"])
    if m.get("fuentes"):
        enlaces = " · ".join(f"[{t}]({u})" for t, u in m["fuentes"][:5])
        st.caption(f"Fuentes: {enlaces}")


# ---------- Varios chats ----------
if "chats" not in st.session_state:
    st.session_state.chats = {1: {"titulo": "Chat 1", "mensajes": []}}
    st.session_state.chat_actual = 1
    st.session_state.siguiente_id = 2


def nuevo_chat():
    i = st.session_state.siguiente_id
    st.session_state.chats[i] = {"titulo": f"Chat {i}", "mensajes": []}
    st.session_state.chat_actual = i
    st.session_state.siguiente_id += 1


def cambiar_chat(i):
    st.session_state.chat_actual = i


def borrar_chat(i):
    st.session_state.chats.pop(i, None)
    if not st.session_state.chats:
        nuevo_chat()
    elif st.session_state.chat_actual not in st.session_state.chats:
        st.session_state.chat_actual = list(st.session_state.chats)[-1]


actual = st.session_state.chat_actual
chat = st.session_state.chats[actual]

st.title("CosteñoBot 🌴")
st.caption("Un barranquillero pa' charlar un rato. Escríbele o mándale un audio con el micrófono.")

for m in chat["mensajes"]:
    with st.chat_message(m["role"]):
        mostrar(m)

# El micrófono sale dentro de la caja de escribir, al lado del botón de enviar
prompt = st.chat_input("Escríbele al costeño...", accept_audio=True)

if prompt:
    if isinstance(prompt, str):  # por si el Streamlit es viejo
        texto, audio = prompt.strip(), None
    else:
        texto = (prompt.text or "").strip()
        audio = getattr(prompt, "audio", None)

    nuevo = {"role": "user", "content": texto}
    if audio is not None:
        nuevo["audio"] = audio.getvalue()
        nuevo["mime"] = audio.type or "audio/wav"

    # El título del chat sale del primer mensaje
    if not chat["mensajes"]:
        chat["titulo"] = (texto[:28] + ("…" if len(texto) > 28 else "")) if texto else "🎤 Audio"

    chat["mensajes"].append(nuevo)
    with st.chat_message("user"):
        mostrar(nuevo)

    fuentes = []
    with st.chat_message("assistant"):
        respuesta = st.write_stream(generar_stream(chat["mensajes"], fuentes))
        if fuentes:
            enlaces = " · ".join(f"[{t}]({u})" for t, u in fuentes[:5])
            st.caption(f"Fuentes: {enlaces}")
    chat["mensajes"].append(
        {"role": "assistant", "content": respuesta, "fuentes": fuentes[:5]}
    )

# La barra lateral va al final para que el título del chat salga actualizado
with st.sidebar:
    st.button("➕ Nuevo chat", on_click=nuevo_chat, width="stretch")
    st.divider()
    for i, c in st.session_state.chats.items():
        st.button(
            c["titulo"],
            key=f"chat_{i}",
            on_click=cambiar_chat,
            args=(i,),
            type="primary" if i == actual else "secondary",
            width="stretch",
        )
    st.divider()
    st.button("🗑️ Borrar este chat", on_click=borrar_chat, args=(actual,), width="stretch")
