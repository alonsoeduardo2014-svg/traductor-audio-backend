"""
Motor de Transcripción + Pinyin + Traducción — Backend
=======================================================
Este es el "cerebro" del proyecto: recibe un archivo de audio + el idioma
que habla, y regresa, por cada frase:
  - el texto transcrito (en chino, siempre en SIMPLIFICADO)
  - el pinyin (SOLO si el idioma es chino)
  - la traducción al español

No maneja cuentas, límites diarios/mensuales ni pagos — eso viene en la
siguiente pieza del proyecto. Este archivo es solo el motor de procesamiento,
pensado para correr como servicio en Render.

REQUISITOS (instalar una sola vez):
    pip install -r requirements.txt

VARIABLE DE ENTORNO NECESARIA:
    GROQ_API_KEY=tu_api_key_de_groq

USO LOCAL (para probar antes de subirlo a Render):
    GROQ_API_KEY=tu_key uvicorn main:app --reload
    Luego abre http://127.0.0.1:8000/docs para probar el endpoint desde el navegador.
"""

import os
import time
import logging
from datetime import date

from dotenv import load_dotenv
load_dotenv()  # lee el archivo .env y carga las variables de entorno

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from groq import Groq
import opencc
from pypinyin import pinyin, Style
from deep_translator import GoogleTranslator, MyMemoryTranslator
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("motor")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
cliente_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY")
cliente_supabase: Client | None = (
    create_client(SUPABASE_URL, SUPABASE_SECRET_KEY) if SUPABASE_URL and SUPABASE_SECRET_KEY else None
)

# Cuántos audios puede procesar cada quien, según su plan.
# "free" = cuenta registrada pero sin suscripción de pago (mismo tope que un
# visitante sin cuenta, pero contado por cuenta en vez de por navegador).
LIMITES_POR_PLAN = {
    "free": 3,
    "plan100": 100,
    "plan200": 200,
}
LIMITE_ANONIMO_DIARIO = 3


class LimiteExcedido(Exception):
    """Se usa cuando alguien ya gastó todos sus audios (gratis o de su plan)."""
    pass

MODELO_WHISPER_GROQ = "whisper-large-v3-turbo"  # el más barato y rápido de Groq

# Chino va primero (es el idioma insignia del producto). El código de cada
# idioma es el que espera tanto Groq como los traductores.
IDIOMAS_SOPORTADOS = {
    "zh": "Chino",
    "en": "Inglés",
    "ja": "Japonés",
    "ko": "Coreano",
    "fr": "Francés",
    "it": "Italiano",
    "de": "Alemán",
}

_conversor_a_simplificado = opencc.OpenCC("t2s")

PAUSA_ENTRE_TRADUCCIONES = 0.3  # segundos, para no saturar las APIs gratuitas de traducción
REINTENTOS_TRADUCCION = 2

MARCADOR_FALLO_TRADUCCION = "(traducción no disponible"


# ---------------------------------------------------------------------------
# Funciones del motor (misma lógica que ya probamos en la app de escritorio,
# adaptada para servir muchos usuarios en vez de un solo archivo local)
# ---------------------------------------------------------------------------

def forzar_simplificado(texto_chino: str) -> str:
    """Whisper a veces transcribe en chino TRADICIONAL sin avisar. Esto lo
    convierte siempre a simplificado; si ya viene en simplificado, no cambia nada."""
    return _conversor_a_simplificado.convert(texto_chino)


def hanzi_a_pinyin(texto_chino: str) -> str:
    """Convierte caracteres chinos a pinyin con tonos, separado por espacios."""
    silabas = pinyin(texto_chino, style=Style.TONE)
    return " ".join(s[0] for s in silabas)


# Cada idioma de origen necesita, a veces, un código distinto para cada
# servicio de traducción (no todos usan el mismo estándar).
_CODIGOS_MYMEMORY = {
    "zh": "zh-CN",
    "en": "en-US",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr-FR",
    "it": "it-IT",
    "de": "de-DE",
}
_CODIGOS_GOOGLE = {
    "zh": "zh-CN",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "it": "it",
    "de": "de",
}


def traducir_es(texto: str, idioma_origen: str) -> str:
    """Traduce cualquiera de los idiomas soportados al español, probando dos
    servicios gratuitos en orden (Google primero, MyMemory de respaldo)."""
    if not texto.strip():
        return ""

    motores = (
        (GoogleTranslator, "Google", _CODIGOS_GOOGLE.get(idioma_origen, idioma_origen), "es"),
        (MyMemoryTranslator, "MyMemory", _CODIGOS_MYMEMORY.get(idioma_origen, idioma_origen), "es-ES"),
    )

    ultimo_error = None
    for Motor, nombre, cod_origen, cod_destino in motores:
        for intento in range(1, REINTENTOS_TRADUCCION + 1):
            try:
                return Motor(source=cod_origen, target=cod_destino).translate(texto)
            except Exception as e:
                ultimo_error = f"{nombre}: {e}"
                time.sleep(1.0 * intento)

    return f"{MARCADOR_FALLO_TRADUCCION} — {ultimo_error})"


def transcribir_con_groq(contenido_audio: bytes, nombre_archivo: str, idioma: str):
    """Manda el audio a la API de Groq (Whisper) y regresa la lista de
    segmentos (frase por frase) tal como los entrega el modelo."""
    if cliente_groq is None:
        raise RuntimeError(
            "El servidor no tiene configurada GROQ_API_KEY. Sin eso no se puede transcribir nada."
        )

    resultado = cliente_groq.audio.transcriptions.create(
        model=MODELO_WHISPER_GROQ,
        file=(nombre_archivo, contenido_audio),
        language=idioma,
        response_format="verbose_json",
    )

    # "segments" no viene tipado en el SDK (es un campo "extra" del JSON que
    # regresa la API), así que llega como una lista de diccionarios simples.
    segmentos = getattr(resultado, "segments", None) or []
    return segmentos


def procesar_audio_completo(contenido_audio: bytes, nombre_archivo: str, idioma: str) -> list[dict]:
    """Pipeline completo: transcribe -> (si es chino) simplifica + pinyin -> traduce.
    Regresa una lista de dicts: {"texto": ..., "pronunciacion": ... o None, "traduccion": ...}
    """
    segmentos = transcribir_con_groq(contenido_audio, nombre_archivo, idioma)

    filas = []
    for seg in segmentos:
        texto = (seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")) or ""
        texto = texto.strip()
        if not texto:
            continue

        pronunciacion = None
        if idioma == "zh":
            texto = forzar_simplificado(texto)
            pronunciacion = hanzi_a_pinyin(texto)

        traduccion = traducir_es(texto, idioma)
        time.sleep(PAUSA_ENTRE_TRADUCCIONES)

        filas.append({
            "texto": texto,
            "pronunciacion": pronunciacion,   # None para idiomas que no son chino
            "traduccion": traduccion,
        })

    return filas


# ---------------------------------------------------------------------------
# Cuentas, planes y límites de uso
# ---------------------------------------------------------------------------

def _mes_actual() -> str:
    return date.today().strftime("%Y-%m")


def obtener_usuario_desde_token(token: str):
    """Verifica el token que manda el navegador (JWT de sesión de Supabase).
    Regresa el objeto de usuario (con .id y .email) si es válido, o None si
    no hay sesión / el token no sirve."""
    if cliente_supabase is None or not token:
        return None
    try:
        respuesta = cliente_supabase.auth.get_user(token)
        return respuesta.user if respuesta else None
    except Exception:
        return None


def obtener_o_crear_perfil(id_usuario: str, email: str | None) -> dict:
    """Trae la fila de 'perfiles' de este usuario (el trigger de Supabase ya
    la crea sola al registrarse), y le reinicia el contador si ya cambió de mes."""
    resp = cliente_supabase.table("perfiles").select("*").eq("id", id_usuario).limit(1).execute()
    filas = resp.data

    if filas:
        perfil = filas[0]
    else:
        # No debería pasar (el trigger la crea), pero por si acaso no se pierde el uso.
        insertado = cliente_supabase.table("perfiles").insert({"id": id_usuario, "email": email}).execute()
        perfil = insertado.data[0]

    if perfil.get("mes_actual") != _mes_actual():
        actualizado = (
            cliente_supabase.table("perfiles")
            .update({"audios_usados_mes": 0, "mes_actual": _mes_actual()})
            .eq("id", id_usuario)
            .execute()
        )
        perfil = actualizado.data[0]

    return perfil


def verificar_y_registrar_uso_autenticado(perfil: dict) -> int:
    """Revisa que a este usuario no se le hayan acabado sus audios del mes,
    y si tiene espacio, le suma uno al contador. Regresa cuántos lleva
    usados YA CONTANDO este audio."""
    plan = perfil.get("plan") or "free"
    limite = LIMITES_POR_PLAN.get(plan, LIMITES_POR_PLAN["free"])
    usados = perfil.get("audios_usados_mes") or 0

    if usados >= limite:
        raise LimiteExcedido(
            f"Ya usaste tus {limite} audios de este mes con tu plan actual. "
            "Espera al siguiente mes o mejora tu plan para seguir traduciendo."
        )

    cliente_supabase.table("perfiles").update({"audios_usados_mes": usados + 1}).eq("id", perfil["id"]).execute()
    return usados + 1


def verificar_y_registrar_uso_anonimo(id_anonimo: str) -> int:
    """Igual que la de arriba, pero para alguien sin cuenta: 3 al día,
    contados por el código aleatorio que genera su navegador. Regresa
    cuántos lleva usados HOY, ya contando este audio."""
    hoy = date.today().isoformat()
    resp = (
        cliente_supabase.table("usos_anonimos")
        .select("*")
        .eq("id_anonimo", id_anonimo)
        .eq("fecha", hoy)
        .limit(1)
        .execute()
    )
    filas = resp.data

    if filas:
        fila = filas[0]
        if fila["cantidad"] >= LIMITE_ANONIMO_DIARIO:
            raise LimiteExcedido(
                f"Ya usaste tus {LIMITE_ANONIMO_DIARIO} audios gratis de hoy. "
                "Crea una cuenta gratis y suscríbete para tener muchos más al mes."
            )
        nueva_cantidad = fila["cantidad"] + 1
        (
            cliente_supabase.table("usos_anonimos")
            .update({"cantidad": nueva_cantidad})
            .eq("id_anonimo", id_anonimo)
            .eq("fecha", hoy)
            .execute()
        )
    else:
        nueva_cantidad = 1
        cliente_supabase.table("usos_anonimos").insert(
            {"id_anonimo": id_anonimo, "fecha": hoy, "cantidad": 1}
        ).execute()

    return nueva_cantidad


def verificar_acceso(autorizacion: str | None, id_anonimo: str | None) -> dict:
    """Punto único de control antes de procesar cualquier audio:
    - Si viene un token válido -> ruta de usuario con cuenta (checa su plan).
    - Si no -> ruta anónima (checa el límite de 3 al día por navegador).
    Regresa info de a quién se le cobró el uso y cuánto lleva, para que la
    página pueda mostrar algo como "te quedan 97 de 100 este mes".
    """
    if cliente_supabase is None:
        raise RuntimeError(
            "El servidor no tiene configuradas SUPABASE_URL / SUPABASE_SECRET_KEY."
        )

    token = None
    if autorizacion and autorizacion.lower().startswith("bearer "):
        token = autorizacion[7:].strip()

    if token:
        usuario = obtener_usuario_desde_token(token)
        if usuario is not None:
            perfil = obtener_o_crear_perfil(usuario.id, usuario.email)
            plan = perfil.get("plan") or "free"
            usados = verificar_y_registrar_uso_autenticado(perfil)
            return {
                "tipo": "cuenta",
                "email": usuario.email,
                "plan": plan,
                "usados": usados,
                "limite": LIMITES_POR_PLAN.get(plan, LIMITES_POR_PLAN["free"]),
            }

    if not id_anonimo:
        raise HTTPException(
            status_code=400,
            detail="Falta identificar el navegador (encabezado X-Id-Anonimo) para contar tus audios gratis.",
        )

    usados_hoy = verificar_y_registrar_uso_anonimo(id_anonimo)
    return {"tipo": "anonimo", "usados": usados_hoy, "limite": LIMITE_ANONIMO_DIARIO}


# ---------------------------------------------------------------------------
# API web (FastAPI)
# ---------------------------------------------------------------------------

app = FastAPI(title="Motor de Transcripción + Pinyin + Traducción")

# CORS abierto por ahora (cualquier página puede llamar a este backend).
# Cuando ya tengamos el dominio final del sitio, lo cerramos solo a ese dominio.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/salud")
def salud():
    """Para que Render (o quien sea) pueda checar que el servicio sigue vivo."""
    return {
        "estado": "ok",
        "groq_configurado": cliente_groq is not None,
        "supabase_configurado": cliente_supabase is not None,
    }


@app.get("/idiomas")
def listar_idiomas():
    """Idiomas que la app puede procesar, en el orden en que deben aparecer
    en el selector (chino primero, como idioma insignia)."""
    return [{"codigo": codigo, "nombre": nombre} for codigo, nombre in IDIOMAS_SOPORTADOS.items()]


@app.get("/estado")
def estado(
    authorization: str | None = Header(default=None),
    x_id_anonimo: str | None = Header(default=None),
):
    """Para que la página consulte el plan y cuántos audios lleva alguien,
    SIN gastarle ninguno (a diferencia de /procesar, este no suma al contador)."""
    if cliente_supabase is None:
        raise HTTPException(status_code=500, detail="El servidor no tiene configurado Supabase.")

    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if token:
        usuario = obtener_usuario_desde_token(token)
        if usuario is not None:
            perfil = obtener_o_crear_perfil(usuario.id, usuario.email)
            plan = perfil.get("plan") or "free"
            return {
                "tipo": "cuenta",
                "email": usuario.email,
                "plan": plan,
                "usados": perfil.get("audios_usados_mes") or 0,
                "limite": LIMITES_POR_PLAN.get(plan, LIMITES_POR_PLAN["free"]),
            }

    if not x_id_anonimo:
        return {"tipo": "anonimo", "usados": 0, "limite": LIMITE_ANONIMO_DIARIO}

    hoy = date.today().isoformat()
    resp = (
        cliente_supabase.table("usos_anonimos")
        .select("*")
        .eq("id_anonimo", x_id_anonimo)
        .eq("fecha", hoy)
        .limit(1)
        .execute()
    )
    filas = resp.data
    usados = filas[0]["cantidad"] if filas else 0
    return {"tipo": "anonimo", "usados": usados, "limite": LIMITE_ANONIMO_DIARIO}


@app.post("/procesar")
async def procesar(
    archivo: UploadFile = File(...),
    idioma: str = Form(...),
    authorization: str | None = Header(default=None),
    x_id_anonimo: str | None = Header(default=None),
):
    if idioma not in IDIOMAS_SOPORTADOS:
        raise HTTPException(
            status_code=400,
            detail=f"Idioma no soportado: '{idioma}'. Usa uno de: {', '.join(IDIOMAS_SOPORTADOS)}",
        )

    contenido = await archivo.read()
    if not contenido:
        raise HTTPException(status_code=400, detail="El archivo llegó vacío.")

    try:
        info_acceso = verificar_acceso(authorization, x_id_anonimo)
    except LimiteExcedido as e:
        raise HTTPException(status_code=429, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        filas = procesar_audio_completo(contenido, archivo.filename or "audio.mp3", idioma)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error procesando audio")
        raise HTTPException(status_code=500, detail=f"Error procesando el audio: {e}")

    return {
        "idioma": idioma,
        "nombre_idioma": IDIOMAS_SOPORTADOS[idioma],
        "resultados": filas,
        "acceso": info_acceso,
    }
