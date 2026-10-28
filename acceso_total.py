# -*- coding: utf-8 -*-
"""
Acceso Total · Instituto Alonso Laoshi
======================================
Una sola suscripción mensual abre TODAS las aplicaciones (Traductor, Trazos
Chinos, Mapa del mundo en chino y las que se agreguen después).

Este archivo es IGUAL en las tres apps. Si se cambia algo aquí, hay que subir
la misma copia a los tres repos.

Planes de PayPal (se ponen en Render → Environment de cada app):
    PLAN_TOTAL  → Acceso Total, 4 USD al mes (Traductor: 100 audios al mes)
    PLAN_PLUS   → Acceso Total Plus, 6 USD al mes (Traductor: 200 audios al mes)

Se guarda en la tabla "acceso_total" de Supabase (una fila por persona):
    usuario, email, id_suscripcion, plan ('total' o 'plus'),
    estado ('activa', 'cancelada' o 'inactiva'), pagado_hasta, actualizado

Reglas:
  - activa    → tiene todo; se cobra cada mes.
  - cancelada → ya no se cobra, pero conserva todo hasta "pagado_hasta".
  - inactiva  → PayPal no pudo cobrar, o ya pasó la fecha: versión gratis.

Solo usa lo que ya trae Python (no hay que instalar nada).
"""
import os
import json
import time
import logging
import threading
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("acceso-total")

# ---------------------------------------------------------------- configuración
SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_KEY") or ""
PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID") or ""
PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET") or ""
PAYPAL_API = os.environ.get("PAYPAL_API") or (
    "https://api-m.sandbox.paypal.com" if (os.environ.get("PAYPAL_MODO") or "").lower() == "sandbox"
    else "https://api-m.paypal.com")
PLAN_TOTAL = (os.environ.get("PLAN_TOTAL") or "").strip()
PLAN_PLUS = (os.environ.get("PLAN_PLUS") or "").strip()

PRECIO_TOTAL = "4 USD"
PRECIO_PLUS = "6 USD"
TABLA = "acceso_total"

REVISAR_CADA = 12 * 3600       # cada 12 h se le pregunta a PayPal (por si se perdió un aviso)
REVISAR_VENCIDA = 3600         # si ya pasó el día de cobro, cada hora
DIAS_DE_GRACIA = 2             # espera por si PayPal tarda en cobrar el mes
RECUERDA_SEG = 60              # memoria corta para no consultar Supabase en cada clic


class ErrorAcceso(Exception):
    """Error con un mensaje listo para mostrarle a la persona."""
    def __init__(self, status: int, mensaje: str):
        super().__init__(mensaje)
        self.status = status
        self.mensaje = mensaje


def configurado() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def paypal_configurado() -> bool:
    return bool(PAYPAL_CLIENT_ID and PAYPAL_SECRET)


def planes_publicos() -> dict:
    """Lo que la página necesita para dibujar los botones de PayPal."""
    return {"total": PLAN_TOTAL, "plus": PLAN_PLUS, "precio_total": PRECIO_TOTAL, "precio_plus": PRECIO_PLUS}


# ---------------------------------------------------------------- fechas
def _fecha(txt):
    if not txt:
        return None
    try:
        d = datetime.fromisoformat(str(txt).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _ahora():
    return datetime.now(timezone.utc)


def _mas_un_mes(d):
    a, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    for dia in (d.day, 30, 29, 28):
        try:
            return d.replace(year=a, month=m, day=dia)
        except ValueError:
            continue
    return d + timedelta(days=30)


def pagado_hasta_de(sub):
    """Hasta qué día tiene pagado según PayPal: el próximo cobro, o un mes
    después del último pago si ya no habrá cobro."""
    info = (sub or {}).get("billing_info") or {}
    d = _fecha(info.get("next_billing_time"))
    if d:
        return d.isoformat()
    u = _fecha((info.get("last_payment") or {}).get("time"))
    return _mas_un_mes(u).isoformat() if u else None


def plan_de(plan_id):
    """'total', 'plus' o None si el plan de PayPal no es del Acceso Total."""
    if plan_id and plan_id == PLAN_PLUS:
        return "plus"
    if plan_id and plan_id == PLAN_TOTAL:
        return "total"
    return None


# ---------------------------------------------------------------- llamadas web
def _http(metodo, url, headers=None, cuerpo=None, timeout=15):
    datos = None
    h = dict(headers or {})
    if cuerpo is not None:
        if isinstance(cuerpo, (dict, list)):
            datos = json.dumps(cuerpo).encode("utf-8")
            h.setdefault("Content-Type", "application/json")
        elif isinstance(cuerpo, str):
            datos = cuerpo.encode("utf-8")
        else:
            datos = cuerpo
    req = urllib.request.Request(url, data=datos, headers=h, method=metodo)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            texto = r.read().decode("utf-8") or ""
            return r.status, (json.loads(texto) if texto.strip()[:1] in "[{" else texto)
    except urllib.error.HTTPError as e:
        texto = e.read().decode("utf-8", "replace")
        return e.code, texto


def _sb_headers(extra=None):
    h = {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
    if SUPABASE_KEY.startswith("eyJ"):   # llave antigua (JWT): también va aquí
        h["Authorization"] = "Bearer " + SUPABASE_KEY
    if extra:
        h.update(extra)
    return h


def _sb(metodo, consulta, cuerpo=None, extra=None):
    url = f"{SUPABASE_URL}/rest/v1/{TABLA}{consulta}"
    st, d = _http(metodo, url, _sb_headers(extra), cuerpo)
    if st >= 300:
        raise RuntimeError(f"Supabase {st}: {str(d)[:300]}")
    return d


# ---------------------------------------------------------------- tabla
_memoria = {}           # usuario -> (hasta cuándo, fila)
_candado = threading.Lock()


def _olvidar(usuario=None):
    with _candado:
        if usuario:
            _memoria.pop(usuario, None)
        else:
            _memoria.clear()


def fila_de(usuario, fresca=False):
    if not usuario:
        return None
    ahora = time.time()
    if not fresca:
        with _candado:
            m = _memoria.get(usuario)
            if m and m[0] > ahora:
                return m[1]
    d = _sb("GET", "?" + urllib.parse.urlencode({"usuario": f"eq.{usuario}", "select": "*"}))
    fila = d[0] if isinstance(d, list) and d else None
    with _candado:
        if len(_memoria) > 5000:
            _memoria.clear()
        _memoria[usuario] = (ahora + RECUERDA_SEG, fila)
    return fila


def _fila_por_sub(id_sub):
    d = _sb("GET", "?" + urllib.parse.urlencode({"id_suscripcion": f"eq.{id_sub}", "select": "*"}))
    return d[0] if isinstance(d, list) and d else None


def _guardar(usuario, email, id_sub, plan, estado, pagado_hasta):
    datos = {"usuario": usuario, "id_suscripcion": id_sub, "plan": plan or "total", "estado": estado,
             "pagado_hasta": pagado_hasta, "actualizado": _ahora().isoformat()}
    if email:
        datos["email"] = email
    _sb("POST", "?on_conflict=usuario", datos,
        {"Prefer": "resolution=merge-duplicates,return=minimal"})
    _olvidar(usuario)
    logger.info("Acceso Total %s -> %s (%s, hasta %s, usuario %s)", id_sub, estado, plan, pagado_hasta, usuario)


def _cambiar_por_sub(id_sub, datos):
    datos = dict(datos, actualizado=_ahora().isoformat())
    _sb("PATCH", "?" + urllib.parse.urlencode({"id_suscripcion": f"eq.{id_sub}"}), datos,
        {"Prefer": "return=minimal"})
    _olvidar()


# ---------------------------------------------------------------- PayPal
_pp = {"t": None, "exp": 0}


def _token_paypal():
    if _pp["t"] and time.time() < _pp["exp"] - 60:
        return _pp["t"]
    import base64
    cred = base64.b64encode(f"{PAYPAL_CLIENT_ID}:{PAYPAL_SECRET}".encode()).decode()
    st, d = _http("POST", f"{PAYPAL_API}/v1/oauth2/token",
                  {"Authorization": "Basic " + cred, "Content-Type": "application/x-www-form-urlencoded"},
                  "grant_type=client_credentials")
    if st != 200 or not isinstance(d, dict):
        raise RuntimeError(f"PayPal token {st}")
    _pp["t"] = d["access_token"]
    _pp["exp"] = time.time() + int(d.get("expires_in", 3000))
    return _pp["t"]


def consultar_paypal(id_sub):
    """Pregunta directo a PayPal cómo está una suscripción (nadie puede fingir que pagó)."""
    st, d = _http("GET", f"{PAYPAL_API}/v1/billing/subscriptions/{urllib.parse.quote(id_sub)}",
                  {"Authorization": "Bearer " + _token_paypal()})
    if st != 200 or not isinstance(d, dict):
        raise RuntimeError(f"PayPal consulta {st}: {str(d)[:200]}")
    return d


def cancelar_en_paypal(id_sub, razon="La persona canceló desde la aplicación"):
    st, d = _http("POST", f"{PAYPAL_API}/v1/billing/subscriptions/{urllib.parse.quote(id_sub)}/cancel",
                  {"Authorization": "Bearer " + _token_paypal()}, {"reason": razon})
    if st == 204:
        return
    if st == 422 and "SUBSCRIPTION_STATUS_INVALID" in str(d):
        return   # ya estaba cancelada
    raise RuntimeError(f"PayPal cancelar {st}: {str(d)[:200]}")


def verificar_firma(headers, evento, webhook_id):
    """Le pregunta a PayPal si el aviso de verdad viene de él."""
    if not webhook_id:
        return False
    cuerpo = {
        "auth_algo": headers.get("paypal-auth-algo"),
        "cert_url": headers.get("paypal-cert-url"),
        "transmission_id": headers.get("paypal-transmission-id"),
        "transmission_sig": headers.get("paypal-transmission-sig"),
        "transmission_time": headers.get("paypal-transmission-time"),
        "webhook_id": webhook_id,
        "webhook_event": evento,
    }
    st, d = _http("POST", f"{PAYPAL_API}/v1/notifications/verify-webhook-signature",
                  {"Authorization": "Bearer " + _token_paypal()}, cuerpo, timeout=20)
    return st == 200 and isinstance(d, dict) and d.get("verification_status") == "SUCCESS"


# ---------------------------------------------------------------- reglas
def tiene_acceso(fila):
    if not fila or not fila.get("id_suscripcion"):
        return False
    estado = fila.get("estado")
    hasta = _fecha(fila.get("pagado_hasta"))
    ahora = _ahora()
    if estado == "activa":
        # si ya pasó el día de cobro (más unos días) y no se cobró, regresa a gratis
        return hasta is None or ahora <= hasta + timedelta(days=DIAS_DE_GRACIA)
    if estado == "cancelada":
        return bool(hasta and hasta > ahora)
    return False


def marcar_cancelada(id_sub, sub=None, hasta=None):
    """Ya no se cobra, pero conserva todo hasta el día que ya pagó."""
    fila = _fila_por_sub(id_sub)
    if not fila:
        return None
    exacta = _fecha(hasta) or _fecha(fila.get("pagado_hasta"))
    estimada = _fecha(pagado_hasta_de(sub)) if sub else None
    fecha = exacta
    if estimada and (exacta is None or estimada > exacta + timedelta(days=3)):
        fecha = estimada      # PayPal registra un pago más reciente del que teníamos
    if fecha is None or fecha <= _ahora():
        _cambiar_por_sub(id_sub, {"estado": "inactiva"})
        return None
    _cambiar_por_sub(id_sub, {"estado": "cancelada", "pagado_hasta": fecha.isoformat()})
    return fecha.isoformat()


def sincronizar(id_sub):
    """Deja la suscripción igual a lo que dice PayPal. Ignora planes que no son del Acceso Total."""
    sub = consultar_paypal(id_sub)
    plan = plan_de(sub.get("plan_id"))
    if not plan:
        return
    st = sub.get("status")
    if st == "ACTIVE" and sub.get("custom_id"):
        fila = fila_de(sub["custom_id"], fresca=True)
        if fila and fila.get("id_suscripcion") not in (None, sub["id"]) and tiene_acceso(fila):
            return    # esa persona ya tiene otra suscripción más nueva
        _guardar(sub["custom_id"], None, sub["id"], plan, "activa", pagado_hasta_de(sub))
    elif st in ("CANCELLED", "EXPIRED"):
        marcar_cancelada(sub["id"], sub)
    elif st == "SUSPENDED":
        _cambiar_por_sub(sub["id"], {"estado": "inactiva"})


def estado_de(usuario):
    """Lo que la página necesita saber:
    {"suscrito": bool, "plan": 'total'|'plus'|None, "estado": 'activa'|'cancelada'|None, "hasta": fecha}"""
    vacio = {"suscrito": False, "plan": None, "estado": None, "hasta": None}
    if not configurado() or not usuario:
        return vacio
    fila = fila_de(usuario)
    if fila and fila.get("id_suscripcion") and paypal_configurado() and fila.get("estado") in ("activa", "cancelada"):
        desde = time.time() - (_fecha(fila.get("actualizado")) or datetime(2000, 1, 1, tzinfo=timezone.utc)).timestamp()
        hasta = _fecha(fila.get("pagado_hasta"))
        vencida = fila.get("estado") == "activa" and (hasta is None or _ahora() > hasta)
        if desde > REVISAR_CADA or (vencida and desde > REVISAR_VENCIDA):
            try:
                sincronizar(fila["id_suscripcion"])
                fila = fila_de(usuario, fresca=True)
            except Exception:
                logger.exception("No se pudo revisar la suscripción con PayPal")
                # si PayPal no contesta, no se le quita nada a la persona
    if not tiene_acceso(fila):
        return vacio
    return {"suscrito": True, "plan": fila.get("plan") or "total", "estado": fila.get("estado"),
            "hasta": fila.get("pagado_hasta")}


def confirmar(usuario, email, id_sub):
    """La página llama aquí justo después de pagar, para activar al instante."""
    if not configurado() or not paypal_configurado():
        raise ErrorAcceso(500, "El servidor no tiene configurado PayPal o Supabase.")
    try:
        sub = consultar_paypal(id_sub)
    except Exception:
        logger.exception("No se pudo consultar la suscripción en PayPal")
        raise ErrorAcceso(502, "No pudimos confirmar tu pago con PayPal. Si ya pagaste, se activará solo en unos minutos.")
    if sub.get("custom_id") != usuario:
        raise ErrorAcceso(403, "Esta suscripción no pertenece a tu cuenta.")
    plan = plan_de(sub.get("plan_id"))
    if not plan:
        raise ErrorAcceso(400, "Este plan de PayPal no es el Acceso Total.")
    if sub.get("status") not in ("ACTIVE", "APPROVED"):
        raise ErrorAcceso(400, "Tu pago todavía no aparece como activo. Espera unos minutos.")

    # Si ya tenía otra suscripción activa (por ejemplo, pasa de 4 a 6 USD),
    # la anterior se cancela para que no se le cobre doble.
    anterior = fila_de(usuario, fresca=True)
    if anterior and anterior.get("id_suscripcion") and anterior["id_suscripcion"] != sub["id"] \
            and anterior.get("estado") == "activa":
        try:
            cancelar_en_paypal(anterior["id_suscripcion"], "Cambió a otro plan del Acceso Total")
        except Exception:
            logger.exception("No se pudo cancelar la suscripción anterior %s", anterior["id_suscripcion"])

    hasta = pagado_hasta_de(sub) or _mas_un_mes(_ahora()).isoformat()
    _guardar(usuario, email, sub["id"], plan, "activa", hasta)
    return {"ok": True, "plan": plan, "hasta": hasta}


def cancelar(usuario):
    """Botón "Cancelar mi suscripción": ya no se cobra y conserva todo hasta el día que pagó."""
    if not configurado() or not paypal_configurado():
        raise ErrorAcceso(500, "El servidor no tiene configurado PayPal o Supabase.")
    fila = fila_de(usuario, fresca=True)
    if not fila or not fila.get("id_suscripcion") or not tiene_acceso(fila):
        raise ErrorAcceso(404, "No encontramos una suscripción activa en tu cuenta.")
    if fila.get("estado") == "cancelada":
        return {"ok": True, "hasta": fila.get("pagado_hasta"), "ya_estaba": True}
    id_sub = fila["id_suscripcion"]
    hasta = fila.get("pagado_hasta")
    try:
        sub = consultar_paypal(id_sub)        # antes de cancelar, para saber hasta cuándo pagó
        hasta = pagado_hasta_de(sub) or hasta
    except Exception:
        logger.exception("No se pudo consultar la suscripción antes de cancelar")
    try:
        cancelar_en_paypal(id_sub)
    except Exception:
        logger.exception("No se pudo cancelar en PayPal")
        raise ErrorAcceso(502, "No pudimos cancelar en PayPal en este momento. Intenta de nuevo en unos minutos.")
    fecha = marcar_cancelada(id_sub, hasta=hasta)
    return {"ok": True, "hasta": fecha}


def aplicar_aviso(evento):
    """Aplica un aviso de PayPal (ya verificado). Regresa True si era del Acceso Total."""
    tipo = evento.get("event_type", "")
    rec = evento.get("resource", {}) or {}
    if tipo.startswith("BILLING.SUBSCRIPTION."):
        if rec.get("plan_id") and not plan_de(rec.get("plan_id")):
            return False
        if tipo == "BILLING.SUBSCRIPTION.ACTIVATED" and rec.get("custom_id") and rec.get("id"):
            sincronizar(rec["id"])
        elif tipo in ("BILLING.SUBSCRIPTION.CANCELLED", "BILLING.SUBSCRIPTION.EXPIRED") and rec.get("id"):
            marcar_cancelada(rec["id"], rec)
        elif tipo == "BILLING.SUBSCRIPTION.SUSPENDED" and rec.get("id"):
            _cambiar_por_sub(rec["id"], {"estado": "inactiva"})
        return True
    if tipo == "PAYMENT.SALE.COMPLETED" and rec.get("billing_agreement_id"):
        sincronizar(rec["billing_agreement_id"])
        return True
    return False
