# Motor de Transcripción + Pinyin + Traducción — Backend

Esta es la primera pieza del proyecto: el servicio que hace el trabajo pesado
(transcribir el audio, generar pinyin para chino, traducir al español).
Por ahora **no tiene cuentas, límites ni pagos** — eso viene en la siguiente
pieza, que se conecta encima de esto.

## 1. Instalar (una sola vez)

```
pip install -r requirements.txt
```

## 2. Configurar tus variables de entorno

Crea un archivo llamado `.env` en esta misma carpeta con estas líneas (usa tus
valores reales — la de Groq ya la tienes; las de Supabase las sacas de tu
proyecto en Project Settings → API):

```
GROQ_API_KEY=tu_api_key_de_groq
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_SECRET_KEY=tu_secret_key_de_supabase
```

⚠️ La SUPABASE_SECRET_KEY es como una llave maestra — nunca la compartas, ni
la subas a ningún lado público. Solo vive en este archivo `.env`.

## 3. Probarlo en tu computadora

```
uvicorn main:app --reload
```

Abre en tu navegador: **http://127.0.0.1:8000/docs**

Ahí Swagger te da una páginita donde puedes:
- Ver `/idiomas` — la lista de idiomas soportados (chino primero).
- Probar `/procesar` — sube un mp3 de prueba, escribe el código del idioma
  (`zh`, `en`, `ja`, `ko`, `fr`, `it`, `de`) y dale "Execute". Te regresa el
  JSON con texto, pinyin (solo si es chino) y traducción, frase por frase.
- Ver `/salud` — para confirmar que el servicio está vivo y que sí detectó
  tu GROQ_API_KEY.

## Qué sigue

Este backend, una vez que confirmes que te transcribe bien tus audios de
prueba, es sobre el que vamos a construir:
1. El sistema de cuentas y límites (3 gratis/día, 100 o 200 al mes de pago).
2. La página web con el diseño rojo/dorado.
3. La conexión con PayPal.
4. Subirlo todo a Render para que quede en línea de verdad.
