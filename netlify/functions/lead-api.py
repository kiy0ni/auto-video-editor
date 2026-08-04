# Netlify function entry point – delegates to the FastAPI app.
# Netlify automatically creates a handler called `handler`. We expose the same
# name so that Netify can call it. The FastAPI app lives in `vortex_leads/api.py`.

from vortex_leads.api import app as fastapi_app

def handler(event, context):
    # Netlify functions use a WSGI‑like adapter; we rely on the built‑in
    # ` Mangum ` adapter that FastAPI provides for AWS Lambda / Netlify.
    from mangum import Mangum
    asgi_handler = Mangum(fastapi_app)
    return asgi_handler(event, context)
