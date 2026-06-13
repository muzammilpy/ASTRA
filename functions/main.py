"""
ASTRA – Firebase Cloud Functions entry point (Python 3.11, gen2)

Wraps the FastAPI app as a single HTTP Cloud Function.
Firebase Hosting rewrites /scan/* and /health to this function.
"""

import os
import sys

# Put app/ on the path so all internal imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import functions_framework
from mangum import Mangum
from mangum.types import LambdaContext

from app.main import app as fastapi_app

# Mangum wraps ASGI apps for serverless environments (Lambda, Cloud Functions)
_handler = Mangum(fastapi_app, lifespan="off")


@functions_framework.http
def astra_api(request):
    """
    Single HTTP Cloud Function handling all ASTRA API routes.
    Firebase Hosting rewrites:
      /scan   → astra_api
      /scan/* → astra_api
      /health → astra_api
    """
    # Build a minimal Lambda-style event from the Flask request
    import json

    body = request.get_data()
    if isinstance(body, bytes):
        import base64
        body_encoded = base64.b64encode(body).decode()
        is_base64    = True
    else:
        body_encoded = body
        is_base64    = False

    # Build headers dict
    headers = dict(request.headers)

    # Reconstruct query string
    query_params: dict = {}
    for k, v in request.args.items():
        query_params[k] = v

    event = {
        "httpMethod":              request.method,
        "path":                    request.path,
        "queryStringParameters":   query_params or None,
        "headers":                 headers,
        "body":                    body_encoded if body else None,
        "isBase64Encoded":         is_base64 and bool(body),
        "requestContext": {
            "http": {
                "method":    request.method,
                "path":      request.path,
                "protocol":  "HTTP/1.1",
                "sourceIp":  request.remote_addr or "0.0.0.0",
                "userAgent": headers.get("User-Agent", ""),
            }
        },
    }

    # Dummy Lambda context
    class _Ctx:
        function_name       = "astra_api"
        memory_limit_in_mb  = 512
        invoked_function_arn = "arn:aws:lambda:us-east-1:000000000000:function:astra_api"
        aws_request_id      = "local"
        def get_remaining_time_in_millis(self): return 30000

    response = _handler(event, _Ctx())

    import flask
    import base64 as b64

    status  = response.get("statusCode", 200)
    resp_headers = response.get("headers", {})
    resp_body    = response.get("body", "")

    if response.get("isBase64Encoded"):
        resp_body = b64.b64decode(resp_body)

    return flask.Response(
        response=resp_body,
        status=status,
        headers=resp_headers,
    )
