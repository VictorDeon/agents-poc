"""API FastAPI que simula integração com a API do WhatsApp.

Baseada no pipeline RAG em chatbot_com_rag.
"""

from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import hashlib
import hmac
import json
import logging
import time

from agent import Agent
from utils import load_environment_variables, get_env_var

app = FastAPI(title="Chatbot RAG (WhatsApp Simulado)")

logger = logging.getLogger("chatbot_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    _log_event(
        "http_request",
        method=request.method,
        path=str(request.url.path),
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


class WhatsAppMessage(BaseModel):
    """
    Payload simplificado de mensagem recebida do WhatsApp.
    """

    from_number: str = Field(..., alias="from")
    text: str
    session_id: str | None = None


class WhatsAppReply(BaseModel):
    """
    Resposta simulada para o WhatsApp.
    """

    to: str
    reply: str
    documents_used: int | None = None


def _log_event(event: str, **fields: object) -> None:
    payload = {"event": event, "ts": int(time.time()), **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))


def _verify_whatsapp_signature(request_body: bytes, signature_header: str | None) -> None:
    """
    Valida assinatura do webhook usando HMAC SHA256 (X-Hub-Signature-256).
    """

    secret = get_env_var("WHATSAPP_APP_SECRET")
    if not secret:
        return

    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Assinatura ausente ou inválida")

    received = signature_header.replace("sha256=", "").strip()
    expected = hmac.new(secret.encode("utf-8"), request_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(received, expected):
        raise HTTPException(status_code=401, detail="Assinatura inválida")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Inicializa guardrails e pipeline RAG.
    """

    load_environment_variables()
    Agent(session_id="default")


@app.get("/health")
def health() -> dict[str, str]:
    """
    Healthcheck simples.
    """

    return {"status": "ok"}


@app.get("/whatsapp/webhook", response_class=PlainTextResponse)
def verify_webhook(
    hub_mode: str | None = None,
    hub_challenge: str | None = None,
    hub_verify_token: str | None = None,
) -> str:
    """
    Simula o handshake de verificação do webhook do WhatsApp Cloud API.

    Espera parâmetros no formato: hub.mode, hub.challenge, hub.verify_token.
    """

    expected_token = get_env_var("WHATSAPP_VERIFY_TOKEN")
    if hub_mode == "subscribe" and hub_verify_token == expected_token:
        return hub_challenge or ""

    raise HTTPException(status_code=403, detail="Token de verificação inválido")


@app.post("/whatsapp/webhook", response_model=WhatsAppReply)
async def receive_message(request: Request, payload: WhatsAppMessage) -> WhatsAppReply:
    """
    Recebe a mensagem do WhatsApp e responde usando o RAG.
    """

    chat = Agent(session_id=payload.session_id or payload.from_number)

    # raw_body = await request.body()
    # _verify_whatsapp_signature(raw_body, request.headers.get("X-Hub-Signature-256"))

    session_id = payload.session_id or payload.from_number
    _log_event("message_received", from_number=payload.from_number, session_id=session_id)

    try:
        response: dict = chat.invoke(payload.text)
        answer = response.get("answer", "").strip()
        documents_used = len(response.get("context", []))
        _log_event("message_answered", from_number=payload.from_number, session_id=session_id, documents_used=documents_used)
        return WhatsAppReply(to=payload.from_number, reply=answer, documents_used=documents_used)
    except ValueError as exc:
        _log_event("message_rejected", from_number=payload.from_number, session_id=session_id, reason=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log_event("message_error", from_number=payload.from_number, session_id=session_id, reason=str(exc))
        raise HTTPException(status_code=500, detail=f"Erro ao processar mensagem: {exc}")
