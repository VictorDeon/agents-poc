# Chatbot API (FastAPI + WhatsApp Simulado)

API FastAPI que expõe o pipeline RAG de `chatbot_com_rag` e simula a integração com o WhatsApp.

## Endpoints

- `GET /health` — Healthcheck simples
- `GET /whatsapp/webhook` — Verificação do webhook (modo subscribe)
- `POST /whatsapp/webhook` — Recebe mensagem e retorna resposta do RAG

## Payload de exemplo (POST)

```json
{
  "from": "+5511999999999",
  "text": "O que é RAG?",
  "session_id": "usuario-123"
}
```

## Observações

- Use a variável de ambiente `WHATSAPP_VERIFY_TOKEN` para a verificação do webhook.
- O pipeline RAG é inicializado no startup para evitar reindexação a cada request.
