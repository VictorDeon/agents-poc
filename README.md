# Chatbot API (FastAPI + WhatsApp Simulado)

API FastAPI que simula a integração com o WhatsApp.

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

## Rodar o servidor FastAPI

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Configure as variáveis de ambiente necessárias (ex: `WHATSAPP_VERIFY_TOKEN`, `GEMINI_API_KEY`).

3. Inicie o servidor:

```bash
uvicorn api.main:app --reload
```
4. O servidor estará disponível em `http://localhost:8000`. Use os endpoints descritos para interagir com o chatbot simulado.

5. Rodar o servidor mcp com inspect: `npx @modelcontextprotocol/inspector uv run --with mcp mcp run mcp-server/server.py` ou sem o inspect: `uv run --with mcp mcp run mcp-server/server.py` (recomendo usar o inspect para facilitar o desenvolvimento).