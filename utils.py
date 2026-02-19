import os
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from typing import AsyncGenerator
from rich import print
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from contextlib import asynccontextmanager


def get_prompt(template_name: str, context: dict = {}) -> str:
    """
    Carrega e renderiza um template Jinja2 a partir da pasta de prompts.

    Args:
        template_name: Nome do arquivo do template (ex.: "meu_prompt.md").
        context: Dicionário com variáveis para renderizar o template.
    Returns:
        String com o template renderizado.
    """

    # Define o loader para a pasta de prompts do projeto.
    env = Environment(loader=FileSystemLoader("prompts"))
    # Renderiza o template com as variáveis fornecidas no contexto.
    return env.get_template(template_name).render(context)


def load_environment_variables() -> None:
    """
    Carrega variáveis de ambiente a partir do arquivo .env.
    """

    # Carrega variáveis do arquivo .env no ambiente do processo.
    load_dotenv()


def get_env_var(key: str, default: str | None = None) -> str | None:
    """
    Obtém uma variável de ambiente com fallback padrão.

    Args:
        key: Nome da variável de ambiente.
        default: Valor padrão caso a variável não exista.

    Returns:
        O valor da variável de ambiente ou o padrão informado.
    """

    return os.getenv(key, default)


@asynccontextmanager
async def db_checkpointer() -> AsyncGenerator[AsyncPostgresSaver | InMemorySaver, None]:
    """
    Lifespan para carregar variáveis de ambiente e realizar outras tarefas de setup.
    """

    print("Chatbot iniciado. Digite sua pergunta ou 'sair' para encerrar.")

    load_environment_variables()

    try:
        async with AsyncPostgresSaver.from_conn_string(get_env_var("DB_DSN")) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        yield InMemorySaver()  # Fallback para um saver em memória

    print("Finalizando chatbot")