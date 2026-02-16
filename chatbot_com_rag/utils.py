"""Utilitários compartilhados do projeto.

Inclui carregamento de variáveis de ambiente e leitura de prompts Jinja2.
"""

import os
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader


def get_prompt(template_name: str) -> str:
    """
    Carrega e renderiza um template Jinja2 a partir da pasta de prompts.

    Args:
        template_name: Nome do arquivo do template (ex.: "meu_prompt.md").

    Returns:
        String com o template renderizado.
    """

    # Define o loader para a pasta de prompts do projeto.
    env = Environment(loader=FileSystemLoader("chatbot_com_rag/prompts"))
    # Renderiza o template sem variáveis adicionais (render padrão).
    return env.get_template(template_name).render()


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
