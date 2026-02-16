import os
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader


def get_prompt(template_name):
    """
    Carrega um template Jinja2 a partir da pasta 'prompts'
    """

    env = Environment(loader=FileSystemLoader('analise_de_dados/prompts'))
    return env.get_template(template_name).render()


def load_environment_variables():
    """
    Carrega as variáveis de ambiente do arquivo .env
    """

    load_dotenv()


def get_env_var(key, default=None):
    """
    Retorna o valor da variável de ambiente ou o valor padrão
    """

    return os.getenv(key, default)
