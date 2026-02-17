from utils import get_prompt, get_env_var
from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dtos import MainContext
from datetime import datetime
from uuid import uuid4
from langchain_groq import ChatGroq
import pandas as pd


class DataFrameInformationsToolInput(BaseModel):
    """
    Esquema de entrada para a ferramenta de informações do DataFrame.
    """

    question: str = Field(..., description="A pergunta do usuário relacionada a informações gerais do DataFrame.")


@tool(args_schema=DataFrameInformationsToolInput)
def dataframe_informations_tool(question: str, runtime: ToolRuntime) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar informações gerais
    sobre o DataFrame, incluindo número de colunas e linhas, nomes das colunas,
    e seus tipos de dados, contagem de dados nulos e duplicados para dar um
    panorama geral sobre o arquivo.

    Args:
        question: A pergunta do usuário relacionada a informações gerais do DataFrame.
        runtime: O contexto de execução da ferramenta, fornecido pelo agente.
    """

    context: MainContext = runtime.context
    store = runtime.store
    history = runtime.store.search(
        namespace=("sessions", context.session_id, "questions"),
        query="recupere as últimas 3 perguntas",
        filter={"tool": "dataframe_informations_tool"},
        limit=3
    )
    print(f"Histórico de perguntas anteriores nesta sessão: {history}")

    GROQ_API_KEY = get_env_var('GROQ_API_KEY')

    df = pd.read_csv('./assets/dados_entregas.csv')

    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model='llama-3.3-70b-versatile'
    )

    shape = df.shape
    columns = df.dtypes
    nulls = df.isnull().sum()
    nulls_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq("nan").sum())
    duplicates = df.duplicated().sum()

    prompt = get_prompt('exploratoria.prompt.md')

    response_template = PromptTemplate(
        template=prompt,
        input_variables=["question", "shape", "columns", "nulls", "nulls_str", "duplicates"]
    )

    chain = response_template | llm | StrOutputParser()

    response = chain.invoke({
        "question": question,
        "shape": shape,
        "columns": columns,
        "nulls": nulls,
        "nulls_str": nulls_str,
        "duplicates": duplicates
    })

    # Armazena a última pergunta para contexto futuro.
    store.put(
        namespace=("sessions", runtime.session_id, "questions"),
        key=f"qid_{uuid4()}",
        value={
            "question": question,
            "answer": response,
            "tool": "dataframe_informations_tool",
            "created_at": datetime.now().isoformat()
        },
        index=["tool", "created_at"],
        ttl=3600  # Armazena por 1 hora para contexto futuro.
    )

    return response