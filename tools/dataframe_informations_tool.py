from utils import get_prompt, get_env_var
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import pandas as pd


def dataframe_informations_tool(question: str) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar informações gerais
    sobre o DataFrame, incluindo número de colunas e linhas, nomes das colunas,
    e seus tipos de dados, contagem de dados nulos e duplicados para dar um
    panorama geral sobre o arquivo.
    """

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

    return response