from utils import get_prompt, get_env_var
from langchain.tools import tool, ToolRuntime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dtos import MainContext, QuestionInputDTO
import pandas as pd


@tool(args_schema=QuestionInputDTO)
def statistical_summary_tool(question: str, runtime: ToolRuntime[MainContext]) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico
    sobre as colunas numéricas do DataFrame, incluindo medidas como média,
    mediana, desvio padrão, valores mínimos e máximos, e contagem de valores
    únicos para colunas categóricas.

    Args:
        question: A pergunta do usuário relacionada ao resumo estatístico do DataFrame.
    """

    context = runtime.context

    GROQ_API_KEY = get_env_var('GROQ_API_KEY')

    df = pd.read_csv('./assets/dados_entregas.csv')

    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model='llama-3.3-70b-versatile'
    )

    descritive_statistics = df.describe(include='number').transpose().to_string()

    prompt = get_prompt('estatistica.prompt.md')

    response_template = PromptTemplate(
        template=prompt,
        input_variables=["question", "summary"]
    )

    chain = response_template | llm | StrOutputParser()

    response = chain.invoke(
        {
            "question": question,
            "summary": descritive_statistics
        },
        config={"configurable": {"thread_id": context.session_id}},
        context=context
    )

    return response