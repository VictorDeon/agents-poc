from langchain_experimental.tools import PythonAstREPLTool
from langchain.tools import tool, ToolRuntime
from dtos import MainContext, QuestionInputDTO
import pandas as pd


@tool(args_schema=QuestionInputDTO)
def dataframe_python_tool(question: str, runtime: ToolRuntime[MainContext]) -> PythonAstREPLTool:
    """
    Utilize esta ferramenta sempre que o usuário solicitar cálculos,
    consultas ou transformações específicas usando Python diretamente
    sobre o DataFrame df. Exemplos de uso incluem: 'Qual é a média da coluna X?',
    'Quais são os valores únicos da coluna Y?' ou 'Qual a correlação entre A e B?'.
    Evite utilizar esta ferramenta para solicitações mais amplas ou descritivas,
    como informações gerais sobre o DataFrame, resumos estatísticos completos
    ou geração de gráficos — nesses casos, use as ferramentas apropriadas.
    """

    print(f"Entrei na ferramenta 'dataframe_python_tool' com a pergunta: \"{question}\"")

    context = runtime.context

    df = pd.read_csv('./assets/dados_entregas.csv')

    return PythonAstREPLTool(locals={"df": df}).with_config({"configurable": {"thread_id": context.session_id}})