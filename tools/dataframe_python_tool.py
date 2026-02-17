from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd


def dataframe_python_tool() -> PythonAstREPLTool:
    df = pd.read_csv('./assets/dados_entregas.csv')
    return PythonAstREPLTool(locals={"df": df})