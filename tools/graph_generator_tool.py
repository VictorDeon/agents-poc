from utils import get_prompt, get_env_var
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def graph_generator_tool(question: str) -> plt.Figure:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir
    de um DataFrame pandas (`df`) com base em uma instrução do usuário. A instrução
    pode conter pedidos como:
    - "Crie um gráfico da média de tempo de entrega por clima."
    - "Plote a distribuição do tempo de entrega."
    - "Plote a relação entre a classificação dos agentes e o tempo de entrega"

    Palavras-chave comuns que indicam o uso desta ferramenta incluem:
    - "crie um gráfico"
    - "plote"
    - "visualize"
    - "faça um gráfico de"
    - "mostre a distribuição de"
    - "represente graficamente"

    Entre outros pedidos e palavras-chave que indicam a necessidade de gerar um gráfico a partir dos dados do DataFrame.
    """

    GROQ_API_KEY = get_env_var('GROQ_API_KEY')

    df = pd.read_csv('./assets/dados_entregas.csv')

    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model='llama-3.3-70b-versatile'
    )

    columns = [f"- {col}: ({dtype})" for col, dtype in df.dtypes.items()]
    samples = df.head(20).to_dict(orient='records')

    prompt = get_prompt('visual.prompt.md')

    response_template = PromptTemplate(
        template=prompt,
        input_variables=["question", "columns", "sample"]
    )

    chain = response_template | llm | StrOutputParser()

    response_code = chain.invoke({
        "question": question,
        "columns": "\n".join(columns),
        "sample": samples
    })

    clean_code = response_code.replace("```python", "").replace("```", "").strip()

    exec_globals = {"df": df, "plt": plt, "sns": sns}
    exec_locals = {}
    exec(clean_code, exec_globals, exec_locals)

    fig = plt.gcf()  # Get the current figure after executing the code

    return fig