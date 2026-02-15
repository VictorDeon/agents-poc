from utils import load_environment_variables, get_env_var
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pandas as pd

load_environment_variables()


def main():
    GROQ_API_KEY = get_env_var('GROQ_API_KEY')

    df = pd.read_csv('./analise_de_dados/assets/dados_entregas.csv')

    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model='llama-3.3-70b-versatile'
    )

    ai_msg = llm.invoke("""
        Eu tenho um dataframe chamado `df` com as colunas `anos_experiencia_agente` e `tempo_entrega`. Escreva o código python com a biblioteca
        Pandas para calcular a correlação entre as duas colunas.
        Retorne o Markdown para o trecho de código python e nada mais.
    """)

    python_tool = PythonAstREPLTool(locals={"df": df})
    python_tool.invoke(ai_msg.content)

    llm_with_tools = llm.bind_tools(
        [python_tool],
        tool_choice=python_tool.name
    )

    parser = JsonOutputKeyToolsParser(
        key_name=python_tool.name,
        first_tool_only=True
    )

    md_head = df.head().to_markdown()

    system_prompt = f"""
        Você tem acesso a um dataframe pandas `df`.
        Aqui está a saída de `df.head().to_markdown()`:

        ```
        {md_head}
        ```

        Dada uma pergunta do usuário, escreva o código Python para respondê-la.
        Retorne SOMENTE o código python válido e nada mais.
        Não presuma que você tem acesso a nenhuma biblioteca além das bibliotecas Python integradas e pandas.
    """

    # system = System Prompt (prompt para a persona do agente)
    # human = Human Prompt (prompt que o usuário vai enviar)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # Pega o template do prompt envia para a llm, essa envia a resposta para o parser de saida
    # e essa envia a resposta parseada para a ferramenta de execução de codigo python
    chain = prompt | llm_with_tools | parser | python_tool

    response = chain.invoke({"question": """
        Qual é a correlação entre anos de experiência do agente e tempo de entrega?
    """})

    print(response)


if __name__ == "__main__":
    main()
