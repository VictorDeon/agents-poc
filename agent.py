from utils import load_environment_variables, get_env_var
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, create_react_agent, AgentExecutor
from guardrails_security import GuardrailsSecurity
from utils import get_prompt
from tools import (
    dataframe_informations_tool,
    statistical_summary_tool,
    graph_generator_tool,
    dataframe_python_tool,
    rag_tool
)


class Agent:
    """
    Singleton para instância do agente.
    """

    __instance: "Agent" = None
    __session_store: dict[str, InMemoryChatMessageHistory] = {}

    def __init__(self) -> None:
        """
        Inicializa o histórico da sessão.
        """

        if self.__instance is not None:
            raise ValueError("O objeto já existe! utilize a função get_instance()")

        print("Inicializando agente")

        load_environment_variables()
        GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')

        self.__guardrails = GuardrailsSecurity()  # Placeholder para futuras validações de entrada/saída.
        self.__llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",  # Modelo leve/rápido para conversação.
            temperature=0.1,  # Baixa aleatoriedade para respostas mais consistentes.
            api_key=GEMINI_API_KEY  # Credencial exigida pela API.
        )
        self.__chain: AgentExecutor = None
        self.__session: InMemoryChatMessageHistory = None
        self.__session_id: str = None

    @staticmethod
    def get_instance(session_id: str) -> "Agent":
        """
        Inicializa o agente, configurando o pipeline de RAG e o histórico de mensagens.
        """

        if Agent.__instance is None:
            Agent.__instance = Agent()

        print(f"Criando nova sessão de conversa com o agente '{session_id}'...")

        Agent.__instance.__session_id = session_id

        if session_id not in Agent.__instance.__session_store:
            Agent.__instance.__session_store[session_id] = InMemoryChatMessageHistory()

        Agent.__instance.__session = Agent.__instance.__session_store[session_id]
        Agent.__instance.__chain = Agent.__instance.__build_tool_agent()

        return Agent.__instance

    def __build_tool_agent(self) -> AgentExecutor:
        """
        Cria um agente com ferramentas do RAG e do agente de dados.
        """

        tools = [
            Tool(
                name="Aulas RAG Answer",
                func=lambda question: rag_tool(question, self.__session, self.__session_id),
                description="""
                    Utilize esta ferramenta para responder perguntas usando os documentos do RAG (conteúdo de PDFs e dados).
                    Perguntas referentes os tópicos: Arquitera de RAG, Armazenamento Vetorial, Embeddings,
                    Pipeline de dados, Cadeias de Conversação, LLMs, Avaliação com LangSmith e RAGAS,
                    Hybrid Search e técnicas Avançadas de RAG devem ser respondidas utilizando esta ferramenta,
                    que tem acesso ao conteúdo dos documentos.
                """
            ),
            Tool(
                name="DataFrame Informations",
                func=lambda question: dataframe_informations_tool(question),
                description="""
                    Utilize esta ferramenta sempre que o usuário solicitar
                    informações gerais sobre o dataframe, incluindo número
                    de colunas e linhas, nomes das colunas e seus tipos
                    de dados, contagem de dados nulos e duplicados para
                    dar um panorama geral sobre o arquivo.
                """
            ),
            Tool(
                name="Statistical Summary",
                func=lambda question: statistical_summary_tool(question),
                description="""
                    Utilize esta ferramenta sempre que o usuário solicitar um
                    resumo estatístico completo e descritivo da base de dados,
                    incluindo várias estatísticas (média, desvio padrão,
                    mínimo, máximo etc.). Não utilize esta ferramenta para
                    calcular uma única métrica como 'qual é a média de X'
                    ou qual a correlação das variáveis'. Nesses casos,
                    utilize a ferramenta python_executor.
                """
            ),
            Tool(
                name="Graph Generator",
                func=lambda question: graph_generator_tool(question),
                description="""
                    Utilize esta ferramenta sempre que o usuário solicitar um
                    gráfico a partir de um DataFrame pandas (df) com base em
                    uma instrução do usuário. A instrução pode conter
                    pedidos como: 'Crie um gráfico da média de tempo de entrega por clima',
                    'Plote a distribuição do tempo de entrega' ou
                    'Plote a relação entre a classificação dos agentes e o tempo de entrega'.
                    Palavras-chave comuns que indicam a necessidade de gerar um gráfico incluem:
                    'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de',
                    'mostre a distribuição', 'represente graficamente', entre outros.
                """
            ),
            Tool(
                name="Python Executor",
                func=dataframe_python_tool(),
                description="""
                    Utilize esta ferramenta sempre que o usuário solicitar cálculos,
                    consultas ou transformações específicas usando Python diretamente
                    sobre o DataFrame df. Exemplos de uso incluem: 'Qual é a média da coluna X?',
                    'Quais são os valores únicos da coluna Y?' ou 'Qual a correlação entre A e B?'.
                    Evite utilizar esta ferramenta para solicitações mais amplas ou descritivas,
                    como informações gerais sobre o DataFrame, resumos estatísticos completos
                    ou geração de gráficos — nesses casos, use as ferramentas apropriadas.
                """
            )
        ]

        prompt_react_template = ChatPromptTemplate.from_messages([
            ("system", get_prompt("react.prompt.md")),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        agent = create_react_agent(llm=self.__llm, tools=tools, prompt=prompt_react_template)

        memory = ConversationBufferMemory(
            chat_memory=self.__session,
            return_messages=True,
            memory_key="chat_history"
        )

        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=60
        )

    def invoke(self, question: str) -> str:
        """
        Executa o agente com ferramentas (RAG + análise de dados).
        """

        self.__guardrails.validate_input(question)
        response = self.__chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": self.__session_id}}
        )
        self.__guardrails.validate_output(response["output"])
        return response["output"]
