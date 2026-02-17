from utils import load_environment_variables, get_env_var
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from guardrails_security import GuardrailsSecurity
from langgraph.checkpoint.memory import InMemorySaver
from dtos import MainContext, ResponseSchema
from utils import get_prompt
from tools import (
    dataframe_informations_tool,
    statistical_summary_tool,
    graph_generator_tool,
    dataframe_python_tool
)


class Agent:
    """
    Singleton para instância do agente.
    """

    __instance: "Agent" = None
    __session_store: dict[str, InMemoryStore] = {}

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
        self.__session: InMemoryStore = None
        self.__session_id: str = None
        self.__chain = self.__build_tool_agent()

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
            Agent.__instance.__session_store[session_id] = InMemoryStore()

        Agent.__instance.__session = Agent.__instance.__session_store[session_id]

        return Agent.__instance

    def __build_tool_agent(self):
        """
        Cria um agente com ferramentas do RAG e do agente de dados.
        """

        tools = [
            dataframe_informations_tool,
            statistical_summary_tool,
            graph_generator_tool,
            dataframe_python_tool
        ]

        checkpointer = InMemorySaver()

        return create_agent(
            self.__llm,
            tools=tools,
            system_prompt=get_prompt('agent_system.prompt.md'),
            context_schema=MainContext,
            store=self.__session,
            response_format=ResponseSchema,
            checkpointer=checkpointer
        )

    def invoke(self, question: str) -> str:
        """
        Executa o agente com ferramentas (RAG + análise de dados).
        """

        self.__guardrails.validate_input(question)
        response = self.__chain.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": self.__session_id}},
            context=MainContext(session_id=self.__session_id)
        )
        structured_response: ResponseSchema = response["structured_response"]
        self.__guardrails.validate_output(structured_response.answer)
        return structured_response.answer
