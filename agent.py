from utils import load_environment_variables, get_env_var
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from guardrails_security import GuardrailsSecurity
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.agents.middleware import ModelCallLimitMiddleware
from dtos import MainContext, ResponseSchema
from utils import get_prompt
from tools import (
    dataframe_informations_tool,
    statistical_summary_tool,
    graph_generator_tool,
    dataframe_python_tool,
    multimodal_inputs_tool,
    rag_tool
)


@dynamic_prompt
def agent_system_prompt(request: ModelRequest) -> str:
    """
    Middleware para injetar o prompt do sistema dinamicamente, permitindo personalização
    com base no contexto da conversa (ex.: sentimento do usuário).
    """

    sentiment = request.runtime.context.sentiment

    # Exemplo de personalização: ajustar o tom do agente com base no sentimento detectado.
    tone_instruction = "Tom de Voz: "
    if sentiment == "negative":
        tone_instruction += "Responda de forma empática e compreensiva."
    elif sentiment == "positive":
        tone_instruction += "Mantenha um tom entusiástico e amigável."
    elif sentiment == "expert":
        tone_instruction += "Responda de forma técnica e detalhada, adequada para um público especializado."
    elif sentiment == "beginner":
        tone_instruction += "Responda de forma simples e didática, adequada para um público iniciante com analogias simples."
    elif sentiment == "baby":
        tone_instruction += "Responda de forma extremamente simples e lúdica, adequada para uma criança pequena, usando analogias divertidas e linguagem muito acessível."
    else:
        tone_instruction += "Responda de forma clara e profissional."

    # Carrega o prompt base do sistema e injeta a instrução de tom personalizada.
    base_prompt = get_prompt('agent_system.prompt.md')
    full_prompt = f"{base_prompt}\n\n{tone_instruction}"

    return full_prompt


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
            dataframe_python_tool,
            multimodal_inputs_tool,
            rag_tool
        ]

        checkpointer = InMemorySaver()

        return create_agent(
            self.__llm,
            tools=tools,
            context_schema=MainContext,
            middleware=[
                agent_system_prompt,
                ModelCallLimitMiddleware(
                    thread_limit=10,     # Limite de 10 chamadas por thread para evitar loops infinitos.
                    run_limit=5,         # Limite de 5 chamadas por execução do agente para evitar abusos.
                    exit_behavior="end"  # Se os limites forem atingidos, o agente responderá com uma mensagem de encerramento e não fará mais chamadas ao modelo.
                )
            ],
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
            context=MainContext(session_id=self.__session_id, sentiment="baby")
        )
        structured_response: ResponseSchema = response["structured_response"]
        self.__guardrails.validate_output(structured_response.answer)
        return structured_response.answer
