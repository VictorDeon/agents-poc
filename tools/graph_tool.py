from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.graph.state import RunnableConfig
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.chat_models import init_chat_model
from enum import Enum
from utils import checkpointer
from dtos import MainContext, QuestionInputDTO
from typing import TypedDict, Annotated, Sequence
from rich import print


class GraphType(Enum):
    CALL_LLM = "call_llm"


class ToolState(TypedDict):
    # Usamos Annotated para marcar que esses campos são mensagens criadas pelo add_messages que devem ser
    # adicionadas ao contexto da conversa. (SystemMessage, HumanMessage, AIMessage, etc.)
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_llm(state: ToolState) -> ToolState:
    print("Entrei no node `call_llm` do grafo")
    llm = init_chat_model(model="google_genai:gemini-2.5-flash-lite")
    llm_result = llm.invoke(state["messages"])
    return ToolState(messages=[llm_result])


@tool(args_schema=QuestionInputDTO)
def graph_tool(question: str, runtime: ToolRuntime[MainContext]) -> str:
    """
    Utilize esta ferramenta SEMPRE que o usuário quiser conversar sobre qualquer assunto.
    Se não usar nenhuma outra ferramenta, use esta. Ela é a mais genérica de todas, e pode ser usada para qualquer tipo de pergunta.

    Args:
        question: A pergunta do usuário relacionada a informações gerais do DataFrame.
        runtime: O contexto de execução da ferramenta, fornecido pelo agente.
    """

    context = runtime.context

    print(f"Entrei na ferramenta 'graph_tool' com a pergunta: \"{question}\"")

    builder = StateGraph(ToolState, context_schema=MainContext)

    # Adicionando os nós
    builder.add_node(GraphType.CALL_LLM.value, call_llm)

    # Adicionando as arestas
    builder.add_edge(START, GraphType.CALL_LLM.value)
    builder.add_edge(GraphType.CALL_LLM.value, END)

    graph = builder.compile(checkpointer=checkpointer)

    config = RunnableConfig(
        configurable={"thread_id": context.session_id}
    )

    result = graph.invoke(
        {"messages": [HumanMessage(question)]},
        config=config,
        context=context
    )

    return result