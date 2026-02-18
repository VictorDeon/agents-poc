from langgraph.graph import StateGraph, START, add_messages
from langgraph.graph.state import RunnableConfig
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
from enum import Enum
from utils import checkpointer
from dtos import MainContext, QuestionInputDTO
from typing import TypedDict, Annotated, Sequence, Literal
from rich import print


@tool
def multiply_subtool(a: float, b: float) -> float:
    """
    Multiplica dois números.

    Args:
        a: O primeiro número.
        b: O segundo número.

    Returns:
        O resultado da multiplicação de a e b.
    """

    return a * b


@tool
def add_subtool(a: float, b: float) -> float:
    """
    Soma dois números.

    Args:
        a: O primeiro número.
        b: O segundo número.

    Returns:
        O resultado da soma de a e b.
    """

    return a + b


@tool
def subtract_subtool(a: float, b: float) -> float:
    """
    Subtrai dois números.

    Args:
        a: O primeiro número.
        b: O segundo número.

    Returns:
        O resultado da subtração de a e b.
    """

    return a - b


@tool
def divide_subtool(a: float, b: float) -> float:
    """
    Divide dois números.

    Args:
        a: O primeiro número (dividendo).
        b: O segundo número (divisor).

    Returns:
        O resultado da divisão de a por b, ou uma mensagem de erro se b for zero.
    """

    if b == 0:
        return "Divisão por zero não é permitida."

    return a / b


class GraphType(Enum):
    CALL_LLM = "call_llm"
    TOOL_NODE = "tool_node"


class ToolState(TypedDict):
    # Usamos Annotated para marcar que esses campos são mensagens criadas pelo add_messages que devem ser
    # adicionadas ao contexto da conversa. (SystemMessage, HumanMessage, AIMessage, etc.)
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_llm(state: ToolState) -> ToolState:
    """
    Node que carrega a llm com as ferramentas matemáticas.
    """

    print("Entrei no node `call_llm` do grafo")
    llm = init_chat_model(model="google_genai:gemini-2.5-flash-lite")
    llm_with_tools = llm.bind_tools([multiply_subtool, add_subtool, subtract_subtool, divide_subtool])
    llm_result = llm_with_tools.invoke(state["messages"])
    return ToolState(messages=[llm_result])


def tool_node(state: ToolState) -> ToolState:
    """
    Node roteador que identifica chamadas de ferramentas feitas pela LLM e as executa,
    retornando o resultado como uma nova mensagem.
    """

    print("Entrei no node `tool_node` do grafo")
    llm_response = state["messages"][-1]
    if not isinstance(llm_response, AIMessage) or not getattr(llm_response, "tool_calls", None):
        return state

    call = llm_response.tool_calls[-1]
    name, args, id_ = call["name"], call["args"], call["id"]
    print(f"LLM solicitou a ferramenta '{name}' com os argumentos {args}")

    # Mapeamento local das ferramentas
    tools_by_name = {
        tool.name: tool for tool in [multiply_subtool, add_subtool, subtract_subtool, divide_subtool]
    }

    try:
        content = str(tools_by_name[name].invoke(args))
        status = "success"
    except Exception as e:
        content = f"Por favor, arrume o seu error: {str(e)}"
        status = "error"

    tool_message = ToolMessage(content=content, tool_call_id=id_, status=status)
    return ToolState(messages=[tool_message])


def router(state: ToolState) -> Literal["tool_node", "__end__"]:
    """
    Roteador que ou vai rodar uma ferramenta matematica ou finaliza.
    """

    print("Entrei no roteador do grafo")

    llm_response = state["messages"][-1]
    if getattr(llm_response, "tool_calls", None):
        return "tool_node"

    return "__end__"


@tool(args_schema=QuestionInputDTO)
def graph_tool(question: str, runtime: ToolRuntime[MainContext]) -> str:
    """
    Utilize esta ferramenta SEMPRE que o usuário pedir alguma conta matemática básica como
    soma, multiplicação, divisão ou subtração.

    Args:
        question: A pergunta do usuário relacionada a operações matemáticas básicas.
        runtime: O contexto de execução da ferramenta, fornecido pelo agente.

    Returns:
        A resposta gerada pelo modelo de linguagem após processar a pergunta do usuário.

    Ferramentas Disponíveis:
        - multiply_subtool: Multiplica dois números.
        - add_subtool: Soma dois números.
        - subtract_subtool: Subtrai dois números.
        - divide_subtool: Divide dois números.
    """

    print(f"Entrei na ferramenta 'graph_tool' com a pergunta: \"{question}\"")

    context = runtime.context

    builder = StateGraph(ToolState, context_schema=MainContext)

    # Adicionando os nós
    builder.add_node(GraphType.CALL_LLM.value, call_llm)
    builder.add_node("tool_node", tool_node)

    # Adicionando as arestas
    builder.add_edge(START, GraphType.CALL_LLM.value)
    builder.add_conditional_edges('call_llm', router, ["tool_node", "__end__"])
    builder.add_edge(GraphType.TOOL_NODE.value, GraphType.CALL_LLM.value)

    graph = builder.compile(checkpointer=checkpointer)

    # Usar thread_id isolado para evitar conflito de estrutura com o agente principal
    config = RunnableConfig(
        configurable={"thread_id": f"graph_tool_{context.session_id}"}
    )

    result = graph.invoke(
        {"messages": [
            SystemMessage(content="Você é um assistente matemático. Após executar as ferramentas matemáticas, SEMPRE responda com o resultado em linguagem natural, dizendo apenas o número resultado sem explicações adicionais."),
            HumanMessage(question)
        ]},
        config=config,
        context=context
    )

    last_message = result["messages"][-1]
    answer = last_message.content if result["messages"] else "Desculpe, não consegui gerar uma resposta."

    return answer