from pydantic import BaseModel, Field


class MainContext(BaseModel):
    """
    Contexto que será passado para o agente, contendo informações relevantes
    para a execução das ferramentas e do RAG.
    """

    session_id: str = Field(
        ...,
        description="Identificador único para a sessão de conversa do agente."
    )


class QuestionInputDTO(BaseModel):
    """
    Esquema de entrada para a ferramenta de informações do DataFrame.
    """

    question: str = Field(..., description="A pergunta do usuário relacionada a informações gerais do DataFrame.")


class ResponseSchema(BaseModel):
    """
    Esquema de resposta do agente, para validação de saída.
    """

    answer: str = Field(
        ...,
        description="A resposta final do agente para a pergunta do usuário."
    )