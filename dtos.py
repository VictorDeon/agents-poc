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