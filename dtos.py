from pydantic import BaseModel, Field
from typing import Literal


class MainContext(BaseModel):
    """
    Contexto que será passado para o agente, contendo informações relevantes
    para a execução das ferramentas e do RAG.
    """

    session_id: str = Field(
        ...,
        description="Identificador único para a sessão de conversa do agente."
    )

    sentiment: str = Field(
        None,
        description="Sentimento detectado na pergunta do usuário, útil para personalizar respostas."
    )


class QuestionInputDTO(BaseModel):
    """
    Esquema de entrada para a ferramenta de informações do DataFrame.
    """

    question: str = Field(..., description="A pergunta do usuário relacionada a informações gerais do DataFrame.")


class AttachmentInputDTO(BaseModel):
    """
    Esquema de entrada para a ferramenta de informações multimodais.
    """

    question: str = Field(..., description="A pergunta do usuário relacionada ao anexo.")
    attachment_type: Literal["image", "video"] = Field(..., description="O tipo do anexo, pode ser 'image' ou 'video'.")
    attachment_url: str = Field(..., description="A URL do anexo, que pode ser uma imagem ou um vídeo.")


class ResponseSchema(BaseModel):
    """
    Esquema de resposta do agente, para validação de saída.
    """

    answer: str = Field(
        ...,
        description="A resposta final do agente para a pergunta do usuário."
    )