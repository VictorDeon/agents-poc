from utils import load_environment_variables, get_env_var
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from vetorial_db import results_by_chromadb
from etls import etl_pdf_process, etl_db_process

load_environment_variables()


def main():
    GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')

    # Instanciando um modelo de embeddings do google
    # permitindo transformar texto em vetores numéricos
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    documents = []
    documents += etl_pdf_process()
    documents += etl_db_process()
    print("Total de documentos para indexação:", len(documents))

    vector_store = results_by_chromadb(documents, embeddings)

    # Armazenar o histórico da conversa, mantendo apenas as últimas 5 interações
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",  # chave que vai guardar o histórico da conversa
        return_messages=True,       # retorna as mensagens no formato de lista de mensagens (ChatMessage)
        output_key='answer'         # chave que vai guardar a resposta gerada pelo modelo
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.1,
        api_key=GEMINI_API_KEY,
        convert_system_message_to_human=True  # Converte mensagens do sistema para mensagens do usuário
    )

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Configura o número de documentos a serem recuperados
        return_source_documents=True,  # Retorna os documentos de origem junto com a resposta
        verbose=True  # Ativa o modo verbose para mostrar detalhes do processo de recuperação e geração
    )

    question = "Qual foi o total de vendas no primeiro trimestre de 2024?"
    print(f"\nPergunta: {question}")
    try:
        response = chat_chain({"question": question})
        print(f"Resposta: {response['answer']}")
        print(f"\nDocumentos utilizados: {len(response['source_documents'])}")
    except Exception as e:
        print(f"Erro ao processar a pergunta: {e}")


if __name__ == "__main__":
    main()