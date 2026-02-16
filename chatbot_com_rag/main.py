from utils import load_environment_variables, get_env_var
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
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

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.1,
        api_key=GEMINI_API_KEY
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Dado o histórico da conversa e a pergunta do usuário, reformule a pergunta para uma versão independente, "
            "sem perder o contexto. NÃO responda a pergunta."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Você é um assistente útil. Use o contexto recuperado para responder à pergunta. "
            "Se a resposta não estiver no contexto, diga que não encontrou a informação.\n\n"
            "Contexto:\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    session_store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    chat_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    question = "Qual foi o total de vendas no primeiro trimestre de 2024?"
    print(f"\nPergunta: {question}")
    try:
        response = chat_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "default"}}
        )
        print(f"Resposta: {response['answer']}")
        print(f"\nDocumentos utilizados: {len(response['context'])}")
    except Exception as e:
        print(f"Erro ao processar a pergunta: {e}")


if __name__ == "__main__":
    main()