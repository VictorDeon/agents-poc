from utils import load_environment_variables, get_env_var
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash-lite",
    #     temperature=0,
    #     api_key=GEMINI_API_KEY
    # )

    documents = []
    documents += etl_pdf_process()
    documents += etl_db_process()
    print("Total de documentos para indexação:", len(documents))

    vector_store = results_by_chromadb(documents, embeddings)
    pergunta1 = "Qual foi a receita com laptops?"

    results = vector_store.similarity_search(pergunta1, k=2)
    print("Documentos mais relevantes para a pergunta 1:")
    for doc in results:
        print(f"- {doc.metadata['source']}: {doc.page_content}")

    pergunta2 = "Me fale sobre o teclado mecânico"
    results2 = vector_store.similarity_search(pergunta2, k=2)
    print("Documentos mais relevantes para a pergunta 2:")
    for doc in results2:
        print(f"- {doc.metadata['source']}: {doc.page_content}")


if __name__ == "__main__":
    main()