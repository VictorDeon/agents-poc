from utils import get_env_var
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_pinecone import Pinecone
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from pinecone import ServerlessSpec, Pinecone as PineconeClient
from langchain.schema import Document
import os
import shutil
import faiss


def results_by_cache(embeddings: GoogleGenerativeAIEmbeddings) -> CacheBackedEmbeddings:
    """
    Cria um cache persistente de embeddings para acelerar consultas.

    Args:
        embeddings: Modelo de embeddings base (ex.: Gemini).

    Returns:
        Wrapper com cache persistente para reutilizar vetores já calculados.
    """

    # Pasta local para persistir vetores (reduz custo e latência).
    store = LocalFileStore("./embeddings_cache")
    cached_embeddings: CacheBackedEmbeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace="gemini-embedding-cache"
    )

    return cached_embeddings


def results_by_faissdb(company_documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """
    Cria um índice FAISS local a partir de documentos.

    Args:
        company_documents: Lista de documentos para indexação.
        embeddings: Modelo de embeddings usado para vetorizar textos.

    Returns:
        Índice FAISS pronto para busca por similaridade.
    """

    # Remove metadados complexos que o FAISS não consegue serializar.
    filtered_documents = filter_complex_metadata(company_documents)

    # Configuração do índice HNSW para ganho de performance em bases maiores.
    dimension = 768  # Dimensão dos embeddings do modelo gemini-embedding-001.
    neighbors = 32  # Número de vizinhos para o grafo HNSW.
    faiss.IndexHNSWFlat(dimension, neighbors)  # Instancia HNSW (aproximado e rápido).

    # Cria índice FAISS e persiste localmente para reuso.
    vector_store = FAISS.from_documents(filtered_documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

    return vector_store


def results_by_chromadb(company_documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """
    Cria (ou recria) um índice ChromaDB persistente.

    Args:
        company_documents: Lista de documentos para indexação.
        embeddings: Modelo de embeddings usado para vetorizar textos.

    Returns:
        Repositório Chroma pronto para busca e persistência em disco.
    """

    # Remove metadados complexos para garantir serialização correta.
    filtered_documents = filter_complex_metadata(company_documents)

    # Pasta persistente do ChromaDB.
    persist_directory = "./chroma_db"
    should_reset = str(get_env_var("CHROMA_RESET", "false")).lower() in {"1", "true", "yes"}
    if should_reset and os.path.isdir(persist_directory):
        # Remove o índice para recriação limpa (útil em desenvolvimento).
        shutil.rmtree(persist_directory)

    # Cria e persiste o índice localmente.
    vector_store = Chroma.from_documents(filtered_documents, embeddings, persist_directory=persist_directory)

    return vector_store


def results_by_pinecone(company_documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Pinecone:
    """
    Cria e popula um índice Pinecone gerenciado (SaaS).

    Args:
        company_documents: Lista de documentos para indexação.
        embeddings: Modelo de embeddings usado para vetorizar textos.

    Returns:
        Repositório Pinecone pronto para busca por similaridade.
    """

    # Remove metadados complexos que o Pinecone não suporta.
    filtered_documents = filter_complex_metadata(company_documents)

    # Recupera credencial e define nome do índice no Pinecone.
    PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
    index_name = "pinecone-poc"  # Nome do índice criado no painel do Pinecone

    # Cliente oficial para gerenciar índices.
    pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)

    # Especificação do índice serverless (nuvem/região).
    spec = ServerlessSpec(
        cloud="aws",  # Nuvem onde o índice será hospedado (gcp, aws ou azure)
        region="us-east-1"  # Região onde o índice será hospedado.
    )

    # Remove índice existente para garantir dimensão correta.
    if index_name in pinecone_client.list_indexes().names():
        pinecone_client.delete_index(index_name)
        print(f"Índice '{index_name}' deletado.")

    # Cria novo índice com a dimensão do modelo escolhido.
    pinecone_client.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=spec
    )
    print(f"Índice '{index_name}' criado com sucesso no Pinecone.")

    # Popula o índice com documentos vetorizados.
    pinecone_db = Pinecone.from_documents(
        documents=filtered_documents,
        embedding=embeddings,
        index_name=index_name,
    )
    print(f"Documentos inseridos no índice '{index_name}' com sucesso.")

    return pinecone_db
