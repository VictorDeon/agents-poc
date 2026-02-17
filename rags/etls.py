from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils import get_prompt
from datetime import datetime
from pathlib import Path


def etl_pdf_process(llm: ChatGoogleGenerativeAI | None = None) -> list[Document]:
    """
    Extrai e transforma documentos de PDF em chunks com metadados.

    Args:
        llm: LLM opcional para gerar um resumo do PDF e adicionar como documento extra.

    Returns:
        Lista de documentos prontos para indexação.
    """

    loader = PyPDFDirectoryLoader("chatbot_com_rag/aulas", glob="*.pdf")
    docs = loader.load()  # Carrega todos os PDFs da pasta.

    # Transformação de dados (metadados adicionais por página).
    docs_with_metadata: list[Document] = []
    for i, doc in enumerate(docs):
        # Normaliza numeração de páginas para iniciar em 1.
        page_number = (doc.metadata.get("page", i) or i) + 1
        metadata = {
            "id_doc": f"doc{i + 1}",
            "source": doc.metadata.get("source", "N/A"),
            "page_number": page_number,
            "categoria": "N/A",
            "id_produto": "N/A",
            "preco": "N/A",
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "data_owner": "Departamento de Vendas",
            **doc.metadata.copy()
        }
        # Cabeçalho textual facilita rastreamento do trecho na resposta.
        page_header = f"[Relatório de Vendas | Página {page_number}]\n"
        docs_with_metadata.append(
            Document(page_content=f"{page_header}{doc.page_content}", metadata=metadata)
        )

    # Transformação de dados (chunking)
    # Dividimos o texto para respeitar limites de contexto dos embeddings.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Mais contexto por chunk para preservar trechos inteiros do PDF.
        chunk_overlap=200,  # Sobreposição para manter continuidade entre trechos.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(docs_with_metadata)
    # print("Total de chunks gerados:", len(chunks))

    # Resumo opcional do PDF para fornecer visão geral ao modelo.
    summary_chunks = []
    if llm is not None and chunks:
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", get_prompt("document_summary.prompt.md")),
            ("human", "{doc_content}")
        ])

        chain = summary_prompt | llm
        summaries = chain.batch([{"doc_content": chunk.page_content} for chunk in chunks])
        for summary in summaries:
            summary_text = summary.content.strip()
            summary_metadata = {
                "id_doc": "pdf_summary",
                "source": doc.metadata.get("source", "N/A"),
                "page_number": 1,
                "categoria": "N/A",
                "id_produto": "N/A",
                "preco": "N/A",
                "timestamp": datetime.now().strftime("%Y-%m-%d"),
                "data_owner": "Departamento de Vendas",
                "type": "summary"
            }
            summary_chunks.append(
                Document(page_content=f"[Resumo do PDF]\n{summary_text}", metadata=summary_metadata)
            )
    else:
        summary_chunks = chunks.copy()  # Se não houver LLM, usamos os chunks originais sem resumo.

    return summary_chunks


def etl_text_process() -> list[Document]:
    """
    Extrai e transforma documentos Markdown/TXT em chunks com metadados.

    Returns:
        Lista de documentos prontos para indexação.
    """

    base_dir = Path("chatbot_com_rag/assets")
    file_paths = sorted(list(base_dir.glob("*.txt")))

    if not file_paths:
        return []

    docs_with_metadata: list[Document] = []
    for file_path in file_paths:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        for i, doc in enumerate(docs):
            metadata = {
                "id_doc": f"text_{file_path.stem}_{i + 1}",
                "source": file_path.name,
                "page_number": "N/A",
                "categoria": "N/A",
                "id_produto": "N/A",
                "preco": "N/A",
                "timestamp": datetime.now().strftime("%Y-%m-%d"),
                "data_owner": "Departamento de Vendas",
                **doc.metadata.copy()
            }
            file_header = f"[Arquivo: {file_path.name}]\n"
            docs_with_metadata.append(
                Document(page_content=f"{file_header}{doc.page_content}", metadata=metadata)
            )

    # Transformação de dados (chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(docs_with_metadata)

    return chunks
