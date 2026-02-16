from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import duckdb
import pandas as pd
from datetime import datetime


def etl_pdf_process() -> list[Document]:
    """
    O UnstructuredPDFLoader é projetado para entender a estrutura de um PDF, como títulos, parágrafos, tabelas e imagens.
    Ele pode extrair o conteúdo de forma organizada, mantendo a hierarquia e a formatação do documento.
    Isso é especialmente útil para documentos complexos, onde a estrutura é importante para a compreensão do conteúdo.
    Ele é ótimo para chunking, pois pode dividir o conteúdo em partes menores, como seções ou parágrafos, facilitando a análise
    e o processamento posterior.
    """

    # Extração de dados
    loader = UnstructuredPDFLoader("chatbot_com_rag/assets/relatorio_vendas.pdf", mode="elements")
    docs_unstructed = loader.load()
    print("Total de elementos extraídos:", len(docs_unstructed))

    # Transformação de dados (metadados adicionais)
    docs_with_metadata = []
    for i, doc in enumerate(docs_unstructed):
        metadata = {
            "id_doc": f"doc{i + 1}",
            "source": "relatorio_vendas.pdf",
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "data_owner": "Departamento de Vendas",
            **doc.metadata.copy()
        }
        docs_with_metadata.append(Document(page_content=doc.page_content, metadata=metadata))

    print("Total de documentos com metadata:", len(docs_with_metadata))

    # Transformação de dados (chunking)
    # Agora que temos os elementos do PDF, precisamos dividi-los em partes menores para o banco vetorial.
    # Garantindo que não ultrapassem o limite de contexto do modelo de embeddings, que geralmente é de 512 a 1024 tokens.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Tamanho ideal para embeddings, considerando a média de tokens por palavra e a necessidade de contexto.
        chunk_overlap=50,  # Sobreposição para manter o contexto entre os chunks, evitando perda de informações importantes.
    )

    chuncks = text_splitter.split_documents(docs_with_metadata)
    print("Total de chunks gerados:", len(chuncks))

    return chuncks


def etl_db_process():
    """
    O DuckDB é um banco de dados analítico embutido que pode ser usado para processar grandes volumes de dados de forma eficiente.
    """

    # Extração de dados
    connection = duckdb.connect(database=':memory:')
    connection.execute("""
        CREATE TABLE produtos (
            id INTEGER,
            nome VARCHAR,
            categoria VARCHAR,
            preco FLOAT,
            estoque INTEGER,
            descricao TEXT
        );
    """)

    produtos_df = pd.DataFrame({
        'id': [1, 2, 3],
        'nome': ['Produto A', 'Produto B', 'Produto C'],
        'categoria': ['Eletrônicos', 'Roupas', 'Alimentos'],
        'preco': [199.99, 49.99, 9.99],
        'estoque': [100, 200, 300],
        'descricao': [
            'Um smartphone de última geração com tela AMOLED e câmera de alta resolução.',
            'Uma camiseta de algodão confortável disponível em várias cores.',
            'Um pacote de biscoitos deliciosos feitos com ingredientes naturais.'
        ]
    })

    connection.register("produtos_df", produtos_df)
    connection.execute("INSERT INTO produtos SELECT * FROM produtos_df")

    print("Tabela 'produtos' criada e populada no DuckDB.")

    # Transformação de dados (metadados adicionais)
    df_products = connection.execute("SELECT * FROM produtos").fetchdf()

    docs_with_metadata = []
    for _, row in df_products.iterrows():
        metadata = {
            "id_doc": f"produto_{row['id']}",
            "source": "tabela_produtos_duckdb",
            "id_produto": row['id'],
            "categoria": row['categoria'],
            "preco": row['preco'],
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "data_owner": "Departamento de Vendas"
        }
        docs_with_metadata.append(Document(page_content=row['descricao'], metadata=metadata))

    print("Total de documentos com metadata:", len(docs_with_metadata))

    connection.close()

    # Transformação de dados (chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chuncks = text_splitter.split_documents(docs_with_metadata)
    print("Total de chunks gerados:", len(chuncks))

    return chuncks
