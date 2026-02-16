from utils import load_environment_variables, get_env_var
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_pinecone import Pinecone
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from pinecone import ServerlessSpec, Pinecone as PineconeClient
from langchain.schema import Document
import faiss

load_environment_variables()


def results_by_cache(embeddings: GoogleGenerativeAIEmbeddings) -> CacheBackedEmbeddings:
    """
    O Cache de Embeddings: é uma camada de armazenamento intermediária que armazena os vetores de embeddings
    gerados a partir de textos ou outros dados. Ele é projetado para acelerar o processo de recuperação de
    informações, evitando a necessidade de recalcular os embeddings para os mesmos dados repetidamente.

    Funciona da seguinte maneira:
    1. Geração de Embeddings: Quando um texto é processado pela primeira vez, o modelo de embeddings gera um vetor numérico
    que representa o significado do texto.
    2. Armazenamento em Cache: Esse vetor é então armazenado em um cache (que pode ser na memória, em um
    banco de dados ou em um sistema de arquivos).
    3. Recuperação Rápida: Na próxima vez que o mesmo texto for processado, o sistema verifica primeiro o cache.
    Se o vetor já estiver lá, ele é recuperado imediatamente, economizando tempo e recursos computacionais.
    4. Atualização do Cache: Se o texto for modificado ou se um novo texto for processado, um novo vetor
    será gerado e armazenado no cache para futuras consultas.

    O uso do cache de embeddings é especialmente benéfico em aplicações onde os mesmos textos são processados repetidamente,
    como em sistemas de busca, chatbots ou qualquer cenário onde a eficiência na recuperação de informações seja crucial.
    Ele ajuda a melhorar significativamente a performance e a escalabilidade da aplicação.
    """

    store = LocalFileStore("./embeddings_cache")
    cached_embeddings: CacheBackedEmbeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace="gemini-embedding-cache"
    )

    return cached_embeddings


def results_by_faissdb(company_documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """
    O que é o FAISS: É uma biblioteca de código para fazer buscas de similaridade
    muito rápidas. Dado um item, ela encontra os mais parecidos dentro de um grande volume de dados.

    O FAISS oferece principalmente duas estratégias de busca:
    1. Índice Flat (Busca Exata):
        - Como funciona: Compara o seu item de busca com todos os outros itens no banco de dados, um por um.
        - Vantagem: Precisão de 100%. Você tem a garantia de encontrar os resultados exatos.
        - Desvantagem: Lento para muitos dados, pois o número de comparações é enorme.

    2. Índice HNSW (Busca Aproximada):
        - Como funciona: Cria uma estrutura de dados otimizada (um grafo) que permite pular comparações desnecessárias.
        A busca é guiada de forma inteligente para a área mais provável dos resultados.
        - Vantagem: Muito mais rápido, ideal para aplicações em tempo real com grandes volumes de dados.
        - Desvantagem: A precisão não é 100%. A busca é muito boa, mas pode raramente deixar de fora o resultado mais exato.
        Essa perda é geralmente aceitável em troca de velocidade.

    Por padrão o LangChain usa o método Flat. Ele escolhe a segurança (100% de precisão) em vez
    da velocidade, pois é mais simples e confiável para projetos iniciais e pequenos.
    """

    filtered_documents = filter_complex_metadata(company_documents)

    dimension = 768  # Dimensão dos embeddings do modelo gemini-embedding-001 (768 valores por vetor)
    neighbors = 32  # Número de vizinhos a serem considerados para consultas de similaridade
    faiss.IndexHNSWFlat(dimension, neighbors)  # Usando HNSW para melhor desempenho em grandes volumes de dados

    vector_store = FAISS.from_documents(filtered_documents, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Salva o índice FAISS localmente para persistência e reutilização futura

    return vector_store


def results_by_chromadb(company_documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """
    O Chroma DB: O Banco de Dados Vetorial Simples e Poderoso para IA
    Chroma é um banco de dados vetorial open-source, criado especialmente para
    ser intuitivo e fácil de usar em aplicações de IA.

    Seu grande diferencial é a combinação de duas funções essenciais:
    1. Busca por Similaridade: Armazena e busca vetores (a representação numérica de textos, imagens e etc.)
    para encontrar itens parecidos com base em seus significados ou conteúdos.
    2. Filtros por Metadados: Permite associar informações adicionais (como datas, categorias, fontes, IDs de usuários)
    a cada vetor. Com isso, você pode refinar suas buscas de forma precisa.

    Na prática, isso permite consultas complexas como: "Encontre documentos parecidos com este texto sobre finanças,
    mas que forma publicados apenas no último mês e que pertencem à categoria 'notícias'"
    """

    filtered_documents = filter_complex_metadata(company_documents)

    vector_store = Chroma.from_documents(filtered_documents, embeddings, persist_directory="./chroma_db")
    vector_store.persist()  # Salva os vetores e metadados no armazenamento do Chroma

    return vector_store


def results_by_pinecone(company_documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Pinecone:
    """
    O Pinecone: é um banco de dados vetorial de nível profissional, oferecido como
    um serviço na nuvem (SaaS). totalmente gerenciado. Sua proposta é eliminar toda
    a complexidade de infraestrutura, permitindo que as equipes foquem exclusivamente
    no desenvolvimento da aplicação.

    É a escolha ideal para cenários que exigem:
    1. Alta Escalabilidade: Projetado para crescer junto com sua aplicação suportando
    bilhões de vetores e um alto volume de buscas sem degradação de performace.
    2. Disponibilidade e Confiabilidade: Garante que o banco de dados esteja sempre online,
    otimizado e seguro, algo crítico para produtos em produção.
    3. Zero Manutenção: Você não precisa se preocupar com servidores, atualizações, backups ou otimizações.
    O Pinecone cuida de tudo.

    Em resumo, você obtém a potência de um banco vetorial de ponta através de uma simples API, sem os custos
    e o trabalho de gerencia-lo,

    Pré requisitos para conectar: Para usar o ponecone neste código você precisa:
    1. Configurar sua chave de API: Garantir que a variavel de ambiente
    PINECONE_API_KEY esteja definida com sua chave de API do Pinecone.
    2. Criar um índice no Pinecone: Antes de executar o código, crie um
    índice no painel do Pinecone, escolhendo o nome, a dimensão (768 para os
    embeddings do modelo gemini-embedding-001) e a configuração de replicação
    desejada.
    """

    filtered_documents = filter_complex_metadata(company_documents)

    PINECONE_API_KEY = get_env_var('PINECONE_API_KEY')
    index_name = "pinecone-poc"  # Nome do índice criado no painel do Pinecone

    pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)

    spec = ServerlessSpec(
        cloud="aws",  # Nuvem onde o índice será hospedado (gcp, aws ou azure)
        region="us-east-1"  # Região onde o índice será hospedado (ex: us-west1, europe-west1, etc.)
    )

    # Delete the index if it exists to ensure correct dimension
    if index_name in pinecone_client.list_indexes().names():
        pinecone_client.delete_index(index_name)
        print(f"Índice '{index_name}' deletado.")

    # Create new index
    pinecone_client.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=spec
    )
    print(f"Índice '{index_name}' criado com sucesso no Pinecone.")

    pinecone_db = Pinecone.from_documents(
        documents=filtered_documents,
        embedding=embeddings,
        index_name=index_name,
    )
    print(f"Documentos inseridos no índice '{index_name}' com sucesso.")

    return pinecone_db
