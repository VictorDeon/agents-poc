from utils import load_environment_variables, get_env_var
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_pinecone import Pinecone
from pinecone import ServerlessSpec, Pinecone as PineconeClient
from langchain.schema import Document
import faiss

load_environment_variables()


def results_by_faissdb(company_documents, embeddings):
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

    dimension = 768  # Dimensão dos embeddings do modelo gemini-embedding-001 (768 valores por vetor)
    neighbors = 32  # Número de vizinhos a serem considerados para consultas de similaridade
    index = faiss.IndexHNSWFlat(dimension, neighbors)  # Usando HNSW para melhor desempenho em grandes volumes de dados

    faiss_db = FAISS.from_documents(company_documents, embeddings)

    question = "Como peço minhas férias na empresa?"
    # Busca os 2 documentos mais parecidos para a pergunta usando o índice FAISS
    results = faiss_db.similarity_search(question, k=2)
    print("Documentos mais relevantes para a pergunta:")
    for doc in results:
        print(f"- {doc.metadata['id_doc']}: {doc.page_content}")


def results_by_chromadb(company_documents, embeddings):
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

    chroma_db = Chroma.from_documents(company_documents, embeddings)

    question = "Quais são as regras da empresa?"
    # Busca os 2 documentos mais parecidos para a pergunta usando o índice Chroma
    results = chroma_db.similarity_search(question, k=2, filter={"$and": [{"departamento": "RH"}, {"tipo": "política"}]})
    print("Documentos mais relevantes para a pergunta:")
    for doc in results:
        print(f"- {doc.metadata['id_doc']}: {doc.page_content}")


def results_by_pinecone(company_documents, embeddings):
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
        documents=company_documents,
        embedding=embeddings,
        index_name=index_name,
    )
    print(f"Documentos inseridos no índice '{index_name}' com sucesso.")

    if pinecone_db:
        # Busca por similaridade
        question = "Como eu configuro minha VPN?"
        results = pinecone_db.similarity_search(question, k=2)
        print("Documentos mais relevantes para a pergunta:")
        for doc in results:
            print(f"- {doc.metadata['id_doc']}: {doc.page_content}")
    else:
        print("Erro ao conectar ao índice do Pinecone. Verifique as configurações e tente novamente.")


def main():
    GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')

    # Instanciando um modelo de embeddings do google
    # permitindo transformar texto em vetores numéricos
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    # Verificar a dimensão dos embeddings
    test_embedding = embeddings.embed_query("test")
    print(f"Dimensão dos embeddings: {len(test_embedding)}")
    # No ambiente real seria preenchido por um pdf ou algum outro tipo de documento.
    company_documents = [
        Document(
            page_content="Política de férias: Funcionários têm direito a 30 dias de férias após 12 meses. A solicitação deve ser feita com 30 dias de antecedência.",
            metadata={"tipo": "política", "departamento": "RH", "ano": 2024, "id_doc": "doc001"}
        ),
        Document(
            page_content="Processo de reembolso de despesas: Envie a nota fiscal pelo portal financeiro. O reembolso ocorre em até 5 dias úteis.",
            metadata={"tipo": "processo", "departamento": "Financeiro", "ano": 2023, "id_doc": "doc002"}
        ),
        Document(
            page_content="Guia de TI: Para configurar a VPN, acesse vpn.nossaempresa.com e siga as instruções para seu sistema operacional.",
            metadata={"tipo": "tutorial", "departamento": "TI", "ano": 2024, "id_doc": "doc003"}
        ),
        Document(
            page_content="Código de Ética e Conduta: Valorizamos o respeito, a integridade e a colaboração. Casos de assédio não serão tolerados.",
            metadata={"tipo": "política", "departamento": "RH", "ano": 2022, "id_doc": "doc004"}
        )
    ]

    # results_by_faissdb(company_documents, embeddings)
    # results_by_chromadb(company_documents, embeddings)
    results_by_pinecone(company_documents, embeddings)

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash-lite",
    #     temperature=0,
    #     api_key=GEMINI_API_KEY
    # )


if __name__ == "__main__":
    main()