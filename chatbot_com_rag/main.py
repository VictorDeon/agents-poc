"""
Resumo simples do que o código faz:

1. Carrega a chave da API do Gemini do ambiente.
2. Cria um modelo de embeddings (para transformar texto em vetores).
3. Cria um modelo de chat (para responder perguntas).
4. Valida a entrada e a saída para evitar conteúdo inseguro (ex.: chaves de API).
5. Lê dados de PDF e de um “banco” (ETL), junta tudo e adiciona metadados padrão.
6. Indexa os documentos em um banco vetorial (Chroma) para busca semântica.
7. Quando recebe uma pergunta, ela é reescrita considerando o histórico.
8. Busca os documentos mais relevantes.
9. Monta um prompt com esses documentos e gera a resposta.
10. Mostra a resposta e quantos documentos foram usados.
"""

from utils import load_environment_variables, get_env_var
from guardrails_security import GuardrailsSecurity
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from vetorial_db import results_by_chromadb
from etls import etl_pdf_process

# Carrega variáveis de ambiente (ex.: chaves de API) antes de usar qualquer SDK.
load_environment_variables()

# Mantém histórico entre perguntas na mesma execução.
session_store: dict[str, InMemoryChatMessageHistory] = {}
_chat_chain: RunnableWithMessageHistory | None = None
_guardrails: GuardrailsSecurity | None = None


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retorna (ou cria) o histórico da sessão informada.
    """

    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()

    return session_store[session_id]


def _get_chat_chain() -> RunnableWithMessageHistory:
    """
    Cria (uma vez) e retorna a cadeia RAG com memória.
    """

    global _chat_chain
    if _chat_chain is not None:
        return _chat_chain

    # Recupera a chave de API do Gemini do ambiente. Falha cedo se ausente.
    GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')

    # Instancia o modelo de embeddings do Google para vetorizar texto.
    # Esses vetores são necessários para busca semântica no banco vetorial.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",  # Modelo específico de embeddings.
        google_api_key=GEMINI_API_KEY  # Credencial exigida pela API.
    )

    # Instancia o LLM para respostas e para reescrever a pergunta com contexto.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # Modelo leve/rápido para conversação.
        temperature=0.1,  # Baixa aleatoriedade para respostas mais consistentes.
        api_key=GEMINI_API_KEY  # Credencial exigida pela API.
    )

    # Executa ETL dos PDFs (pode usar o LLM para limpeza/extração).
    documents = etl_pdf_process(llm)

    # Metadados obrigatórios para o prompt dos documentos.
    # Valores padrão evitam KeyError quando a fonte não fornece algum campo.
    required_metadata_defaults = {
        "id_doc": "N/A",
        "source": "N/A",
        "page_number": "N/A",
        "categoria": "N/A",
        "id_produto": "N/A",
        "preco": "N/A",
        "timestamp": "N/A",
        "data_owner": "N/A",
    }

    # Normaliza metadados (inclui defaults e converte tipos não serializáveis).
    for doc in documents:
        metadata = doc.metadata or {}
        for key, default_value in required_metadata_defaults.items():
            if key not in metadata:
                metadata[key] = default_value
            else:
                value = metadata[key]
                # Converte objetos tipo NumPy scalar em tipos Python nativos.
                if hasattr(value, "item"):
                    metadata[key] = value.item()

        doc.metadata = metadata

    # Log simples para visibilidade do volume indexado.
    # print("Total de documentos para indexação:", len(documents))

    # Indexa documentos no ChromaDB e retorna o repositório vetorial.
    vector_store = results_by_chromadb(documents, embeddings)

    # Prompt para reescrever a pergunta com base no histórico (sem responder).
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Dado o histórico da conversa e a pergunta do usuário, reformule a pergunta para uma versão independente, "
            "sem perder o contexto. NÃO responda a pergunta."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Prompt principal de QA: usa contexto recuperado e histórico de conversa.
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

    # Prompt que define como cada documento aparece no contexto da resposta.
    document_prompt = PromptTemplate.from_template(
        "Fonte: {source}\n"
        "Página: {page_number}\n"
        "ID: {id_doc}\n"
        "Categoria: {categoria}\n"
        "Produto: {id_produto}\n"
        "Preço: {preco}\n"
        "Data: {timestamp}\n"
        "Dono: {data_owner}\n"
        "Conteúdo:\n{page_content}"
    )

    # Recuperador semântico com top-k documentos mais relevantes.
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Lexical retriever (BM25) para complementar a busca semântica, especialmente útil para termos específicos.
    lexical_retriever = BM25Retriever.from_documents(documents)
    lexical_retriever.k = 5  # Configura para retornar os 5 documentos mais relevantes.

    # Fazer o merge dos resultados dos dois recuperadores (semântico + lexical) para melhorar a cobertura.
    # O EnsembleRetriever combina os resultados de ambos, dando mais peso ao semântico.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, lexical_retriever],
        weights=[0.7, 0.3]  # Dá mais peso ao recuperador semântico, mas ainda considera o lexical.
    )

    # Recuperador que reescreve a pergunta considerando o histórico.
    history_aware_retriever = create_history_aware_retriever(llm, ensemble_retriever, contextualize_q_prompt)

    # Cadeia de QA que insere documentos no prompt de resposta.
    # Junta os documentos no prompt e faz a resposta
    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
        document_prompt=document_prompt,
        document_variable_name="context"  # Nome esperado no prompt {context}.
    )

    # Encadeia recuperação + resposta para formar o pipeline RAG.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Wrapper que injeta histórico e registra novas mensagens automaticamente.
    _chat_chain = RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",  # Chave do texto da pergunta.
        history_messages_key="chat_history",  # Onde o histórico é lido/escrito.
        output_messages_key="answer",  # Campo de resposta no output.
    )

    return _chat_chain


def main(question: str):
    """
    Orquestra todo o fluxo de RAG, do ETL à resposta da pergunta.
    """

    global _guardrails
    if _guardrails is None:
        # Guardrails de segurança para validar entrada e saída.
        _guardrails = GuardrailsSecurity()

    chat_chain = _get_chat_chain()

    # Pergunta de exemplo (pode ser substituída por entrada do usuário).
    try:
        question = _guardrails.validate_input(question)
        # Executa o pipeline RAG com uma sessão fixa ("default").
        response = chat_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "default"}}
        )
        # Resposta final do modelo com base nos documentos recuperados.
        safe_answer = _guardrails.validate_output(response['answer'])
        print(f"Resposta: {safe_answer}")
        # Documentos usados na resposta para auditoria/inspeção.
        # print(f"\nDocumentos utilizados: {len(response['context'])}")
    except Exception as e:
        # Captura qualquer erro de execução para facilitar debug.
        print(f"Erro ao processar a pergunta: {e}")


if __name__ == "__main__":
    """
    Exemplo de perguntas:
    - "Qual é o preço do Produto A?"
    """

    # Ponto de entrada para execução via CLI.
    print("Chatbot iniciado. Digite sua pergunta ou 'sair' para encerrar.")
    while True:
        question = input("\nPergunta: ").strip()
        if not question:
            print("Por favor, digite uma pergunta válida.")
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando o chatbot.")
            break

        main(question)