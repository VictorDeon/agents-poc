from agent import Agent


def main(question: str):
    """
    Orquestra todo o fluxo de RAG, do ETL à resposta da pergunta.
    """

    agent = Agent(session_id="default")

    try:
        response = agent.invoke(question)
        print(f"Resposta: {response}")
    except Exception as e:
        print(f"Erro ao processar a pergunta: {e}")


if __name__ == "__main__":
    """
    Exemplo de perguntas:
    - "O que é RAG"
    - "para que ela serve?"
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