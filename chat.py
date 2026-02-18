from agent import Agent
from rich import print
from rich.markdown import Markdown


def main(question: str):
    """
    Orquestra todo o fluxo de RAG, do ETL à resposta da pergunta.
    """

    agent = Agent.get_instance(session_id="default")

    try:
        response = agent.invoke(question)
        print(Markdown(response))
        print(Markdown("---"))
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
        if question.lower() in {"sair", "exit", "quit", "q"}:
            print("Encerrando o chatbot.")
            break

        main(question)