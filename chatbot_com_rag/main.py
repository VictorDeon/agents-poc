from utils import load_environment_variables, get_env_var
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

load_environment_variables()


def main():
    GEMINI_API_KEY = get_env_var('GEMINI_API_KEY')

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        api_key=GEMINI_API_KEY
    )

    question = "Qual é a política de home office na nossa empresa?"

    prompt = ChatPromptTemplate.from_template(
        "Responda a seguinte pergunta: {question}"
    )

    chain = prompt | llm

    response = chain.invoke({"question": question})
    print(response)


if __name__ == "__main__":
    main()