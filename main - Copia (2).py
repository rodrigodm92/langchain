import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

BASE_URL = "http://127.0.0.1:1234/v1/"
MODEL = "qwen2.5-coder-7b-instruct"

numero_dias = 7
numero_criancas = 2
atividade = "praia"

modelo_de_prompt = PromptTemplate(
    template="""
    Crie um roteiro de viagem de {dias} dias para uma familia com {numero_criancas} criancas que gosta de {atividade}.
    """
)

prompt = modelo_de_prompt.format(dias=numero_dias, numero_criancas=numero_criancas, atividade=atividade)

print("Prompt: \n", prompt)

modelo = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key="lm-studio",
    http_client=httpx.Client(
        transport=httpx.HTTPTransport(local_address="127.0.0.1"),
        trust_env=False,
    ),
    temperature=0.5,
)

resposta = modelo.invoke(prompt)
print(resposta.content)
