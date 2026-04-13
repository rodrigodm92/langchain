import httpx
from typing import Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_URL = "http://127.0.0.1:1234/v1/"
MODEL = "qwen2.5-coder-7b-instruct"

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

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra Praia. Você é uma especialista em viagens com destinos para praia."),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr Montanha. Você é uma especialista em viagens com destinos para montanhas e atividades radicais."),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]
    
prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}")
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)

def responda (pergunta :str):
    rota = roteador.invoke({"query":pergunta})['destino']
    print(rota)
    if rota == "praia":
        return cadeia_praia.invoke({"query":pergunta})
    return cadeia_montanha.invoke({"query":pergunta})

print(responda("Quero passear por praias belas no Brasil."))