import httpx
from openai import OpenAI, DefaultHttpxClient
from dotenv import load_dotenv
import os

BASE_URL = "http://127.0.0.1:1234/v1/"
MODEL = "qwen2.5-coder-7b-instruct"


cliente = OpenAI(
    base_url = BASE_URL,
    api_key="lm-studio",
    http_client=DefaultHttpxClient(
        transport=httpx.HTTPTransport(local_address="127.0.0.1"),
        trust_env=False
    )
)

numero_dias = 7
numero_criancas = 2
atividade = "música"

prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma família com {numero_criancas} crianças, que gosta de {atividade}"

resposta = cliente.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Você é um assistente de roteiro de viagens."},
        {"role": "user", "content": prompt}
    ]
)

print(resposta.choices[0].message.content)