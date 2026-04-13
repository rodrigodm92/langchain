import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_URL = "http://127.0.0.1:1234/v1/"
MODEL = "qwen2.5-coder-7b-instruct"

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    """,
    input_variables=["interesse"]
)

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

cadeia = prompt_cidade | modelo | StrOutputParser()

resposta = cadeia.invoke(
    {
        "interesse" : "praia"
    }
)
print(resposta)
