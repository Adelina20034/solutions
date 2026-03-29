from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools import tool
from langchain.schema.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
import pandas as pd


llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url='http://localhost:1234/v1',
    api_key=SecretStr('fake'),
    temperature=0.7,
)


@tool
def get_price(product: str, city: str) -> str:
    """
    Получение цены на указанный продукт в конкретном городе.
    Возвращает таблицу с ценами и магазинами.
    """
    df = pd.DataFrame([
        {'Продукт': product, 'Цена (руб.)': '89', 'Магазин': 'Магнит'},
        {'Продукт': product, 'Цена (руб.)': '45', 'Магазин': 'Пятёрочка'},
        {'Продукт': product, 'Цена (руб.)': '120/кг', 'Магазин': 'Перекрёсток'}
    ])
    return df.to_markdown(index=False)


tools = [get_price]
system_prompt = 'Ты помощник по планированию покупок.'

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent='zero-shot-react-description',
    verbose=True,
    handle_parsing_errors=True,
    system_message=SystemMessage(content=system_prompt)
)

question = "Помоги составить список покупок: молоко, хлеб, яблоки. Я нахожусь в Казани."
response = agent_executor.invoke({"input": question})
print(response['output'])