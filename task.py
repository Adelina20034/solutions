from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_agent
from langchain.schema.messages import SystemMessage
from langchain.tools import tool
from pydantic import SecretStr
import pandas as pd

# Локальная модель через OpenAI совместимый API
llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url='http://localhost:1234/v1',
    api_key=SecretStr('fake'),
    temperature=0.7,
)

# Инструмент get_price с субагентом внутри
@tool
def get_price(product: str, city: str) -> str:
    """
    Получение цены на указанный продукт в конкретном городе.
    Возвращает таблицу с ценами и магазинами.
    """
    subagent = create_agent(
        model=llm,
        tools=[],
        system_prompt=f"Генерируй реалистичные цены на {product} в {city}. Верни таблицу.",
    )
    answer = subagent.invoke({"messages": [{"role": "human", "content": product}]})
    df = pd.read_json(answer['messages'][0].content)
    return df.to_markdown(index=False)

# Список продуктов для примера
products = ["молоко", "хлеб", "яблоки"]
city = "Казань"

# Формирование запроса пользователю
query = f"Помогите составить список покупок: {', '.join(products)}. Я нахожусь в {city}."

# Основной агент
system_prompt = "Ты помощник по планированию покупок."
tools = [get_price]
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

# Выполнение запроса
response = agent.invoke({"messages": [SystemMessage(content=query)]})
final_answer = response['messages'][-1].content

print(final_answer)