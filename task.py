from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_agent
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
    sub_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=f"Генерируй реалистичные цены на {product} в городе {city}. Верни результат в формате таблицы.",
    )
    answer = sub_agent.invoke({"messages": [{"role": "system", "content": ""}]})
    df = pd.read_json(answer['messages'][0].content, orient='records')
    return df.to_markdown(index=False)

# Основной агент
agent = create_agent(
    model=llm,
    tools=[get_price],
    system_prompt="Ты помощник по планированию покупок."
)

# Функция вывода всех шагов выполнения запроса
def print_messages(messages: List[Dict]):
    for msg in messages:
        content = msg.get('content')
        if content is not None and len(content.strip()) > 0:
            print(f"\n---\n{content}\n---")
        else:
            tool_call = msg.get('tool_calls')
            if tool_call:
                name = tool_call[0]['function']['name']
                args = tool_call[0]['function'].get('arguments', {})
                print(f"\n---\n{name}({args})\n---")

# Тестовый запрос
question = {
    "messages": [
        {"role": "human", "content": "Помоги составить список покупок: молоко, хлеб, яблоки. Я нахожусь в Казани."},
    ],
}

# Выполнение запроса
response = agent.invoke(question)
print_messages(response['intermediate_steps'])
final_answer = response['messages'][-1]
print("\nФинальный ответ:\n", final_answer.content)