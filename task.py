from typing import List, Dict
from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain.schema.messages import SystemMessage
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.stdout_callback_handler import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

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
    subagent_prompt_template = (
        "Представь себя экспертом по ценам на продукты питания.\n"
        "Пользователь запрашивает информацию о цене {product} в городе {city}.\n"
        "Используя свои знания исторических данных о ценах,\n"
        "предоставь следующую таблицу:\n\n"
        "| Продукт   | Цена (руб.) | Магазин |\n"
        "|-----------|-------------|---------|\n"
        "{{ product }} | {{ price }} | {{ store }}\n"
    )
    prompt = PromptTemplate.from_template(subagent_prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"product": product, "city": city})
    return response.strip()

# Список доступных инструментов
tools = [get_price]

# Система подсказка для основного агента
system_prompt = SystemMessage(content="Ты помощник по планированию покупок.")

# Основной агент
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": system_prompt},
)

# Вопрос пользователю
question = {
    "messages": [
        {"role": "human", "content": "Помоги составить список покупок: молоко, хлеб, яблоки. Я нахожусь в Казани."}
    ]
}

# Выполнение запроса
response = agent_executor.invoke(question)

# Форматирование вывода всех сообщений
for msg in response.get("intermediate_steps", []):
    print(f"--- {msg}")
print("\nФинальный ответ:")
print(response["output"])