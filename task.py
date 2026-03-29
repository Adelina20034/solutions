from typing import List, Dict
from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools import BaseTool, StructuredTool
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.callbacks.stdout_callback import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import pandas as pd

class GetPriceTool(BaseTool):
    name = "get_price"
    description = (
        "Используется для получения цены конкретного товара в конкретном городе."
    )

    def _run(self, product: str, city: str) -> str:
        df = pd.DataFrame([
            {'Продукт': product, 'Цена (руб.)': self._generate_random_price(), 'Магазин': self._random_store()}
        ])
        return df.to_markdown(index=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented.")

    @staticmethod
    def _generate_random_price():
        return round(float(f'{int(60 + 100*abs(hash(str((id(GetPriceTool), id(pd))))) % 100):.{2}f}'), 2)

    @staticmethod
    def _random_store():
        stores = ["Магнит", "Пятёрочка", "Перекрёсток"]
        return stores[int(abs(hash(str(id(stores)))) % len(stores))]

tools = [GetPriceTool()]

llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url='http://localhost:1234/v1',
    api_key=SecretStr('fake'),
    temperature=0.7,
)

system_prompt = """
Ты помощник по планированию покупок. 
Пользователь задает запрос со списком товаров и городом, а ты предоставляешь информацию о стоимости этих товаров в данном городе.
"""

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}

agent_executor = initialize_agent(
    tools,
    llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=None,
    agent_kwargs=agent_kwargs,
    handle_parsing_errors=True,
)

question = "Помоги составить список покупок: молоко, хлеб, яблоки. Я нахожусь в Казани."

response = agent_executor.run(question)
print(response)