from typing import List, Dict
from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools import BaseTool, StructuredTool
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

class GetPriceTool(BaseTool):
    name = "get_price"
    description = (
        "Используется для получения цены конкретного товара в конкретном городе."
    )

    def _run(self, product: str, city: str, run_manager: CallbackManagerForToolRun = None) -> str:
        sub_llm = ChatOpenAI(model='gpt-3.5-turbo', base_url='http://localhost:1234/v1', api_key=SecretStr('fake'), temperature=0.7)
        prompt_template = """
            Ты эксперт по ценам товаров в разных городах России.
            Назови среднюю цену {product} в городе {city}.
            Форматируй ответ в виде таблицы Markdown со столбцами 'Продукт', 'Цена (руб.)', 'Магазин':