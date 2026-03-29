from typing import List, Dict
from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools import BaseTool, StructuredTool
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.stdout_callback_handler import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

class GetPriceTool(BaseTool):
    name = "get_price"
    description = (
        "Используется для получения цены конкретного товара в конкретном городе."
    )

    def _run(self, product: str, city: str) -> str:
        prompt_template = """
            Ты эксперт по ценам товаров в разных городах России.
            Назови среднюю цену {product} в городе {city}.
            Форматируй ответ в виде таблицы Markdown: