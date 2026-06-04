from dotenv import load_dotenv

load_dotenv()

from typing import List

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient

llm = ChatOpenAI(model="gpt-5")
tavily = TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches the web
#     Args:
#         query: The query to search for
#     Returns:
#         The search Result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job posting on LinkedIn Data engineer in Bengaluru posted in the last 3 days")})
    print(result)


if __name__ == "__main__":
    main()
