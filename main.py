from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient

llm = ChatOpenAI(model="gpt-5-nano")
tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches the web
    Args:
        query: The query to search for
    Returns:
        The search Result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job posting on LinkedIn AI ML engineers in Bay Area, San Francisco")})
    print(result)


if __name__ == "__main__":
    main()
