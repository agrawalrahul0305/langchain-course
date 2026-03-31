from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field
import os

load_dotenv()
tavily = TavilyClient()

class Source(BaseModel):
   """Schema for a source used by the agent"""

   url:str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources to generate the answer")

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


llm = ChatOllama(model="functiongemma")
structured_llm = llm.with_structured_output(AgentResponse)
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    userquery = "search for good piano classes in san jose."
    #systemquery = "You are a helpful assistant. Once you have found the information, you MUST use the final response tool to format your answer."
    result = agent.invoke(
        {"messages": [HumanMessage(content=userquery)]}
    )
    raw_answer = result["messages"][3].content
    """ functiongemma does not suppport nested pydantic models so this is a workaround
        we can also openai models.
    """
    structured_final_result = structured_llm.invoke(
        f"Extract the piano class names and URLs from this text into the required format: {raw_answer}"
    )
    #structured_data = structured_final_result.get("structured_response")
    print(structured_final_result)
    #print(result)
    
if __name__ == "__main__":
    main()
