from langchain.tools import Tool
from tavily import TavilyClient
import os

def build_web_search_tool()-> Tool:
    tavilyClient = TavilyClient(api_key = os.environ["TAVILY_API_KEY"])

    def tavilySearchFunc(query: str) -> str:
        '''Web search using tavily'''

        try:
            results = tavilyClient.search(query=query, max_results = 3)
            return "\n---\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results['results']])
        except Exception as e:
            return f"Error occured during the web search: {e}. "
        
    tool = Tool.from_function(
        func=tavilySearchFunc,
        name = "Web Searching Tool",
        description = "Use this tool if the ContextPresenceJudge outputs 'context_missing'. It searches the web to find information for the user's question. Input is the user's original question."
    )
    return tool