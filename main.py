from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_openai import ChatOpenAI
from tools.context_presence_judge import build_context_presence_tool
from tools.web_search_tool import build_web_search_tool
from tools.context_relevance_checker import build_context_relevance_tool
from tools.context_splitter import build_context_splitter_tool
from dotenv import load_dotenv
import os

load_dotenv()


llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.5,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "LangChain Agent"
    }
)

#build all four tools
contextJudgeTool = build_context_presence_tool(llm)
webSearchTool = build_web_search_tool()
contextRelevanceTool = build_context_relevance_tool(llm)
contextSplitterTool = build_context_splitter_tool(llm)

#added all four tools to the tools list
tools = [
    contextJudgeTool, 
    webSearchTool, 
    contextRelevanceTool, 
    contextSplitterTool
]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

print("--- Agent Test 1 (Context Missing) ---")
responseMissing = agent_executor.invoke({"input": "What is langchain used for? "})
print(responseMissing['output'])

print("\n--- Agent Test 2 (Context Provided) ---")
responseProvided = agent_executor.invoke({
    "input": "Given that LangChain is a framework for LLMs, what is it used for?"
})
print(responseProvided['output'])

print("\n--- Agent Test 3 (Irrelevant Context) ---")
responseIrrelevant = agent_executor.invoke({
    "input": "My favorite food is pizza. That being said, what is the capital of France?"
})
print(responseIrrelevant['output'])