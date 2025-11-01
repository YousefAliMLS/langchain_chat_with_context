from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser

def build_context_presence_tool(llm: BaseLanguageModel) -> Tool:
    with open ('C:\\Users\\Al Badr\\Desktop\\NU UNI Documents\\Third Year Fall 2025\\NLP Internship\\Cellula_6week_[Yousef_Mahmoud_Ali]\\langchain_chat_with_context\\prompts\\context_judge_prompt.txt','r') as f:
        prompt_template = f.read();

    prompt = PromptTemplate.from_template(prompt_template);

    chain = prompt | llm | StrOutputParser()



    tool = Tool.from_function(
        func=lambda input_str: chain.invoke({"input": input_str}),
        name="ContextPresenceJudge",
        description="Use this tool FIRST. It checks if the user's message includes background context or is just a direct question. Input must be the user's full message."
    )
    return tool