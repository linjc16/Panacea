import os
from openai import OpenAI
import pdb
with open('./openai_api_azure.key', 'r') as f:
    api_key = f.read().strip()
    
    
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://trialmind.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from typing import List

def gpt_chat(prompt, query_dict):
    prompt = ChatPromptTemplate.from_template(prompt)

    model = AzureChatOpenAI(
        deployment_name="gpt-35", # "gpt-35"
        model_name='gpt-35-turbo',
    )
    chain = prompt | model | StrOutputParser()
    
    return chain.invoke(query_dict)



if __name__ == "__main__":
    temp = gpt_chat("Translate this sentence from English to French. {query}", "I love programming.")
    print(temp)