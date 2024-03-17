from openai import OpenAI
import pdb
import os

with open('./openai_api.key', 'r') as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)
def gpt_chat(model, prompt, seed=44):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=4096,
    temperature=0.5,
    logprobs=True
    )
    
    return response.choices[0].message.content