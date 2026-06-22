#pip -q install langchain langchain-cohere cohere wikipedia-api

from langchain_cohere import ChatCohere
import wikipediaapi, getpass

# API Key
api_key = getpass.getpass("Enter Cohere API Key: ")
llm = ChatCohere(cohere_api_key=api_key, model="command-r-plus-08-2024")   #command-r-plus-08-2024

# Load IPC content
wiki = wikipediaapi.Wikipedia(user_agent="ipc-bot", language="en")
page = wiki.page("Indian Penal Code")
ipc_text = page.text[:2000]

# Chat loop
while True:
    q = input("\nAsk about IPC (type 'exit'): ")
    if q == "exit":
        break

    prompt = f"Answer based on IPC:\n{ipc_text}\n\nQ: {q}"
    res = llm.invoke(prompt)

    print("\nAnswer:\n", res.content)



prompt = f"""
You are an assistant that answers ONLY from the following IPC document.

If the answer exists in the document, answer directly.
If it does not exist, reply "Information not found."

Document:
{ipc_text}

Question: {q}
"""
