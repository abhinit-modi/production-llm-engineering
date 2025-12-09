from clients import client
from data_indexing import DataIndexer

indexer = DataIndexer()

# TODO: Write down a complete SYSTEM_PROMPT to use a chat history 
# and the retrieved documents as part of the context
HISTORY_SUMMARIZER = """Given the following conversation, rephrase the latest question to be a standalone question, in its original language.

Chat History:
{chat_history}

Standalone question:"""

SYSTEM_PROMPT = """Answer the user's based only on the following context:

{context}"""

def streaming_with_rag(chat_history: list[dict[str, str]], hybrid_search=False):
    # TODO: Create a user query from the chat history for the search function. 
    # It could be as simple as using the last user message,
    # or you can come up with something more complex.
    user_query = client.chat_completion(
        messages=[{'role': 'user', 'content': HISTORY_SUMMARIZER.format(chat_history="\n".join([f"{chat['role']}: {chat['content']}" for chat in chat_history]))}],
    ).choices[0].message.content
    print(f"User query: {user_query}")

    documents = indexer.search(user_query, hybrid_search=hybrid_search)
    print(f"Retrieved documents: {len(documents)}")

    # TODO: create the messages from the SYSTEM_PROMPT, 
    # the chat history, and the retrieved documents
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT.format(context="\n".join([doc.content for doc in documents]))}, {'role': 'user', 'content': user_query}]

    return client.chat_completion(
        messages=messages,
        stream=True
    )

