from clients import client
from data_indexing import DataIndexer

indexer = DataIndexer()

# Prompt to condense chat history into standalone query for retrieval
HISTORY_SUMMARIZER = """Given the following conversation, rephrase the latest question to be a standalone question, in its original language.

Chat History:
{chat_history}

Standalone question:"""

SYSTEM_PROMPT = """Answer the user's based only on the following context:

{context}"""

def streaming_with_rag(chat_history: list[dict[str, str]], hybrid_search=False):
    # Generate standalone query from chat history using LLM summarization
    user_query = client.chat_completion(
        messages=[{'role': 'user', 'content': HISTORY_SUMMARIZER.format(chat_history="\n".join([f"{chat['role']}: {chat['content']}" for chat in chat_history]))}],
    ).choices[0].message.content
    print(f"User query: {user_query}")

    documents = indexer.search(user_query, hybrid_search=hybrid_search)
    print(f"Retrieved documents: {len(documents)}")

    # Build prompt with retrieved context and user query
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT.format(context="\n".join([doc.content for doc in documents]))}, {'role': 'user', 'content': user_query}]

    return client.chat_completion(
        messages=messages,
        stream=True
    )

