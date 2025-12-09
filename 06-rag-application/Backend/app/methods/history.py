from clients import client

def streaming_with_history(chat_history: list[dict[str, str]]):
    # Stream LLM response using full conversation history for context
    return client.chat_completion(
        messages=chat_history,
        stream=True
    )