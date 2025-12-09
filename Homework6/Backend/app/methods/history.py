from clients import client

def streaming_with_history(chat_history: list[dict[str, str]]):
    # TODO: implement the streaming_with_history function.
    #  As for simple_streaming, the function returns a generator for a streaming response.
    # chat_history is a list of the previous messages exchanged in the chatbot.
    return client.chat_completion(
        messages=chat_history,
        stream=True
    )