from clients import client

def simple_streaming(user_query: str):
    return client.chat_completion(
        messages=[{'role': 'user', 'content': user_query}],
        stream=True
    )
