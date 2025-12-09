import requests
url='http://localhost:8000/filtered_rag/stream'
payload = {'question':'where does langserve fit in? how is it related to langchain?', 'username':'new_user'}
response = requests.post(url, json=payload)
for chunk in response.iter_lines():
    if chunk:
        print(chunk.decode('utf-8').replace('data: chunk: ', ''), end="", flush=True)