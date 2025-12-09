import requests
import json

def get_response(user_input, url, username):
    
    try:
            
        # Prepare request payload
        payload = {
            "question": user_input,
            "username": username
        }
        
        with requests.post(
            url, 
            json=payload,
            headers={
                "Accept": "text/event-stream",
                # "Cache-Control": "no-cache"
            },
            stream=True
        ) as response:
 
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    json_data = line[6:]  # Remove 'data: ' prefix
                    
                    if json_data == '[DONE]':
                        break
                    try:
                        data = json.loads(json_data)
                        chunk = data.get('chunk', '')
                        if not chunk:
                            continue
                        yield chunk
                    except json.JSONDecodeError:
                        continue
            
    except Exception as e:
        print(e)


def get_simple_response(user_input, url, username):
    
    try:
            
        # Prepare request payload
        payload = {
            "question": user_input,
            "username": username
        }
        
        response = requests.post(url, json=payload)

        data = response.json()
        return data['response']
            
    except Exception as e:
        print(e)