import requests  # For making HTTP requests
import sseclient  # For handling Server-Sent Events (SSE)
import json  

# Define the URL of the text generation API endpoint
url = 'https://[YOUR ENDPOINT URL]/generate-stream'

# Define the input text (prompt) for the text generation
# The special tokens (<|system|>, <|user|>, etc.) are used to structure the input
text = """
<|system|>
You are a helpful assistant.<|end|>
<|user|>
How are you?<|end|>
<|assistant|> 
"""

# Prepare the data to be sent to the API
data = {
    "prompt": text,  # The input text
    "max_tokens": 50,  # Maximum number of tokens to generate
    "temperature": 0.7  # Controls randomness in generation (0.7 is moderately random)
}

# Set the headers for the HTTP request
# This tells the server we're sending JSON data
headers = {'Content-Type': 'application/json'}

# Send a POST request to the API
# json.dumps(data) converts the data dictionary to a JSON string
# stream=True keeps the connection open for receiving streaming data
response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

# Create an SSE client to handle the streaming response
client = sseclient.SSEClient(response)

# Initialize a variable to store the full generated text
full_text = ""

# Process each event in the SSE stream
for event in client.events():
    if event.data == '[DONE]':
        # If we receive a '[DONE]' message, the generation is complete
        break
    else:
        # Parse the JSON data from the event
        data = json.loads(event.data)
        
        # Extract only the new part of the text
        # This avoids repetition in the output
        new_text = data['text'][len(full_text):]
        
        # Print the new text without a newline and flush the output
        # This gives a smooth, real-time display of the generated text
        print(new_text, end="", flush=True)
        
        # Add the new text to our full text
        full_text += new_text

# Close the SSE client connection
client.close()