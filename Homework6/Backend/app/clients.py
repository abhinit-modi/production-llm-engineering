import os
from huggingface_hub import InferenceClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.3",
    token=os.environ['HF_TOKEN'],
)

openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])