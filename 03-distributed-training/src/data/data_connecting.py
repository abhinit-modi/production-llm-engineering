from datasets import load_dataset
from huggingface_hub import login

class DataConnector:
    
    # Load dataset from HuggingFace hub by path

    @staticmethod
    def get_data(path):
        return load_dataset(path, trust_remote_code=True)
        