from datasets import load_dataset
from huggingface_hub import login

class DataConnector:
    
    # TODO:  Implement the DataConnector to pull the data from the path. 
    # As an example, you can choose to pull the data from the datasets 
    # package with the load_dataset function.

    @staticmethod
    def get_data(path):
        return load_dataset(path, trust_remote_code=True)
        