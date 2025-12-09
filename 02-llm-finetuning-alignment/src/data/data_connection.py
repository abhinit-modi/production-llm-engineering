
from datasets import load_dataset

class DataConnector:
 
    @staticmethod
    def get_data(path, split, sample_size=1000, start=0):
        # Load dataset from HuggingFace hub with specified sample range
        # Uses train split with index slicing for controlled sample size
        dataset = load_dataset(path, split=f'train[{start}:{start + sample_size}]', trust_remote_code=True)
        return dataset
