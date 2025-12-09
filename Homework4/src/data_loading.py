from datasets import load_dataset

DATA_PATH = 'Yukang/LongAlpaca-12k'

class DataLoader:
    
    @staticmethod
    def get_data(data_path=DATA_PATH):
        return load_dataset(data_path, split='train')  # Load a subset for testing
    