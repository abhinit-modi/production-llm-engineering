
from datasets import load_dataset

class DataConnector:
 
    @staticmethod
    def get_data(path, split, sample_size=1000, start=0):
        # TODO: Implement the DataConnector.get_data method 
        # of the data_connection.py file. Use the load_dataset 
        # function from the datasets package and select split = "train" 
        # to make sure we only train on the training data.
        dataset = load_dataset(path, split=f'train[{start}:{start + sample_size}]', trust_remote_code=True)
        return dataset
