class DataProcessor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def transform(self, data):
        # TODO: Implement the DataProcessor. 
        # You need first to tokenize the data, 
        # and you can assume that the class has as an attribute 
        # the tokenizer related to the model you are going to use. 
        # Consider truncating or padding the resulting tokenized data. 
        # You need to make sure that the resulting data has the right 
        # column names for training: labels, input_ids, attention_mask. 
        # Make sure the resulting tensors are PyTorch tensors 
        # by using the set_format(type='torch') function.

        dataset = data.map(lambda x: self.tokenizer(
            x['text'],
            truncation=True,
            padding='max_length',
            max_length=128,
        ), remove_columns=['text'])
        dataset = dataset.rename_column('label', 'labels')
        dataset.set_format(type='torch')
        return dataset
