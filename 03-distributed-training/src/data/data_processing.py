class DataProcessor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def transform(self, data):
        # Tokenize text data with padding and truncation
        # Renames 'label' to 'labels' for HuggingFace trainer compatibility
        # Sets output format to PyTorch tensors

        dataset = data.map(lambda x: self.tokenizer(
            x['text'],
            truncation=True,
            padding='max_length',
            max_length=128,
        ), remove_columns=['text'])
        dataset = dataset.rename_column('label', 'labels')
        dataset.set_format(type='torch')
        return dataset
