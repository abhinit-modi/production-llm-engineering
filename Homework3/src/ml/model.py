from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Model:

    def __init__(self, model_id, num_labels):
        self.model_id = model_id
        self.model, self.tokenizer = self._get_model(
            model_id, 
            num_labels
        )

    def _get_model(self, model_id, num_labels):
        # TODO: Implement the Model class in to get the model and its tokenizer. 
        # Make sure to set the padding token.
        # tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = tokenizer.pad_token_id
        # 
        # I make the assumption that the model is a AutoModelForSequenceClassification model, 
        # therefore I provide the num_labels attribute
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, 
            num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer