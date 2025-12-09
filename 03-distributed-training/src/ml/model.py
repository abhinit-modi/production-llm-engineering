from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Model:

    def __init__(self, model_id, num_labels):
        self.model_id = model_id
        self.model, self.tokenizer = self._get_model(
            model_id, 
            num_labels
        )

    def _get_model(self, model_id, num_labels):
        # Load sequence classification model and tokenizer
        # Configure padding token (required for GPT-2 style models)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, 
            num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer