
MAX_SEQ_LEN = 1024  # Maximum sequence length for tokenization

class DataProcessor:

    def __init__(self, tokenizer, max_seq_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def trim_text(self, example):
        return {
            'text': example['text'][:self.max_seq_len]  # Trimming to 1024 characters
        }
    
    def prepare_for_supervised_training(self, dataset):
        # TODO: implement the method. 
        # The method should return 2 columns named "completion" and "prompt".
        # 
        # - The "prompt" should have the form of "Question: {goal}\nAnswer: ", 
        # because it is the way the instructions get presented 
        # to the models in the PIQA test of the Evaluation Harness.
        # The {goal} represents the text of the original "goal" column.
        # 
        # - The "completion" is the solution that corresponds to the "label" column. 
        # The returned dataset doesn't need to be tokenized, 
        # but it should be in a DatasetDict format. 
        # You could decide to return a validation dataset as well 
        # to get better visibility during your training.
        def prepare_for_training(example):
            return {
                'text': f"Question: {example['goal']}\nAnswer: {example['sol1'] if example['label'] == 0 else example['sol2']}",
            }
            
        processed_dataset = dataset.map(prepare_for_training, remove_columns=dataset.column_names)
        processed_dataset = processed_dataset.map(self.trim_text, remove_columns=processed_dataset.column_names)
        return processed_dataset
   

    def prepare_for_reward_training(self, dataset):
        # TODO: The prepare_for_reward_training method should do the following things:
        # - create two new columns, "chosen" and "rejected," from the original data.
        # The "chosen" column should have the following format:
        # "Question: {goal}\nAnswer: {chosen_solution}"
        # and the "rejected" column should have the following format:
        # "Question: {goal}\nAnswer: {rejected_solution}"
        def prepare_for_reward(example):
            chosen_solution = example['sol1'] if example['label'] == 0 else example['sol2']
            rejected_solution = example['sol2'] if example['label'] == 0 else example['sol1']
            return {
                'chosen': f"Question: {example['goal']}\nAnswer: {chosen_solution}",
                'rejected': f"Question: {example['goal']}\nAnswer: {rejected_solution}"
            }
        
        def tokenize_for_reward_training(examples):
            # TODO: The RewardTrainer expects the data 
            # to be tokenized with very specific column names:
            # - "input_ids_chosen"
            # - "attention_mask_chosen"
            # - "input_ids_rejected"
            # - "attention_mask_rejected"
            # Implement the method to tokenize the data using the expected format for the RewardTrainer.

            chosen_tokenized = self.tokenizer(examples['chosen'], add_special_tokens=False, truncation=True, max_length=self.max_seq_len, padding='max_length')
            rejected_tokenized = self.tokenizer(examples['rejected'], add_special_tokens=False, truncation=True, max_length=self.max_seq_len, padding='max_length')
            chosen_tokenized.update({'input_ids_chosen': chosen_tokenized.pop('input_ids'), 'attention_mask_chosen': chosen_tokenized.pop('attention_mask')})
            rejected_tokenized.update({'input_ids_rejected': rejected_tokenized.pop('input_ids'), 'attention_mask_rejected': rejected_tokenized.pop('attention_mask')})
            return {**chosen_tokenized, **rejected_tokenized}
        
        dataset = dataset.map(prepare_for_reward, remove_columns=dataset.column_names)
        tokenized_data = dataset.map(tokenize_for_reward_training, remove_columns=dataset.column_names)
        tokenized_data.set_format(type="torch")
        return tokenized_data
    
    def prepare_for_ppo_training(self, dataset):
        # TODO:  Implement the method. 
        # We just need to add the indicators "Question: {goal}\nAnswer: " 
        # and tokenize the resulting text.
        def prepare_for_ppo(example):
            return {
                'text': f"Question: {example['goal']}\nAnswer: "
            }
        print('dataset_sample', dataset[1])
        dataset = dataset.map(prepare_for_ppo, remove_columns=dataset.column_names)
        print('dataset_sample_after_formatting', dataset[1])
        tokenized_dataset = dataset.map(lambda x: self.tokenizer(x['text'], add_special_tokens=True, truncation=True, max_length=self.max_seq_len, padding='max_length'),
                                        remove_columns=dataset.column_names)
        print('dataset_sample_after_tokenization', tokenized_dataset[1])
        tokenized_dataset.set_format(type="torch")
        # print(dataset[0]['text'])
        # tokenized_dataset = self.tokenizer.encode(dataset['text'])
        print(f"Tokenized dataset: {len(tokenized_dataset)}", tokenized_dataset[0])
        return tokenized_dataset
    
    def prepare_for_dpo_training(self, dataset):
        # TODO: implement the metho. The HuggingFace DPOTrainer 
        # expects a very specific format of the input data. 
        # We don't need to input data to be tokenized, 
        # but we need three columns with those names:
        # - 'prompt': in our case, the 'goal' column.
        # - 'chosen': the correct response
        # - 'rejected': the wrong response
        def prepare_for_dpo(example):
            chosen_solution = example['sol1'] if example['label'] == 0 else example['sol2']
            rejected_solution = example['sol2'] if example['label'] == 0 else example['sol1']
            return {
                'prompt': f"Question: {example['goal']}",
                'chosen': f"Answer: {chosen_solution}",
                'rejected': f"Answer: {rejected_solution}"
            }
        processed_dataset = dataset.map(prepare_for_dpo, remove_columns=dataset.column_names)
        return processed_dataset
    
    def prepare_for_orpo_training(self, dataset):
        # TODO: implement the metho. The HuggingFace ORPOTrainer 
        # expects a very specific format of the input data. 
        # We don't need to input data to be tokenized, 
        # but we need three columns with those names:
        # - 'prompt': in our case, the 'goal' column.
        # - 'chosen': the correct response
        # - 'rejected': the wrong response
        def prepare_for_orpo(example):
            chosen_solution = example['sol1'] if example['label'] == 0 else example['sol2']
            rejected_solution = example['sol2'] if example['label'] == 0 else example['sol1']
            return {
                'prompt': f"Question: {example['goal']}",
                'chosen': f"Answer: {chosen_solution}",
                'rejected': f"Answer: {rejected_solution}"
            }
        processed_dataset = dataset.map(prepare_for_orpo, remove_columns=dataset.column_names)
        return processed_dataset