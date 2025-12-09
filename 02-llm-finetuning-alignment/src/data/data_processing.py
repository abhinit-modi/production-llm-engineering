
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
        # Format data for supervised fine-tuning with prompt-completion pairs
        # Output format matches PIQA evaluation: "Question: {goal}\nAnswer: {solution}"
        # The correct solution is selected based on the label column
        def prepare_for_training(example):
            return {
                'text': f"Question: {example['goal']}\nAnswer: {example['sol1'] if example['label'] == 0 else example['sol2']}",
            }
            
        processed_dataset = dataset.map(prepare_for_training, remove_columns=dataset.column_names)
        processed_dataset = processed_dataset.map(self.trim_text, remove_columns=processed_dataset.column_names)
        return processed_dataset
   

    def prepare_for_reward_training(self, dataset):
        # Format data for reward model training with chosen/rejected pairs
        # Creates "chosen" and "rejected" columns with full prompt-answer format
        def prepare_for_reward(example):
            chosen_solution = example['sol1'] if example['label'] == 0 else example['sol2']
            rejected_solution = example['sol2'] if example['label'] == 0 else example['sol1']
            return {
                'chosen': f"Question: {example['goal']}\nAnswer: {chosen_solution}",
                'rejected': f"Question: {example['goal']}\nAnswer: {rejected_solution}"
            }
        
        def tokenize_for_reward_training(examples):
            # Tokenize with RewardTrainer-expected column names:
            # input_ids_chosen, attention_mask_chosen, input_ids_rejected, attention_mask_rejected

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
        # Format data for PPO training: prompt-only format for generation
        # Creates "Question: {goal}\nAnswer: " prompts for the policy to complete
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
        # Format data for DPO training with prompt/chosen/rejected columns
        # DPOTrainer expects untokenized data with these specific column names
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
        # Format data for ORPO training with prompt/chosen/rejected columns
        # ORPOTrainer expects untokenized data with these specific column names
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