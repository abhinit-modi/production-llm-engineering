import torch
import math


prompt_input = (
    "[INST] <<SYS>>\n"
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
    "<</SYS>> \n\n {instruction} \n{input} [/INST]"
)

prompt_no_input = "[INST]{instruction}[/INST]"

IGNORE_INDEX = -100


class DataProcessor:

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def transform(self, dataset):
        # Map tokenization function across dataset and convert to PyTorch format
        print("Transforming dataset...")
        tokenized_data = dataset.map(
            self.tokenize_function,
            batched=True,
            desc="Tokenizing dataset"
        )
        tokenized_data.set_format(
            type='torch')
        print("Dataset transformed.")
        return tokenized_data

    def create_prompt(self, example):
        # Format prompt using instruction template (with or without input context)
        if example['input']:
            prompt = prompt_input.format(instruction=example['instruction'], input=example['input'])
        else:
            prompt = prompt_no_input.format(instruction=example['instruction'])
        return prompt
        
    def create_target(self, output):
        # Create target by appending EOS token to the output
        try:
            return output + self.tokenizer.eos_token if output else self.tokenizer.eos_token
        except TypeError as e:
            print(f"Error processing output: {output}. Error: {e}")
            return self.tokenizer.eos_token
    
    def tokenize_function(self, examples):
        # Create prompts and targets from instruction/output pairs
        prompts = []
        targets = []
        num_items = len(examples['instruction'])
        for i in range(num_items):
            prompts.append(self.create_prompt({'instruction': examples['instruction'][i], 'input': examples['input'][i] if 'input' in examples else None}))
            targets.append(self.create_target(examples['output'][i]))

        # Concatenate prompts with targets to form complete training sequences
        input_texts = [f"{prompt}{target}" for prompt, target in zip(prompts, targets)]
        print(f"Processed {len(input_texts)} examples.")

        # Tokenize full sequences and prompts separately (no padding - handled by collator)
        tokenized_inputs = self.tokenizer(input_texts, padding=False, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors=None)
        tokenized_prompts = self.tokenizer(prompts, padding=False, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors=None)

        input_ids = tokenized_inputs["input_ids"]
        labels = [ids.copy() for ids in input_ids]

        # Mask prompt tokens with IGNORE_INDEX (-100) so loss is only computed on completion
        for i, label_ids in enumerate(labels):
            prompt_length = len(tokenized_prompts['input_ids'][i])
            label_ids[:prompt_length] = [IGNORE_INDEX] * prompt_length

        print(f"Processed {len(input_ids)} examples.")
        # print(f"First example input_ids: {input_ids[0]}")
        # print(f"First example labels: {labels[0]}")
        return dict(input_ids=input_ids, labels=labels)
    

class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer, group_size_ratio) -> None:
        self.tokenizer = tokenizer
        self.group_size_ratio = group_size_ratio

    def __call__(self, instances):
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]

        # Calculate target length: smallest multiple of group_size >= max sequence length
        # This ensures sequences can be evenly divided into groups for sparse attention
        max_length = max(len(ids) for ids in input_ids)
        group_size = math.ceil(max_length * self.group_size_ratio)
        target_length = math.ceil(max_length / group_size) * group_size
        
        # Pad sequences: input_ids with pad_token, labels with IGNORE_INDEX (-100)
        input_ids = [self.pad_sequence(ids, target_length, self.tokenizer.pad_token_id) for ids in input_ids]
        labels = [self.pad_sequence(ids, target_length, IGNORE_INDEX) for ids in labels]
        
        # Stack into batch tensors
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        
        # Create attention mask: True for real tokens, False for padding
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    
    def pad_sequence(self, sequence, target_length, pad_value=None):
        # Pad sequence on the right to target_length, return as PyTorch tensor
        if len(sequence) <= target_length:
            return torch.concat((sequence, torch.tensor([pad_value] * (target_length - len(sequence)))))
        return torch.tensor(sequence[:target_length])
        