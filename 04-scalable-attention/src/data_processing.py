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
        # TODO: Implement DataProcessor.transform by mapping 
        # the tokenize_function to the dataset.
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
        # TODO: I provide two prompt templates: PROMPT_INPUT,  PROMPT_NO_INPUT. 
        # If the example has an input, use the PROMPT_INPUT and PROMPT_NO_INPUT otherwise.
        if example['input']:
            prompt = prompt_input.format(instruction=example['instruction'], input=example['input'])
        else:
            prompt = prompt_no_input.format(instruction=example['instruction'])
        return prompt
        
    def create_target(self, output):
        # TODO: The target will just be the output column with 
        # the end-of-sequence token (available in the 
        # tokenizer: tokenizer.eos_token) appended to it.
        try:
            return output + self.tokenizer.eos_token if output else self.tokenizer.eos_token
        except TypeError as e:
            print(f"Error processing output: {output}. Error: {e}")
            return self.tokenizer.eos_token
    
    def tokenize_function(self, examples):
        # TODO:  iterate through the examples, and create the prompts 
        # and the targets by using the DataProcessor.create_prompt 
        # and DataProcessor.create_target functions.
        prompts = []
        targets = []
        num_items = len(examples['instruction'])
        for i in range(num_items):
            prompts.append(self.create_prompt({'instruction': examples['instruction'][i], 'input': examples['input'][i] if 'input' in examples else None}))
            targets.append(self.create_target(examples['output'][i]))

        # TODO: Create the input_texts by iterating through the prompts 
        # and targets and concatenating them. For example, 
        # input_texts[0] = prompt[0] + targets[0].
        input_texts = [f"{prompt}{target}" for prompt, target in zip(prompts, targets)]
        print(f"Processed {len(input_texts)} examples.")

        # TODO: use the tokenizer to tokenize the input_texts and prompts. 
        # Truncate to the model_max_length but don't pad the sequence, 
        # as we are going to handle the padding in the data collator. 
        # Use return_tensors=None to return a Python list 
        # (we are going to change to PyTorch in the data collator as well)
        tokenized_inputs = self.tokenizer(input_texts, padding=False, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors=None)
        tokenized_prompts = self.tokenizer(prompts, padding=False, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors=None)

        input_ids = tokenized_inputs["input_ids"]
        labels = [ids.copy() for ids in input_ids]

        # TODO: For language modeling, the labels are the same as the inputs, 
        # but we are going to replace the tokens related to the prompt 
        # by the IGNORE_INDEX = -100. In labels, replace the token corresponding 
        # to the prompt by IGNORE_INDEX.
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

        # TODO: input_ids is a batch of tokenized input texts. 
        # For that batch:
        # - find the maximum input sequence length: \(l\)
        # - find the smallest integer value \(L\) greater than \(l\) that is divisible by num_group.  
        # \(L\) is the target_length for that batch 
        # Find the max length in the batch
        max_length = max(len(ids) for ids in input_ids)
        # Calculate the group size
        group_size = math.ceil(max_length * self.group_size_ratio)
        # Round up max_length to the next multiple of group_size
        target_length = math.ceil(max_length / group_size) * group_size
        
        # TODO: pad the input_ids with the padding token 
        # (tokenizer.pad_token_id) and the labels with the IGNORE_INDEX.
        input_ids = [self.pad_sequence(ids, target_length, self.tokenizer.pad_token_id) for ids in input_ids]
        labels = [self.pad_sequence(ids, target_length, IGNORE_INDEX) for ids in labels]
        
        # TODO: Make sure to convert the resulting data structures into a PyTorch tensor. 
        # If you have a list of PyTorch tensors, you can use the torch.stack 
        # function to convert, for example.
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        
        # TODO: compute the attention_mask as a boolean tensor where True means 
        # input_ids is not equal to the padding token, and False means otherwise. 
        # You can use the torch.ne function to compute this.
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    
    def pad_sequence(self, sequence, target_length, pad_value=None):
        # TODO: pad_sequence should pad the input sequence on the right up to 
        # target_length with pad_value. It should return a PyTorch tensor.
        if len(sequence) <= target_length:
            return torch.concat((sequence, torch.tensor([pad_value] * (target_length - len(sequence)))))
        return torch.tensor(sequence[:target_length])
        