from transformers import Trainer, get_cosine_schedule_with_warmup, TrainingArguments
import torch

AdamW = torch.optim.AdamW

WARM_UP_STEPS = 10
TOTAL_TRAINING_STEPS = 1000

class LoRATrainer:

    def __init__(self, model, tokenizer, data_collator, group_size_ratio) -> None:
        # TODO: Complete the LoRATrainer to train with AdamW and a cosine scheduler 
        # with warmup. You can use the AdamW Pytorch class and 
        # the get_cosine_schedule_with_warmup function. 
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.training_arguments = TrainingArguments(
            output_dir="./lora_model",
            overwrite_output_dir=True,
            num_train_epochs=1,
            batch_eval_metrics= False,
            # "per_device_train_batch_size": 8,
            # "save_steps": 10_000,
            # "save_total_limit": 2,
            # "logging_steps": 500,
            # "evaluation_strategy": "steps",
            # "eval_steps": 1000,
        )
        self.optimizer = AdamW(model.parameters())
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=WARM_UP_STEPS,
            num_training_steps=TOTAL_TRAINING_STEPS,
        )
        self.group_size_ratio = group_size_ratio

    def train(self, tokenized_data):
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            train_dataset=tokenized_data,
            data_collator=self.data_collator,
            optimizers=(self.optimizer, self.scheduler)
        )

        trainer.train()
        self.save()

    def save(self):
        # TODO: Implement the LoRATrainer.save function.
        self.model.save_pretrained("./lora_model", push_to_hub=True)
        self.tokenizer.save_pretrained("./lora_model", push_to_hub=True)
