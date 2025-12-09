from training.base_train import BaseTrainer

from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

class SupervisedTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=16, 
            output_dir='HW2-supervised',
            result_file='supervised_results.json'
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir, result_file)
        
        # Configure supervised fine-tuning arguments
        self.args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=num_epoch,
            report_to='none',
            push_to_hub=True,
        )
        # Data collator that masks prompt tokens, only training on completions
        self.collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template="Answer:",
        )

    def train(self, dataset):
        # Train with SFTTrainer for instruction fine-tuning
        train_dataset, eval_dataset = self.split_dataset(dataset)
        print(f"Train dataset size: {len(train_dataset)}")
        trainer = SFTTrainer(
            args=self.args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.collator
        )
        trainer.train()
        self.save(trainer)


