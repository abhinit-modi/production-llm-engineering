from training.base_train import BaseTrainer
from trl import ORPOConfig, ORPOTrainer
from huggingface_hub import login
from training.base_train import BaseTrainer, login, HUGGINGFACE_TOKEN


class MyORPOTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=4, 
            output_dir='HW2-dpo',
            result_file='dpo_results.json'
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir, result_file)
        
        # TODO: Set the training arguments up with the ORPOConfig
        self.args = ORPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=batch_size // 2,
            num_train_epochs=num_epoch,
            report_to='none',
            push_to_hub=True
        )

    def train(self, dataset):
        # TODO:  Set the training up with the ORPOTrainer.  
        # Call the train method of the ORPOTrainer class, 
        # and don't forget to push the model to the model hub
        train_dataset, eval_dataset = self.split_dataset(dataset)
        print(f"Train dataset size: {len(train_dataset)}")
        trainer = ORPOTrainer(
            args=self.args,
            processing_class=self.tokenizer,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            
        )
        trainer.train()
        self.save(trainer)
    
    def save(self, trainer):
        trainer.push_to_hub(self.output_dir)