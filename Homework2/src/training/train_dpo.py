import copy
from training.base_train import BaseTrainer
from trl import DPOConfig, DPOTrainer
from huggingface_hub import login
from training.base_train import BaseTrainer, login, HUGGINGFACE_TOKEN


class MyDPOTrainer(BaseTrainer):

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
        
        # TODO: Set the training arguments up with the DPOConfig
        self.args = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=num_epoch,
            report_to='none',
            push_to_hub=True
        )

    def train(self, dataset):
        # TODO:  Set the training up with the DPOTrainer.  
        # Call the train method of the DPOTrainer class, 
        # and don't forget to push the model to the model hub
        model_ref = copy.deepcopy(self.model)
        train_dataset, eval_dataset = self.split_dataset(dataset)
        print(f"Train dataset size: {len(train_dataset)}")
        trainer = DPOTrainer(
            args=self.args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=model_ref,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
        self.save(trainer)

    def save(self, trainer):
        trainer.push_to_hub(self.output_dir)