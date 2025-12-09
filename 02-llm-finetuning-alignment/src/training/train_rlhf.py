from training.base_train import BaseTrainer, login, HUGGINGFACE_TOKEN
from trl import RewardTrainer, RewardConfig, PPOConfig, PPOTrainer
from transformers import pipeline


class RewardModelTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=4, 
            output_dir='HW2-reward',
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir)

        # TODO: set up the training arguments
        self.tokenizer = tokenizer
        self.args = RewardConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=batch_size // 2,
            num_train_epochs=num_epoch,
            report_to='none',
            push_to_hub=True
        )

    def train(self, tokenized_data):
        # TODO: Use the RewardTrainer to set up the training. 
        # Call the train method of the RewardTrainer class, 
        # and don't forget to push the model to the model hub.
        train_dataset, eval_dataset = self.split_dataset(tokenized_data)
        print(f"Train dataset size: {len(train_dataset)}")
        trainer = RewardTrainer(
            args=self.args,
            processing_class=self.tokenizer,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()    
        self.save(trainer)

class RLHFTrainer(BaseTrainer):

    def __init__(
            self, 
            policy_model, 
            reward_model,
            value_model,
            tokenizer, 
            num_epoch=3, 
            batch_size=16, 
            output_dir='HW2-ppo',
            result_file='ppo_results.json'
        ):
        super().__init__(policy_model, tokenizer, num_epoch, batch_size, output_dir, result_file)
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.value_model = value_model
        
        self.value_model = value_model
        self.reward_model = reward_model

        # TODO: implement the training arguments with the PPOConfig. 
        self.args = PPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=num_epoch,
            report_to='none',
            push_to_hub=True,
        )

    def train(self, tokenized_data):
        # TODO: implement the trainer with the PPOTrainer
        train_dataset, eval_dataset = self.split_dataset(tokenized_data)
        print(f"Train dataset size: {len(train_dataset)}")
        trainer = PPOTrainer(
            args=self.args,
            processing_class=self.tokenizer,
            model=self.policy_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            ref_model=None
        )

        trainer.train()
        self.save(trainer)

    def train_steps(self, tokenized_data):
        train_dataset, eval_dataset = self.split_dataset(tokenized_data)
        print(f"Train dataset size: {len(train_dataset)}")
        trainer = PPOTrainer(
            args=self.args,
            processing_class=self.tokenizer,
            model=self.policy_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            ref_model=None
        )

        # TODO: implement the generation_kwargs that will be used in the PPOTrainer.generate method 
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # TODO: Implement the reward_pipeline by using your reward model and pipeline function
        reward_pipeline = pipeline(
            "text-classification",
            model=self.reward_model,
            tokenizer=self.tokenizer)

        for epoch in range(self.num_epoch):
            for batch in trainer.dataloader: 
                print(batch)
                trainer.optimizer.zero_grad()

                query_tensors = batch["input_ids"]
                input_text = self.tokenizer(query_tensors, skip_special_tokens=True)    
                print(f"Input text: {input_text[0]}")
                
                #### Get response from SFTModel
                # TODO: Generate the response_tensors from the query_tensors
                response_tensors = self.policy_model.generate(query_tensors, **generation_kwargs)
                print(f"Generated response tensors: {response_tensors[0]}")

                # TODO: Decode the response_tensors by using the tokenizer
                batch["response"] = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                print(f"Decoded response: {batch['response'][0]}")
            
                #### Compute reward score
                # TODO: Create the input text for the reward_pipeline by using batch["goal"] and batch["response"]
                texts = [f"Question: {q}\nAnswer: {r}" for q, r in zip(input_text, batch["response"])]

                # TODO: Pass the input text to the reward_pipeline and extract the score output.
                rewards = reward_pipeline(texts)
                print(f"Rewards: {rewards[0]}")
            
                #### Run PPO step
                # TODO: Update the PPO model by using the query_tensors, response_tensors,  and rewards.
                stats = trainer.optimizer.step(query_tensors, response_tensors, rewards)
                trainer.log_stats(stats, batch, rewards)
        
        self.save(trainer)

    def save(self, trainer):
        login(token=HUGGINGFACE_TOKEN)
        trainer.push_to_hub(self.output_dir)
