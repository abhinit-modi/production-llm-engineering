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

        # Configure reward model training arguments
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
        # Train reward model to score chosen vs rejected responses
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

        # Configure PPO training arguments
        self.args = PPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=num_epoch,
            report_to='none',
            push_to_hub=True,
        )

    def train(self, tokenized_data):
        # Train policy with PPO using reward and value models
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

        # Generation config for policy rollouts
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # Create reward pipeline for scoring generated responses
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
                # Generate responses from policy model
                response_tensors = self.policy_model.generate(query_tensors, **generation_kwargs)
                print(f"Generated response tensors: {response_tensors[0]}")

                # Decode generated tokens to text
                batch["response"] = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                print(f"Decoded response: {batch['response'][0]}")
            
                #### Compute reward score
                # Format prompt-response pairs for reward scoring
                texts = [f"Question: {q}\nAnswer: {r}" for q, r in zip(input_text, batch["response"])]

                # Get reward scores from the reward model
                rewards = reward_pipeline(texts)
                print(f"Rewards: {rewards[0]}")
            
                #### Run PPO step
                # Update policy using PPO with computed rewards
                stats = trainer.optimizer.step(query_tensors, response_tensors, rewards)
                trainer.log_stats(stats, batch, rewards)
        
        self.save(trainer)

    def save(self, trainer):
        login(token=HUGGINGFACE_TOKEN)
        trainer.push_to_hub(self.output_dir)
