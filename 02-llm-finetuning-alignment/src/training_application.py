from model.model_connection import Model
from data.data_connection import DataConnector
from data.data_processing import DataProcessor
from training.base_train import BaseTrainer
from training.train_supervised import SupervisedTrainer
from training.train_rlhf import RewardModelTrainer, RLHFTrainer
from training.train_dpo import MyDPOTrainer
from training.train_orpo import MyORPOTrainer

import argparse

def run(training_type, training_mode='local'):
    # TODO: implement this function.
    trainer = None
    processed_data = None

    if training_type == 'supervised':
        model_id = 'openai-community/gpt2'
        model, tokenizer = Model.get_model_for_LM(model_id)
        
        data_path = 'ybisk/piqa'
        if training_mode == 'aws':
            dataset = DataConnector.get_data(data_path, split='train')
        else:
            dataset = DataConnector.get_data(data_path, split='train', sample_size=1000, start=0)

        data_processor = DataProcessor(tokenizer)
        processed_data = data_processor.prepare_for_supervised_training(dataset)

        trainer = SupervisedTrainer(model=model, tokenizer=tokenizer)

    elif training_type == 'reward':
        model_id = 'openai-community/gpt2' # replace with sft model
        model, tokenizer = Model.get_model_for_reward(model_id)

        data_path = 'ybisk/piqa'
        if training_mode == 'aws':
            dataset = DataConnector.get_data(data_path, split='train')
        else:
            dataset = DataConnector.get_data(data_path, split='train', sample_size=200, start=1000)

        data_processor = DataProcessor(tokenizer)
        processed_data = data_processor.prepare_for_reward_training(dataset)

        trainer = RewardModelTrainer(model=model, tokenizer=tokenizer)

    elif training_type == 'ppo':
        reward_model_id = 'Abhinit/HW2-reward'
        model_id = 'Abhinit/HW2-supervised'
        policy_model, tokenizer, reward_model, value_model = Model.get_model_for_PPO(model_id, reward_model_id)
        data_path = 'ybisk/piqa'

        if training_mode == 'aws':
            dataset = DataConnector.get_data(data_path, split='train')
        else:
            dataset = DataConnector.get_data(data_path, split='train', sample_size=100, start=1200)

        data_processor = DataProcessor(tokenizer)
        processed_data = data_processor.prepare_for_ppo_training(dataset)

        trainer = RLHFTrainer(
            policy_model=policy_model, 
            reward_model=reward_model, 
            value_model=value_model, 
            tokenizer=tokenizer
        )

    elif training_type == 'dpo':
        model_id = 'Abhinit/HW2-supervised'

        model, tokenizer = Model.get_model_for_LM(model_id)
        data_path = 'ybisk/piqa'

        if training_mode == 'aws':
            dataset = DataConnector.get_data(data_path, split='train')
        else:
            dataset = DataConnector.get_data(data_path, split='train', sample_size=100, start=1300)

        data_processor = DataProcessor(tokenizer)
        processed_data = data_processor.prepare_for_dpo_training(dataset)

        trainer = MyDPOTrainer(model=model, tokenizer=tokenizer)

    elif training_type == 'orpo':
        model_id = 'Abhinit/HW2-supervised'
        model, tokenizer = Model.get_model_for_LM(model_id)
        data_path = 'ybisk/piqa'
        if training_mode == 'aws':
            dataset = DataConnector.get_data(data_path, split='train')
        else:
            dataset = DataConnector.get_data(data_path, split='train', sample_size=100, start=1400)
        data_processor = DataProcessor(tokenizer)
        processed_data = data_processor.prepare_for_orpo_training(dataset)
        trainer = MyORPOTrainer(model=model, tokenizer=tokenizer)

    else: 
        raise NotImplemented
        
    trainer.train(processed_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training application.")
    parser.add_argument(
        "--training_type",
        type=str,
        required=True,
        choices=["supervised", "reward", "ppo", "dpo", "orpo"],
        help="Type of training to run."
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="local",
        help="AWS or local training."
    )
    args = parser.parse_args()
    run(training_type=args.training_type, training_mode=args.training_mode)
    