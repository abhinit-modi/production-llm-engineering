from evaluation.evaluate import Evaluator
import argparse
import os

# Configurable model repository names (set via environment variables)
SFT_MODEL_ID = os.environ.get('SFT_MODEL_ID', 'YOUR_USERNAME/sft-model')
PPO_MODEL_ID = os.environ.get('PPO_MODEL_ID', 'YOUR_USERNAME/ppo-model')
DPO_MODEL_ID = os.environ.get('DPO_MODEL_ID', 'YOUR_USERNAME/dpo-model')
ORPO_MODEL_ID = os.environ.get('ORPO_MODEL_ID', 'YOUR_USERNAME/orpo-model')


def run(training_type):
    # Run evaluation for each model variant with corresponding output file
    if training_type == 'base':
        Evaluator.run(
            model_id='openai-community/gpt2',
            result_file='baseline_results.jsonl'
        )
    elif training_type == 'supervised':
        Evaluator.run(
            model_id=SFT_MODEL_ID,
            result_file='supervised_results.jsonl'
        )
    elif training_type == 'ppo':
        Evaluator.run(
            model_id=PPO_MODEL_ID,
            result_file='ppo_results.jsonl'
        )
    elif training_type == 'dpo':
        Evaluator.run(
            model_id=DPO_MODEL_ID,
            result_file='dpo_results.jsonl'
        )
    elif training_type == 'orpo':
        Evaluator.run(
            model_id=ORPO_MODEL_ID,
            result_file='orpo_results.jsonl'
        )
    else: 
        raise NotImplemented


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation application.")
    parser.add_argument(
        "--training_type",
        type=str,
        required=True,
        choices=["base", "supervised", "ppo", "dpo", "orpo"],
        help="Type of model to evaluate."
    )
    args = parser.parse_args()
    run(args.training_type)