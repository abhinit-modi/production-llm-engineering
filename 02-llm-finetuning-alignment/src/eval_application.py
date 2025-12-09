from evaluation.evaluate import Evaluator
import argparse


def run(training_type):
    # TODO: implement the right model_id and result_file
    if training_type == 'base':
        Evaluator.run(
            model_id='openai-community/gpt2',
            result_file='baseline_results.jsonl'
        )
    elif training_type == 'supervised':
        Evaluator.run(
            model_id='Abhinit/HW2-supervised',
            result_file='supervised_results.jsonl'
        )
    elif training_type == 'ppo':
        Evaluator.run(
            model_id='Abhinit/HW2-ppo',
            result_file='ppo_results.jsonl'
        )
    elif training_type == 'dpo':
        Evaluator.run(
            model_id='Abhinit/HW2-dpo',
            result_file='dpo_results.jsonl'
        )
    elif training_type == 'orpo':
        Evaluator.run(
            model_id='Abhinit/HW2-orpo',
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