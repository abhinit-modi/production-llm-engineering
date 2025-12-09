import subprocess


BASH_COMMAND = """
lm_eval \
    --model hf \
    --model_args pretrained={model_id} \
    --batch_size auto \
    --tasks {task} \
    --device {device} \
    --output_path {result_file} \
"""

class Evaluator:
    @staticmethod
    def run(
        model_id='openai-community/gpt2', 
        task='piqa', 
        device='cpu', 
        result_file='results.json'
    ):
        command = BASH_COMMAND.format(
            model_id=model_id,
            task=task,
            device=device,
            result_file=result_file
        )
        print(f"Running evaluation with command: {command}")
        result = subprocess.run(
            ['bash', '-c', command], 
            capture_output=True, 
            text=True
        )
        print(result)
        return result.stdout

if __name__ == '__main__':

    results = Evaluator.run()