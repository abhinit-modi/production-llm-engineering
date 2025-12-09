from huggingface_hub import login
from evaluation.evaluate import Evaluator
import os

from dotenv import load_dotenv
load_dotenv()

# TODO: replace by yours
HUGGINGFACE_TOKEN = os.environ.get('HF_TOKEN')
REPO_NAME = 'training_experiments'
EVAL_SPLIT = 0.2

class BaseTrainer:

    def __init__(self, model, tokenizer, num_epoch=3, batch_size=8, output_dir='', result_file='', eval_split=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.result_file = result_file
        self.eval_split = eval_split
        # TODO: make sure to export you Hugging Face Token:
        # export HF_TOKEN=[YOUR TOKEN]
        login(token=HUGGINGFACE_TOKEN)

    def split_dataset(self, dataset):
        """
        Split the dataset into training and evaluation sets.
        """
        dataset = dataset.train_test_split(test_size=self.eval_split, shuffle=True)
        return dataset['train'], dataset['test']

    def save(self, trainer):
        login(token=HUGGINGFACE_TOKEN)
        trainer.push_to_hub()

    def evaluate(self):
        Evaluator.run(
            model_id='{}/{}'.format(REPO_NAME, self.output_dir),
            result_file=self.result_file
        )