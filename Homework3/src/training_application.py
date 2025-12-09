from dotenv import load_dotenv
load_dotenv()

import os


HF_TOKEN=os.environ['HF_TOKEN']
WANDB_TOKEN=os.environ['WANDB_TOKEN']
import os

os.environ['HF_TOKEN'] = HF_TOKEN
os.environ["WANDB_API_KEY"] = WANDB_TOKEN


from data.data_connecting import DataConnector
from data.data_processing import DataProcessor
from ml.training import BasicTrainer, AcceleratedTrainer
from ml.model import Model
import argparse


MODEL_ID = 'gpt2'
DATA_PATH = 'dair-ai/emotion'

def run(training_type="basic"):
    # - Get the data
    data = DataConnector.get_data(DATA_PATH)
    print(f"Data loaded: {len(data)} samples, eg: {data['train'][0]}")
    # - Get the model
    print(data['train'].features['label'].num_classes)
    model = Model(model_id=MODEL_ID, num_labels=data['train'].features['label'].num_classes)
    print(f"Model loaded: {model.model.config}")
    # - Process the data
    data_processor = DataProcessor(tokenizer=model.tokenizer)
    tokenized_data = data_processor.transform(data)
    print(f"Data tokenized: {len(tokenized_data)} samples, eg: {tokenized_data['train'][0]}")

    # Downsize for testing
    tokenized_data['train'] = tokenized_data['train'].take(32)
    tokenized_data['validation'] = tokenized_data['validation'].take(8)
    print(tokenized_data['train'])

    # - Train the model with the data
    if training_type == "accelerated":
        trainer = AcceleratedTrainer(model=model.model, tokenizer=model.tokenizer, num_epochs=3, batch_size=8)
    else:
        trainer = BasicTrainer(model=model.model, tokenizer=model.tokenizer, num_epochs=3, batch_size=8)
    trainer.train(tokenized_data)
    # - Save the model
    trainer.save(model.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with specified training type.")
    parser.add_argument('--training_type', type=str, default="basic", choices=["basic", "accelerated"],
                        help="Type of training to use: 'basic' or 'accelerated'")
    args = parser.parse_args()
    run(training_type=args.training_type)