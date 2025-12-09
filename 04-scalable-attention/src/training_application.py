from data_loading import DataLoader
from data_processing import DataProcessor, DataCollatorForSupervisedDataset
from model_loading import ModelLoader
from training import LoRATrainer

GROUP_SIZE_RATIO = 1/4

def run():
    # TODO: implement:
    # - get the data
    # - load the model
    # - get the data collator
    # - get the trainer
    # - process the data
    # - train the model
    data = DataLoader.get_data()
    print(f"Loaded {len(data)} examples from the dataset.")
    print(f"First example: {data[0]}")

    model_loader = ModelLoader(group_size_ratio=GROUP_SIZE_RATIO)
    model, tokenizer = model_loader.load_and_prepare_model()
    print("Model and tokenizer loaded and prepared.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer, GROUP_SIZE_RATIO)

    trainer = LoRATrainer(model, tokenizer, data_collator, GROUP_SIZE_RATIO)

    data_processor = DataProcessor(tokenizer)
    tokenized_data = data_processor.transform(data)
    print(f"Data processed and tokenized {len(tokenized_data)} examples.")
    print(f"First tokenized example: {tokenized_data[0]}")

    trainer.train(tokenized_data)

if __name__ == '__main__':
    run()