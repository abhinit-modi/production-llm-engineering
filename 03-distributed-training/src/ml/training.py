import torch
from torch import optim
import evaluate
from huggingface_hub import login

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from dotenv import load_dotenv
load_dotenv()

import os

HF_TOKEN = os.environ['HF_TOKEN']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Weights & Biases for experiment tracking
import wandb
wandb.login()
wandb.init(
    project=os.environ.get("WANDB_PROJECT", "distributed-training"),
    entity=os.environ.get("WANDB_ENTITY")) 


class BasicTrainer:

    def __init__(self, model, tokenizer, num_epochs, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(params=model.parameters())

    def train(self, tokenized_data):
        # Standard PyTorch training loop with W&B logging
        # Create train/eval data loaders from tokenized data
        train_dataloader, eval_dataloader = self.create_dataloaders(tokenized_data)

        # Move model to GPU/CPU device
        self.model.to(device)

        for epoch_index, epoch in enumerate(range(self.num_epochs)):
            # Set model to training mode
            self.model.train()
            for batch_index, batch in enumerate(train_dataloader):
                # Zero gradients before backward pass
                self.optimizer.zero_grad()
                # Move batch tensors to device
                batch = {k: v.to(device) for k, v in batch.items()}
                # Forward pass
                outputs = self.model(**batch)
                # Backward pass
                loss = outputs.loss
                loss.backward()
                # Update weights
                self.optimizer.step()
                wandb.log({'epoch': epoch_index+1, 'loss': loss.item(), 'step': f'{(epoch_index+1)*(batch_index+1)}'})
            
            eval_metric = self.eval(self.model, eval_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Eval Metric: {eval_metric}")
            wandb.log({'accuracy': eval_metric, 'epoch': epoch_index+1})
        

    def eval(self, model, eval_dataloader):
        # Evaluate model accuracy on validation set
        accuracy_metric = evaluate.load("accuracy")

        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        all_predictions = []
        all_labels = []
        # Iterate through validation batches
        for batch in eval_dataloader: 
            batch = {k: v.to(device) for k,v in batch.items()}
            # Inference without gradient computation
            with torch.no_grad():
                outputs = model(**batch) 
                preds = outputs.logits.argmax(dim=-1)
                print(f"Predictions: {preds}")
                all_predictions.append(preds)
                all_labels.append(batch['labels'])
            
        # Compute accuracy metric across all batches
        eval_metric = accuracy_metric.compute(
            predictions=torch.cat(all_predictions), 
            references=torch.cat(all_labels)
        )
        
        return eval_metric

    def create_dataloaders(self, tokenized_data):

        # Create PyTorch DataLoaders for train and validation sets
        train_dataloader, eval_dataloader = torch.utils.data.DataLoader(tokenized_data['train'], shuffle=True, batch_size=self.batch_size), torch.utils.data.DataLoader(tokenized_data['validation'], shuffle=False, batch_size=self.batch_size) 
        return train_dataloader, eval_dataloader
    
    def save(self, model):
        # Save model and tokenizer to HuggingFace Hub
        repo_name = os.environ.get("HF_REPO_NAME", "YOUR_USERNAME/basic-trainer")
        login(token=HF_TOKEN)
        model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)
    

class AcceleratedTrainer:

    """_summary_
    A Modified version of the BasicTrainer to handle distributed training
    """

    def __init__(self, model, tokenizer, num_epochs, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optim.AdamW(params=model.parameters())
        deepseed_plugin = DeepSpeedPlugin(
            zero_stage=2,  # Enable ZeRO Stage 2 for memory efficiency
        )

        # Initialize Accelerator with W&B logging and DeepSpeed ZeRO-2
        self.accelerator = Accelerator(log_with="wandb", deepspeed_plugin=deepseed_plugin)

    def train(self, tokenized_data):
        
        train_dataloader, eval_dataloader = self.create_dataloaders(tokenized_data=tokenized_data)
        
        # Prepare model, optimizer, and dataloaders for distributed training
        # Accelerate handles device placement automatically
        model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader
        )

        # Initialize W&B tracker for experiment logging
        self.accelerator.init_trackers(project_name="HW3-accelerate")

        for epoch_index, epoch in enumerate(range(self.num_epochs)):
            model.train()
            for batch_index, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model(**batch)
                self.accelerator.backward(outputs.loss)
                optimizer.step()
                self.accelerator.log({'epoch': epoch_index+1, 'loss': outputs.loss.item(), 'step': f'{(epoch_index+1)*(batch_index+1)}'})
            eval_metric = self.eval(model, eval_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Eval Metric: {eval_metric}")
            self.accelerator.log({'epoch': epoch_index+1, 'accuracy': eval_metric})
        
        # Cleanup: disconnect tracker after training completes
        self.accelerator.end_training()


    def eval(self, model, eval_dataloader):
        # Distributed evaluation: gather predictions from all processes
        # before computing metrics on the main process
        accuracy_metric = evaluate.load('accuracy')
        model.eval()
        all_preds=[]
        all_labels=[]
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(-1)
            preds, refs = self.accelerator.gather_for_metrics((predictions, batch['labels']))
            all_preds.append(preds)
            all_labels.append(refs)
        
        eval_metric = accuracy_metric.compute(
            predictions=torch.cat(all_preds),
            references=torch.cat(all_labels)
        )
        return eval_metric


    def create_dataloaders(self, tokenized_data):

        # Create PyTorch DataLoaders for train and validation sets
        train_dataloader, eval_dataloader = torch.utils.data.DataLoader(tokenized_data['train'], shuffle=True, batch_size=self.batch_size), torch.utils.data.DataLoader(tokenized_data['validation'], shuffle=False, batch_size=self.batch_size) 
        return train_dataloader, eval_dataloader
    
    def save(self, model):
        # Unwrap model from Accelerate wrapper before saving
        model = self.accelerator.unwrap_model(model)
        # Save model and tokenizer to HuggingFace Hub
        repo_name = os.environ.get("HF_REPO_NAME", "YOUR_USERNAME/accelerate-trainer")
        login(token=HF_TOKEN)
        model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)
