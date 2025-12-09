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

# TODO: Initialize the wandb library by using the login function
import wandb
wandb.login()
wandb.init(
    project="HW3",  # Specify your project
    entity="learnx") 


class BasicTrainer:

    def __init__(self, model, tokenizer, num_epochs, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(params=model.parameters())

    def train(self, tokenized_data):
        # TODO: implement train. There are a few steps to follow in the train function:
        # from the tokenized data, we need to create the data loaders
        train_dataloader, eval_dataloader = self.create_dataloaders(tokenized_data)

        # TODO: we need to project the model onto the right device (.to(device))
        self.model.to(device)

        for epoch_index, epoch in enumerate(range(self.num_epochs)):
            # TODO: set the model up in train mode: model.train()
            self.model.train()
            for batch_index, batch in enumerate(train_dataloader):
                # TODO: zero out the optimizer: optimizer.zero_grad() 
                self.optimizer.zero_grad()
                # TODO: we project the data batch onto the right device
                batch = {k: v.to(device) for k, v in batch.items()}
                # TODO: we feed the batch to the model and get the model outputs
                outputs = self.model(**batch)
                # TODO: we call the backward function on the loss function: loss.backward()
                loss = outputs.loss
                loss.backward()
                # TODO: we step the optimizer: optimizer.step()
                self.optimizer.step()
                wandb.log({'epoch': epoch_index+1, 'loss': loss.item(), 'step': f'{(epoch_index+1)*(batch_index+1)}'})
            
            eval_metric = self.eval(self.model, eval_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Eval Metric: {eval_metric}")
            wandb.log({'accuracy': eval_metric, 'epoch': epoch_index+1})

            # TODO: as an optional task, we can implement an early stopping to avoid overfitting.
        

    def eval(self, model, eval_dataloader):
        # TODO: Implement eval. The eval function computes a validation metric on the validation data. 
        # You can use the evaluate package to get access to the evaluation metric you prefer:
        accuracy_metric = evaluate.load("accuracy")

        # To accurately compute the validation metric on the validation dataset, there are a few steps:
        # - You need to set the model in eval mode: model.eval()
        model.eval()
        all_predictions = []
        all_labels = []
        # - When you iterate through each batch in the eval_dataloader, you need to project the data on the right device:
        for batch in eval_dataloader: 
            batch = {k: v.to(device) for k,v in batch.items()}
            # - You need to infer the batch with the model without aggregating the gradients: with torch.no_grad()
            with torch.no_grad():
                outputs = model(**batch) 
                preds = outputs.logits.argmax(dim=-1)
                print(f"Predictions: {preds}")
                all_predictions.append(preds)
                all_labels.append(batch['labels'])
            
        # - You need to compare the prediction to the labels to compute the metric. For example:
        eval_metric = accuracy_metric.compute(
            predictions=torch.cat(all_predictions), 
            references=torch.cat(all_labels)
        )
        
        # You can then return the evaluation metric
        return eval_metric

    def create_dataloaders(self, tokenized_data):

        # TODO: Use the torch.utils.data.DataLoader class to create iterators around the data. 
        # Make sure to create a data loader for the training data and one for the validation data.
        train_dataloader, eval_dataloader = torch.utils.data.DataLoader(tokenized_data['train'], shuffle=True, batch_size=self.batch_size), torch.utils.data.DataLoader(tokenized_data['validation'], shuffle=False, batch_size=self.batch_size) 
        return train_dataloader, eval_dataloader
    
    def save(self, model):
        # TODO: Let's save the model to the HuggingFace model hub. Implement the save function:
        # - Give a name to the repo
        repo_name = "Abhinit/HW3-basic-trainer"
        # - call the login function with your HuggingFace token
        login(token=HF_TOKEN)
        # - call push_to_hub on the model and tokenizer
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

        # TODO: instantiate as class attribute an accelerator
        self.accelerator = Accelerator(log_with="wandb", deepspeed_plugin=deepseed_plugin)
        # TODO: We then need to set up the accelerator to log with wandb. Pass the argument log_with="wandb"

        # TODO: Enable Zero Redundancy Optimizer Strategy. 
        # You may need to update the deepspeed package in the requirements.txt.

    def train(self, tokenized_data):
        
        train_dataloader, eval_dataloader = self.create_dataloaders(tokenized_data=tokenized_data)
        
        # TODO: use the prepare function to prepare the model, optimizer, train_dataloader, 
        # and eval_dataloader for distributed training. 
        # Because of this, we don't need to project the model and the data on the device, 
        # as Accelerate does it automatically.
        model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader
        )

        # TODO: Just before the code training the model, we need to initialize the tracker 
        # with the init_trackers function. You can pass additional hyperparameters to that function
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
        
        # TODO: At the end of the training, make sure to disconnect the tracker
        self.accelerator.end_training()


    def eval(self, model, eval_dataloader):
        # TODO: With Accelerate, we cannot directly compute the evaluation metrics 
        # as the data is spread across multiple machines or processes, 
        # so we need to bring it back to the main thread. 
        # To do that, we use the function gather. 
        # Modify the eval function by using the gather function 
        # before computing the evaluation metric.
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

        # TODO: Use the torch.utils.data.DataLoader class to create iterators around the data. 
        # Make sure to create a data loader for the training data and one for the validation data.
        train_dataloader, eval_dataloader = torch.utils.data.DataLoader(tokenized_data['train'], shuffle=True, batch_size=self.batch_size), torch.utils.data.DataLoader(tokenized_data['validation'], shuffle=False, batch_size=self.batch_size) 
        return train_dataloader, eval_dataloader
    
    def save(self, model):
        # TODO:  Before we can save the model, we need to undo what the prepare function did. 
        # For that, we need to call the unwrap_model function.
        #  Modify the save function by calling the unwrap_model function before saving it to the hub.
        model = self.accelerator.unwrap_model(model)
        # TODO: Let's save the model to the HuggingFace model hub. Implement the save function:
        # - Give a name to the repo
        repo_name = "Abhinit/HW3-accelerate-trainer"
        # - call the login function with your HuggingFace token
        login(token=HF_TOKEN)
        # - call push_to_hub on the model and tokenizer
        model.push_to_hub(repo_name)
        self.tokenizer.push_to_hub(repo_name)
