import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B7_Weights
from tqdm import tqdm

from .hyperparams import *
from .config import device

class ActiveLearningNet:
    def __init__(self, model, device, criterion, optimizer_cls, optimizer_params):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params
        
    def train(self, data):
        n_epochs = N_EPOCHS
        dataloader = DataLoader(data, batch_size=TRAIN_BATCH, shuffle=True, generator=torch.Generator(device=device))
        
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
        
        for epoch in range(n_epochs):
            self.model.train()
            
            train_loss = 0.0
            train_acc = 0.0
            total = 0.0
            for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate the loss
                loss = self.criterion(outputs, labels)
                train_loss += loss.item() * inputs.size(0)
                
                # Calculate the accuracy
                _, preds = torch.max(outputs.data, 1)
                train_acc += (preds == labels).sum().item()
                total += labels.size(0)

                # Backpropagation
                loss.backward()

                # Update the weights
                self.optimizer.step()
                
            train_loss /= len(dataloader)
            train_acc /= total
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
                
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
            })
    
    def predict(self, data):
        self.model.eval()
        dataloader = DataLoader(data, batch_size=PREDICT_BATCH, shuffle=False, generator=torch.Generator(device=device))
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc='Predicting unlabeled data...'):
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                _, preds = outputs.max(1)
                predictions.append(preds)
                
        return torch.cat(predictions)
    
    def predict_prob(self, data):
        self.model.eval()
        dataloader = DataLoader(
            data,
            batch_size=PREDICT_BATCH,
            shuffle=False,
            pin_memory=True,
            num_workers=N_WORKERS,
            generator=torch.Generator(device=device)
        )
        
        # Pre-allocate a list with approximate capacity to avoid resizing
        probabilities = [None] * len(dataloader)
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(tqdm(dataloader, desc='Predicting unlabeled data...')):
                # Move inputs to device and use non_blocking for asynchronous transfer
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                probabilities[i] = probs
                
        return torch.cat(probabilities)
    
    def predict_prob_dropout(self, data, n_infer=N_INFER):
        # Use dropout in inference
        self.model.train()
        dataloader = DataLoader(
            data,
            batch_size=PREDICT_BATCH,
            shuffle=False,
            pin_memory=True,
            num_workers=N_WORKERS,
            generator=torch.Generator(device=device)
        )

        # Initialize probabilities tensor with proper shape
        sample_input = next(iter(dataloader))[0][:1].to(self.device)
        num_classes = self.model(sample_input).shape[1]
        probabilities = torch.zeros([len(data), num_classes]).to(self.device)

        # Perform multiple inference passes with dropout
        for _ in tqdm(range(n_infer), desc='Predicting unlabeled data...'):
            batch_probs = []
            with torch.no_grad():
                for inputs, _ in dataloader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    outputs = self.model(inputs)
                    probs = F.softmax(outputs, dim=1)
                    batch_probs.append(probs)

            # Accumulate probabilities for this inference pass
            current_probs = torch.cat(batch_probs)
            probabilities += current_probs

        # Average the probabilities across all inference passes
        probabilities /= n_infer
        return probabilities

    def predict_prob_bald(self, data, n_infer=N_INFER):
        # Use dropout in inference
        self.model.train()
        dataloader = DataLoader(
            data,
            batch_size=PREDICT_BATCH,
            shuffle=False,
            pin_memory=True,
            num_workers=N_WORKERS,
            generator=torch.Generator(device=device)
        )

        # Initialize probabilities tensor with proper shape for BALD
        sample_input = next(iter(dataloader))[0][:1].to(self.device)
        num_classes = self.model(sample_input).shape[1]
        probabilities = torch.zeros([n_infer, len(data), num_classes], device=self.device)
    
        # Perform multiple inference passes with dropout
        for i in tqdm(range(n_infer), desc='Predicting unlabeled data...'):
            batch_idx = 0
            with torch.no_grad():
                for inputs, _ in dataloader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    outputs = self.model(inputs)
                    probs = F.softmax(outputs, dim=1)

                    # Calculate batch size and indices
                    batch_size = inputs.size(0)
                    start_idx = batch_idx * PREDICT_BATCH
                    end_idx = min(start_idx + batch_size, len(data))
                    
                    # Store directly into the pre-allocated tensor
                    probabilities[i, start_idx:end_idx] = probs
                    batch_idx += 1

        return probabilities

class EfficientNetB7_Dropout(nn.Module):
    def __init__(self, num_classes, dropout_rates=[0.0, 0.1, 0.3, 0.4]):
        super().__init__()
        
        # Load pretrained model
        self.effnet = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        self.effnet.features[0][0] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Add dropout layers after MBConv blocks
        for idx, block in enumerate(self.effnet.features):
            if isinstance(block, nn.Sequential):
                # Increase dropout rate with depth
                layer_position = idx / len(self.effnet.features)
                dropout_rate = dropout_rates[int(layer_position * len(dropout_rates))]
                
                # Add dropout after each block
                block.add_module('dropout', nn.Dropout2d(p=dropout_rate))
        
        # Replace the classifier with a new one
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(2560, num_classes, bias=True)
        )

    def forward(self, x):
        return self.effnet(x)
