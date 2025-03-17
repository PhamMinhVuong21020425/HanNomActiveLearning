import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from typing import Type

from classify.transform import ImageTransform
from classify.dataset import SinoNomDataset, ActiveLearningDataset
from classify.models import ActiveLearningNet
from classify.config import *
from classify.utils import *


torch.set_default_device(device)
print(f'Using device: {device}')

strategy_names = [
    "RandomSampling",
    # "LeastConfidenceSampling",
    "MarginSampling",
    # "EntropySampling",
    # "RatioSampling",
    # "DropoutSampling",
    # "BALDSampling",
]


# Image transforms
resize = 64
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = ImageTransform(resize, mean, std)

# Load data
train_list = get_image_paths(DATA_DIR, phase='train')
val_list = get_image_paths(DATA_DIR, phase='val')

# Create datasets and dataloaders
train_dataset = SinoNomDataset(train_list, transform=transform, phase='train')
val_dataset = SinoNomDataset(val_list, transform=transform, phase='val')

train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True, generator=torch.Generator(device=device))
val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH, shuffle=False)

dataloader_dict = {
    'train': train_dataloader,
    'val': val_dataloader
}


def train_active_learning(strategy: Type[BaseStrategy], n_rounds=N_ROUNDS, n_samples=N_SAMPLES):
    accuracies = []
    strategy_name = strategy.__class__.__name__
    
    # Initialize wandb run for this strategy
    wandb.init(
        project=WANDB_PROJECT,
        name=f"Strategy_{strategy_name}",
        config={
            "strategy": strategy_name,
            "n_rounds": n_rounds,
            "samples_per_round": n_samples,
            "batch_size": TRAIN_BATCH,
            "n_epochs": N_EPOCHS,
            "model": "EfficientNetB7_Dropout",
            "optimizer": "AdamW",
        }
    )

    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("train_acc", summary="max")
    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("val_acc", summary="max")
    wandb.define_metric("round")
    wandb.define_metric("labeled_samples")
    
    max_val_acc = 0.0
    for rd in range(n_rounds):
        print(f'Round {rd+1}/{n_rounds}')
        
        # Train model on labeled data
        strategy.train()
        
        # Evaluate on validation set
        strategy.net.model.eval()
        
        val_loss = 0.0
        val_acc = 0.0
        for inputs, labels in dataloader_dict['val']:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = strategy.net.model(inputs)
                loss = strategy.net.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs.data, 1)
                val_acc += (preds == labels).sum().item()
        val_loss /= len(dataloader_dict['val'])
        val_acc /= len(dataloader_dict['val'].dataset)
        accuracies.append(val_acc)
        print(f'Round {rd+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        wandb.log({
            "round": rd+1,
            "labeled_samples": sum(strategy.dataset.labeled_idxs),
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        
        # Save model
        save_model(strategy.net.model, f'{WEIGHT_DIR}/{strategy_name}_last.pth')

        if max_val_acc < val_acc:
            max_val_acc = val_acc
            save_model(strategy.net.model, f'{WEIGHT_DIR}/{strategy_name}_best.pth')
            
        if rd < n_rounds - 1:
            # Query new samples
            query_indices = strategy.query(n_samples)
            strategy.dataset.label_samples(query_indices)
            #  strategy.update(query_indices)

    # Call wandb.finish() when end of a strategy        
    wandb.finish()
    return accuracies

# Run experiments
if __name__ == '__main__':
    # Run different strategies
    results = {}
    for strategy_name in strategy_names:
        print(f"\nRUNNING STRATEGY: {strategy_name}")
        
        # Initialize model and active learning components
        net = ActiveLearningNet(
            model=get_model(),
            device=device,
            criterion=nn.CrossEntropyLoss(),
            optimizer_cls=optim.AdamW,
            optimizer_params={"lr": LEARNING_RATE, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        )

        # Create active learning dataset
        all_dataset = ActiveLearningDataset(train_dataset, initial_labeled=N_INIT_LABELED)
        
        # Training
        strategy = get_strategy(strategy_name)(all_dataset, net)
        accuracies = train_active_learning(strategy)
        results[strategy_name] = accuracies
