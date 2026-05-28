import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.data_loader import create_dataloaders, get_class_mapping
from src.models import BaselineCNN, MediumCNN, MicroResNet
from src.train import train_model, run_grid_search
from src.evaluate import evaluate_model
from src.config import *

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_device()
    print(f"Hardware configuration: Using device '{device}'\n")
    
    class_mapping = get_class_mapping(DATA_DIR)
    num_classes = len(class_mapping)
    print(f"{num_classes} classes found: {class_mapping}\n")

    # --- HYPERPARAMETER TUNING ---
    param_grid = {
        'lr': [0.01, 0.001],
        'batch_size': [16, 32],
        'dropout_rate': [0.3, 0.5],
        'epochs': [5] # Keep epochs low for Grid Search to save time
    }

    print("Initiating automated Grid Search...")
    tuning_results = run_grid_search(
        model_class=ConvNet,
        param_grid=param_grid,
        dataloader_func=create_dataloaders,
        data_dir=DATA_DIR,
        device=device,
        num_classes=num_classes
    )

    # Save results for the final report
    os.makedirs(MODELS_DIR, exist_ok=True)
    tuning_results.to_csv(os.path.join(MODELS_DIR, "grid_search_results.csv"), index=False)
    
    print("\nGrid Search Completed. Top 3 Configurations:")
    print(tuning_results.head(3).to_string(index=False))

    # --- FINAL TRAINING ---
    # Extract the best parameters autonomously
    best_config = tuning_results.iloc[0]
    best_lr = best_config['lr']
    best_batch_size = int(best_config['batch_size'])
    best_dropout = best_config['dropout_rate']
    final_epochs = 20 # Train longer for the final run

    print(f"\nStarting final training with best parameters: LR={best_lr}, Batch={best_batch_size}, Dropout={best_dropout}")
    
    train_loader, val_loader, test_loader, _ = create_dataloaders(data_dir, batch_size=best_batch_size)
    
    model = ConvNet(num_classes=num_classes, dropout_rate=best_dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=best_lr)

    model, history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device,
        num_epochs=final_epochs
    )
    
    print("\nEvaluating on the test set...")
    test_loss, all_preds, all_labels = evaluate_model(
        model=model, 
        test_loader=test_loader, 
        criterion=criterion, 
        device=device,
        class_mapping=class_mapping
    )

    save_path = os.path.join(MODELS_DIR, "convnet_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights successfully saved to {save_path}")

if __name__ == "__main__":
    main()