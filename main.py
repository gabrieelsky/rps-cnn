import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.data_loader import create_dataloaders, get_class_mapping
from src.models import BaselineCNN
from src.train import train_model, run_grid_search
from src.evaluate import evaluate_model
from src.config import *
from src.plot_utils import AccLossPlot, TrainLossPlot

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_device()
    print(f"Hardware configuration: Using device '{device}'\n")

    class_mapping = get_class_mapping(RAW_DATA_DIR)
    num_classes = len(class_mapping)
    print(f"{num_classes} classes found: {class_mapping}\n")

    # Define hyperparameter grid for Stratified K-Fold CV using run_grid_search
    param_grid = {
        'lr': [1e-2, 1e-3, 3e-4],
        'batch_size': [16, 32],
        'dropout_rate': [0.3, 0.5],
        'weight_decay': [0, 1e-4],
        'epochs': [20]
    }

    print('Initiating Stratified K-Fold Grid Search (this may take a while)...')
    tuning_results = run_grid_search(BaselineCNN, param_grid, None, RAW_DATA_DIR, device=device, num_classes=num_classes, n_splits=3, seed=RANDOM_SEED, patience=3, min_delta=1e-4)
    tuning_results.to_csv('saved_models/grid_search_results.csv', index=False)
    print('Grid search complete. Top results:')
    print(tuning_results.head())

    # Extract best configuration from tuning results
    #best_config = tuning_results.iloc[0]
    best_config = {
        'lr': 1e-3,
        'batch_size': 16,
        'dropout_rate': 0.5,
        'weight_decay': 1e-4,
        'epochs': 20
    }

    best_lr = float(best_config['lr'])
    best_batch_size = int(best_config['batch_size'])
    best_dropout = float(best_config.get('dropout_rate', 0.0))
    best_weight_decay = float(best_config.get('weight_decay', 0.0))
    best_epochs = int(best_config.get('epochs', 5))
    final_epochs = 20  # Train longer for the final run
    print(f"Selected best config: lr={best_lr}, batch_size={best_batch_size}, dropout={best_dropout}, weight_decay={best_weight_decay}, epochs={best_epochs}")

    print(f"\nStarting final training with best parameters: LR={best_lr}, Batch={best_batch_size}, Dropout={best_dropout}")

    train_loader, val_loader, test_loader, _ = create_dataloaders(RAW_DATA_DIR, batch_size=best_batch_size)

    model = BaselineCNN(num_classes=num_classes, input_shape=(3, IMG_HEIGHT, IMG_WIDTH)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)

    # %matplotlib notebook
    # plotter = AccLossPlot()

    model, history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device,
        num_epochs=final_epochs,
        patience=5,
        min_delta=1e-4
        # plotter=plotter
    )

    print("\nEvaluating on the test set...")
    test_loss, all_preds, all_labels = evaluate_model(
        model=model, 
        test_loader=test_loader, 
        criterion=criterion, 
        device=device,
        class_mapping=class_mapping,
    )

    save_path = os.path.join(MODELS_DIR, "convnet_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights successfully saved to {save_path}")

if __name__ == "__main__": 
    main()