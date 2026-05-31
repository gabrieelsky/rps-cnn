import time
import itertools
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from src.data_loader import (
    RPSDataset,
    get_train_transform,
    get_eval_transform,
    compute_global_mean_std,
)
from src.config import IMG_HEIGHT, IMG_WIDTH
from src.data_loader import create_dataloaders
from src.config import *
from src.utils import AccLossPlot, TrainLossPlot, AverageMeter, accuracy


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value):
        if self.best_score is None or value < (self.best_score - self.min_delta):
            self.best_score = value
            self.bad_epochs = 0
            return True

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return False

def run_grid_search(model_class, param_grid, dataloader_func, data_dir, device, num_classes, n_splits=3, seed=42, patience=5, min_delta=0.0, restore_best_weights=True, max_epochs=50):
    """
    Grid search over `param_grid` using Stratified K-Fold cross-validation.

    For each hyperparameter configuration, performs `n_splits` stratified folds
    and computes mean/std of accuracy (primary) and macro-F1 (secondary) on
    the validation folds. Returns a DataFrame sorted by `mean_val_acc` descending.
    """
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    print(f"Starting Grid Search with {len(experiments)} configurations (Stratified {n_splits}-Fold CV)...\n")

    # Build full list of file paths and labels from the data directory
    file_paths = []
    labels = []

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        for file_name in os.listdir(cls_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(cls_dir, file_name))
                labels.append(class_to_idx[cls_name])

    if len(file_paths) == 0:
        raise ValueError(f"No image files found in data_dir={data_dir}")

    labels = np.array(labels)

    img_size = (IMG_HEIGHT, IMG_WIDTH)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for i, config in enumerate(experiments):
        print(f"\n{'='*40}")
        print(f"Experiment {i+1}/{len(experiments)}: {config}")
        print(f"{'='*40}")

        # Extract hyperparameters
        lr = config.get('lr', 0.001)
        batch_size = config.get('batch_size', 32)
        epochs = int(max_epochs)
        dropout = config.get('dropout_rate', 0.5)
        weight_decay = config.get('weight_decay', 0.0)

        fold_f1_scores = []
        fold_acc_scores = []

        # Perform Stratified K-Fold CV
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
            print(f"  Fold {fold_idx+1}/{n_splits} (train {len(train_idx)} / val {len(val_idx)})")

            train_paths = [file_paths[idx] for idx in train_idx]
            train_labels = [int(labels[idx]) for idx in train_idx]
            val_paths = [file_paths[idx] for idx in val_idx]
            val_labels = [int(labels[idx]) for idx in val_idx]

            # No leakage in CV: stats are computed from training fold only.
            fold_mean, fold_std = compute_global_mean_std(train_paths, img_size)
            fold_norm_mean = [fold_mean, fold_mean, fold_mean]
            fold_norm_std = [fold_std, fold_std, fold_std]

            train_transform = get_train_transform(
                img_size,
                norm_mean=fold_norm_mean,
                norm_std=fold_norm_std,
            )
            val_transform = get_eval_transform(
                img_size,
                norm_mean=fold_norm_mean,
                norm_std=fold_norm_std,
            )

            train_dataset = RPSDataset(train_paths, train_labels, transform=train_transform)
            val_dataset = RPSDataset(val_paths, val_labels, transform=val_transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model (try supporting common constructors)
            try:
                model = model_class(num_classes=num_classes, dropout_rate=dropout, input_shape=(3, IMG_HEIGHT, IMG_WIDTH)).to(device)
            except TypeError:
                try:
                    model = model_class(num_classes=num_classes, input_shape=(3, IMG_HEIGHT, IMG_WIDTH)).to(device)
                except TypeError:
                    model = model_class(num_classes=num_classes).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Train on this fold
            _, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_epochs=epochs,
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=restore_best_weights
            )

            # Evaluate on validation set (compute macro-F1)
            model.eval()
            all_preds = []
            all_labels_val = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    if device:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels_val.extend(targets.cpu().numpy())

            fold_f1 = f1_score(all_labels_val, all_preds, average='macro')
            fold_acc = accuracy_score(all_labels_val, all_preds)
            fold_f1_scores.append(fold_f1)
            fold_acc_scores.append(fold_acc)

            print(f"    Fold {fold_idx+1} macro-F1: {fold_f1:.4f}, acc: {fold_acc:.4f}")

        mean_f1 = float(np.mean(fold_f1_scores))
        std_f1 = float(np.std(fold_f1_scores))
        mean_acc = float(np.mean(fold_acc_scores))
        std_acc = float(np.std(fold_acc_scores))

        results.append({
            **config,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'mean_val_acc': mean_acc,
            'std_val_acc': std_acc
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='mean_val_acc', ascending=False).reset_index(drop=True)
    return results_df

def epoch_pass(data, model, criterion, optimizer=None, device=None, print_interval=PRINT_INTERVAL):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    # objects to store metric averages
    avg_loss = AverageMeter()
    avg_top1_acc = AverageMeter()
    avg_batch_time = AverageMeter()
    global loss_plot

    tic = time.time()
    for i, (input_tensor, target) in enumerate(data):

        if device:
            input_tensor = input_tensor.to(device)
            target = target.to(device)

        with torch.set_grad_enabled(optimizer is not None):
            output = model(input_tensor)
            loss = criterion(output, target)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        prec1 = accuracy(output, target, topk=(1,))[0]
        batch_time = time.time() - tic
        tic = time.time()

        # update
        avg_loss.update(loss.item())
        avg_top1_acc.update(prec1.item())
        avg_batch_time.update(batch_time)
        if optimizer:
            loss_plot.update(avg_loss.val)
            if i % print_interval == 0:
                loss_plot.plot()
        
        if i % print_interval == 0:
            mode_str = "TRAIN" if optimizer else "EVAL"
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:5.1f} ({top1.avg:5.1f})'.format(
                   mode_str, i, len(data), batch_time=avg_batch_time, loss=avg_loss,
                   top1=avg_top1_acc))

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg Prec@1 {top1.avg:5.2f} %\n'.format(
           batch_time=int(avg_batch_time.sum), loss=avg_loss, top1=avg_top1_acc))

    return avg_top1_acc.avg, avg_loss.avg

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=5, min_delta=0.0, restore_best_weights=True):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'best_epoch': None,
        'stopped_epoch': None
    }

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_state = None

    # init plots
    plot = AccLossPlot()
    global loss_plot
    loss_plot = TrainLossPlot()


    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        train_acc, train_loss = epoch_pass(
            data=train_loader, 
            model=model, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device
        )
        
        val_acc, val_loss = epoch_pass(
            data=val_loader, 
            model=model, 
            criterion=criterion, 
            optimizer=None, 
            device=device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        plot.update(train_loss, val_loss, train_acc, val_acc)

        if early_stopping.step(val_loss):
            best_state = copy.deepcopy(model.state_dict())
            history['best_epoch'] = epoch + 1

        if early_stopping.should_stop:
            history['stopped_epoch'] = epoch + 1
            print(f"Early stopping triggered at epoch {epoch+1} (best epoch: {history['best_epoch']}).")
            break
            
    print("\nTraining complete.")

    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)

    return model, history