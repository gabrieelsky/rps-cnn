import time
import itertools
import pandas as pd
import torch
import torch.nn as nn

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def epoch_pass(data, model, criterion, optimizer=None, device=None, print_interval=10):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    avg_loss = AverageMeter()
    avg_top1_acc = AverageMeter()
    avg_batch_time = AverageMeter()

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

        avg_loss.update(loss.item(), input_tensor.size(0))
        avg_top1_acc.update(prec1.item(), input_tensor.size(0))
        avg_batch_time.update(batch_time)
        
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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

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
            
    print("\nTraining complete.")
    return model, history


def run_grid_search(model_class, param_grid, dataloader_func, data_dir, device, num_classes):
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    print(f"Starting Grid Search with {len(experiments)} configurations...\n")
    
    for i, config in enumerate(experiments):
        print(f"\n{'='*40}")
        print(f"Experiment {i+1}/{len(experiments)}: {config}")
        print(f"{'='*40}")
        
        # Extract hyperparameters
        lr = config.get('lr', 0.001)
        batch_size = config.get('batch_size', 32)
        epochs = config.get('epochs', 10)
        dropout = config.get('dropout_rate', 0.5)
        
        # 1. Recreate DataLoaders with the new batch size
        train_loader, val_loader, _, _ = dataloader_func(data_dir, batch_size=batch_size)
        
        # 2. Reinitialize the model from scratch
        model = model_class(num_classes=num_classes, dropout_rate=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        # 3. Train
        _, history = train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device,
            num_epochs=epochs
        )
        
        # 4. Extract best validation performance
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1
        
        results.append({
            **config,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch
        })
        
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='best_val_acc', ascending=False)
    return results_df