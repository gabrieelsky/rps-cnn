import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os
import csv
from torchvision.utils import save_image

def evaluate_model(model, test_loader, criterion, device, class_mapping,
                   misclassified_log='logs/misclassified_log.csv',
                   save_images=False,
                   misclassified_dir='logs/misclassified',
                   max_save=200,
                   return_misclassified=False):
    """
    Performs the final evaluation of the model on the test dataset.
    Generates statistical metrics and plots the Confusion Matrix.
    """
    print("\nStarting final evaluation on test set...")
    
    # Set the model to evaluation mode (disables Dropout and Batch Norm updates)
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    mis_files = []
    mis_preds = []
    mis_labels = []

    # prepare logging if requested
    log_fp = None
    if misclassified_log:
        log_dir = os.path.dirname(misclassified_log) or "."
        os.makedirs(log_dir, exist_ok=True)
        write_header = not os.path.exists(misclassified_log)
        log_fp = open(misclassified_log, 'a', newline='')
        log_writer = csv.writer(log_fp)
        if write_header:
            log_writer.writerow(['filename', 'predicted_label', 'ground_truth_label'])

    # Disable gradient calculation for inference (saves memory and compute)
    with torch.no_grad():
        running_idx = 0
        try:
            dataset_file_paths = test_loader.dataset.file_paths
        except Exception:
            dataset_file_paths = None

        for inputs, labels in test_loader:
            if device:
                inputs = inputs.to(device)
                labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)

            # Get the predicted classes
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and true labels for scikit-learn metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            

            # collect/save misclassified examples and log filename,pred,true
            mismatch = (preds != labels)
            for bi in range(inputs.size(0)):
                if not mismatch[bi]:
                    continue

                # determine filename: prefer dataset file path when available
                filename = None
                if dataset_file_paths is not None:
                    idx = running_idx + bi
                    if 0 <= idx < len(dataset_file_paths):
                        filename = dataset_file_paths[idx]

                to_log_name = filename or f"idx_{running_idx + bi}"

                saved_path = None
                if save_images and misclassified_dir is not None and len(mis_files) < max_save:
                    try:
                        os.makedirs(misclassified_dir, exist_ok=True)
                        img_tensor = inputs[bi].cpu()
                        true_idx = int(labels[bi].cpu().item())
                        pred_idx = int(preds[bi].cpu().item())
                        true_name = next((k for k,v in class_mapping.items() if v==true_idx), str(true_idx))
                        pred_name = next((k for k,v in class_mapping.items() if v==pred_idx), str(pred_idx))
                        fname = f"true_{true_name}_pred_{pred_name}_{len(mis_files)}.png"
                        saved_path = os.path.join(misclassified_dir, fname)
                        save_image(img_tensor, saved_path)
                        mis_files.append(saved_path)
                        mis_preds.append(pred_idx)
                        mis_labels.append(true_idx)
                    except Exception:
                        saved_path = None

                pred_idx = int(preds[bi].cpu().item())
                true_idx = int(labels[bi].cpu().item())
                if log_fp is not None:
                    log_writer.writerow([to_log_name, pred_idx, true_idx])
                    log_fp.flush()
                # Keep in-memory lists if requested to return later
                if return_misclassified and len(mis_files) < max_save:
                    # if saved_path is present we already appended to mis_files above; otherwise append a placeholder
                    if saved_path:
                        pass
                    else:
                        mis_files.append(to_log_name)
                        mis_preds.append(pred_idx)
                        mis_labels.append(true_idx)

            
            # advance running index by batch size
            running_idx += inputs.size(0)

        # Calculate average test loss
    test_loss = running_loss / len(test_loader.dataset)
    
    # Extract class names ordered by their index
    class_names = [name for name, idx in sorted(class_mapping.items(), key=lambda item: item[1])]

    # 1. Print Standard Metrics
    test_acc = accuracy_score(all_labels, all_preds)

    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    print(f"Average Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 2. Generate and display the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Plotting configuration
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Test Set - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    if log_fp is not None:
        log_fp.close()

    if return_misclassified:
        return test_loss, test_acc, all_preds, all_labels, (mis_files, mis_preds, mis_labels)

    return test_loss, test_acc, all_preds, all_labels