import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, test_loader, criterion, device, class_mapping):
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

    # Disable gradient calculation for inference (saves memory and compute)
    with torch.no_grad():
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

    # Calculate average test loss
    test_loss = running_loss / len(test_loader.dataset)
    
    # Extract class names ordered by their index
    class_names = [name for name, idx in sorted(class_mapping.items(), key=lambda item: item[1])]

    # 1. Print Standard Metrics
    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    print(f"Average Test Loss: {test_loss:.4f}")
    
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

    return test_loss, all_preds, all_labels