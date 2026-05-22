import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from pathlib import Path

class RPSDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]

        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def create_dataloaders(data_dir, batch_size=32, img_size=(200,300), seed=42):
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

    # 2. Perform stratified data splitting to prevent data leakage
    # 70% Train, 30% Temporary (Val + Test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    
    # Split the temporary set equally into Validation (15%) and Test (15%)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=seed
    )

    # 3. Define the transformations using the modern torchvision.transforms.v2 API
    train_transform = v2.Compose([
        v2.Resize(img_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])

    val_test_transform = v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])

    train_dataset = RPSDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = RPSDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = RPSDataset(test_paths, test_labels, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_to_idx