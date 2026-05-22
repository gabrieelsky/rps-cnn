from src.data_loader import create_dataloaders
import matplotlib.pyplot as plt
import torchvision


data_dir = "data/raw"

train_loader, val_loader, test_loader, class_mapping = create_dataloaders(data_dir, batch_size=16)

print(f"Mappatura classi: {class_mapping}")
print(f"Numero di batch: Train={len(train_loader)} | Val={len(val_loader)} | Test={len(test_loader)}\n")

# Estraiamo il primo batch dal train_loader
images, labels = next(iter(train_loader))

print("--- Analisi del Batch ---")
print(f"Forma del tensore immagini: {images.shape}")
print(f"Forma del tensore etichette: {labels.shape}")
print(f"Tipo di dato: {images.dtype}")
print(f"Range dei pixel: min={images.min().item():.4f}, max={images.max().item():.4f}")

def show_batch(images, labels, class_mapping):
    # Invertiamo il dizionario per risalire dal numero al nome della classe
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # Prendiamo solo le prime 8 immagini del batch per comodità visiva
    images_subset = images[:8]
    labels_subset = labels[:8]
    
    # Creiamo una griglia 2x4
    grid = torchvision.utils.make_grid(images_subset, nrow=4)
    
    plt.figure(figsize=(12, 6))
    
    # PyTorch usa (Canali, Altezza, Larghezza)
    # Matplotlib richiede (Altezza, Larghezza, Canali). Usiamo permute per riordinare.
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Sample Training Batch (con Augmentation)")
    plt.show()
    
    # Stampiamo le etichette corrispondenti
    labels_nomi = [idx_to_class[lbl.item()] for lbl in labels_subset]
    print(f"Etichette: {labels_nomi}")

show_batch(images, labels, class_mapping)