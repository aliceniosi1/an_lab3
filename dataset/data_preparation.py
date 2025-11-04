import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_preparation(batch_size=64):
    data_root = os.path.join("data", "tiny-imagenet", "tiny-imagenet-200")

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # Trasformazioni base
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    return train_loader, val_loader

