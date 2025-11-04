import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
def data_preparation(batch_size=64):
    data_root = os.path.join("data", "tiny-imagenet", "tiny-imagenet-200")

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # Trasformazioni base
    transform = T.Compose([
        T.Resize((64, 64)),  # adattato per reti come ResNet
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    tiny_imagenet_dataset_train = ImageFolder(root=train_dir, transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root=val_dir, transform=transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        tiny_imagenet_dataset_train, batch_size=batch_size,
        shuffle=True, num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        tiny_imagenet_dataset_val, batch_size=batch_size,
        shuffle=False, num_workers=8
    )

    print(f"✅ Loaded {len(tiny_imagenet_dataset_train)} training images and {len(tiny_imagenet_dataset_val)} validation images.")
    return train_loader, val_loader