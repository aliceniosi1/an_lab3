import os
import shutil
import os
import shutil
import urllib.request
import zipfile

def download_dataset():
    # Percorsi
    data_dir = "data"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_dir = os.path.join(data_dir, "tiny-imagenet")

    # URL del dataset
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    # Crea la cartella 'data' se non esiste
    os.makedirs(data_dir, exist_ok=True)

    # Scarica il file zip solo se non esiste già
    if not os.path.exists(zip_path):
        print("📥 Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print("✅ File already downloaded.")

    # Estrai il dataset
    if not os.path.exists(extract_dir):
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        print("✅ Dataset already extracted.")

    # Riorganizza la cartella delle immagini di validazione
    val_dir = os.path.join(extract_dir, "tiny-imagenet-200", "val")
    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    images_dir = os.path.join(val_dir, "images")

    print("🗂️ Organizing validation images...")
    with open(annotations_path) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(images_dir, fn)
            dst = os.path.join(cls_dir, fn)
            if os.path.exists(src):
                shutil.copyfile(src, dst)

    # Rimuovi la cartella originale delle immagini non organizzate
    shutil.rmtree(images_dir)
    print("✅ Dataset ready in 'data/tiny-imagenet/tiny-imagenet-200'.")

