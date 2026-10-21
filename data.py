import os
import shutil
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import DATA_DIR, MEAN, SEED, SIZE, STD

TRAIN_TF = transforms.Compose(
    [
        transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
EVAL_TF = transforms.Compose(
    [
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def _find_split_dirs(root):
    root = Path(root)
    train_dir = test_dir = None
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        kids = {c.name for c in p.iterdir() if c.is_dir()}
        if {"hot_dog", "not_hot_dog"} <= kids:
            if p.name == "train":
                train_dir = str(p)
            elif p.name == "test":
                test_dir = str(p)
    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not find train/ and test/ with hot_dog and not_hot_dog under {root}"
        )
    return train_dir, test_dir


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        train_dir, test_dir = _find_split_dirs(DATA_DIR)
        print(f"dataset already present: train={train_dir} test={test_dir}")
        return train_dir, test_dir
    except FileNotFoundError:
        pass

    dest = None
    try:
        import kagglehub

        dest = kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog")
        print(f"kagglehub downloaded to {dest}")
    except Exception as e:
        print(f"kagglehub failed: {e}")
        try:
            import subprocess

            zip_path = os.path.join(DATA_DIR, "hot-dog-not-hot-dog.zip")
            subprocess.check_call(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    "dansbecker/hot-dog-not-hot-dog",
                    "-p",
                    DATA_DIR,
                    "--unzip",
                ]
            )
            dest = DATA_DIR
            print(f"kaggle CLI downloaded to {dest}")
            _ = zip_path
        except Exception as e2:
            raise RuntimeError(
                "Could not download dansbecker/hot-dog-not-hot-dog. "
                "Configure Kaggle API credentials (~/.kaggle/kaggle.json)."
            ) from e2

    search_root = dest if dest else DATA_DIR
    train_dir, test_dir = _find_split_dirs(search_root)
    local_train = os.path.join(DATA_DIR, "train")
    local_test = os.path.join(DATA_DIR, "test")
    if os.path.abspath(train_dir) != os.path.abspath(local_train):
        if os.path.exists(local_train):
            shutil.rmtree(local_train)
        shutil.copytree(train_dir, local_train)
        train_dir = local_train
    if os.path.abspath(test_dir) != os.path.abspath(local_test):
        if os.path.exists(local_test):
            shutil.rmtree(local_test)
        shutil.copytree(test_dir, local_test)
        test_dir = local_test
    print(f"dataset at train={train_dir} test={test_dir}")
    return train_dir, test_dir


def _class_counts(ds):
    if hasattr(ds, "targets"):
        labels = ds.targets
    else:
        labels = [ds.dataset.targets[i] for i in ds.indices]
    names = ds.classes if hasattr(ds, "classes") else ds.dataset.classes
    c = Counter(labels)
    return {names[k]: c[k] for k in sorted(c)}


def get_loaders():
    train_dir, test_dir = download()
    base = ImageFolder(train_dir)
    idx = list(range(len(base)))
    tr, va = train_test_split(
        idx, test_size=0.2, stratify=base.targets, random_state=SEED
    )
    overlap = set(tr) & set(va)
    assert not overlap, f"train/val index overlap: {len(overlap)}"
    train_ds = Subset(ImageFolder(train_dir, TRAIN_TF), tr)
    val_ds = Subset(ImageFolder(train_dir, EVAL_TF), va)
    test_ds = ImageFolder(test_dir, EVAL_TF)

    print("train class counts:", _class_counts(train_ds))
    print("val class counts:", _class_counts(val_ds))
    print("test class counts:", _class_counts(test_ds))

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=2, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    loaders = get_loaders()
    x, y = next(iter(loaders[0]))
    print("one train batch:", tuple(x.shape), y.shape)
    assert tuple(x.shape) == (32, 3, 224, 224)
