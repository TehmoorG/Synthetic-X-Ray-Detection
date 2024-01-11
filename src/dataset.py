from torch.utils.data import Dataset, default_collate
from PIL import Image, UnidentifiedImageError
import os
import torch


class HandDataset(Dataset):
    def __init__(self, image_paths_or_dir, labels=None, transform=None):
        if isinstance(image_paths_or_dir, list):
            self.image_paths = image_paths_or_dir
        else:
            self.image_paths = [
                os.path.join(image_paths_or_dir, img)
                for img in os.listdir(image_paths_or_dir)
                if img.endswith(".jpeg")
            ]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("L")
            if self.transform:
                image = self.transform(image)
        except UnidentifiedImageError:
            # Handle error: Log and continue
            return None, None if self.labels else img_path
        if self.labels:
            label = self.labels[idx]
            return image, label
        return image, os.path.basename(img_path)


def custom_collate_fn(batch):
    # Filter out pairs where the image is None (corrupted files)
    filtered_batch = [data for data in batch if data[0] is not None]

    # If all images in the batch are corrupt, return an empty batch
    if len(filtered_batch) == 0:
        return torch.tensor([]), []

    return default_collate(filtered_batch)
