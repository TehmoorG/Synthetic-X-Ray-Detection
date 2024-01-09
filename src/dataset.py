from torch.utils.data import Dataset, DataLoader, default_collate
from PIL import Image, UnidentifiedImageError

class CustomHandDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            if self.transform:
                image = self.transform(image)
        except UnidentifiedImageError:
            # print(f"UnidentifiedImageError: cannot identify image file {img_path}. It will be skipped.")
            return None, None
        return image, label

def custom_collate_fn(batch):
    batch = [x for x in batch if x[0] is not None]
    return default_collate(batch)

class testDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpeg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            if self.transform:
                image = self.transform(image)
        except UnidentifiedImageError:
            return None, img_path  # Return None and image path if error occurs
        return image, os.path.basename(img_path)  # Return image and file name

def custom_collate_hands(batch):
    # Filter out pairs where the image is None (corrupted files)
    batch = [data for data in batch if data[0] is not None]
    if len(batch) == 0:  # If all images in the batch are corrupt, return an empty batch
        return torch.tensor([]), []
    return torch.utils.data.dataloader.default_collate(batch)