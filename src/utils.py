import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, device, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            y_true.extend(target.view_as(pred).cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    class_names = ['Real', 'VAE', 'GANs']
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # Normalized confusion matrix
    ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Normalized Confusion Matrix')
    plt.show()

# Function to set up the device
def set_device(device="cpu", idx=0):
    """
    Sets up the device for training or inference.

    This function checks if CUDA is available and sets the device accordingly.
    If CUDA is available, it tries to use the specified GPU. If the specified GPU
    is not available, it defaults to the first GPU. If CUDA is not available, it defaults to CPU.

    Args:
        device (str, optional): Desired device type. Default is "cpu".
        idx (int, optional): Index of the GPU to be used if available. Default is 0.

    Returns:
        torch.device: The device that will be used for training or inference.
    """
    if device != "cpu":
        if torch.cuda.device_count() > idx and torch.cuda.is_available():
            print(
                "Cuda installed! Running on GPU {} {}!".format(
                    idx, torch.cuda.get_device_name(idx)
                )
            )
            device = "cuda:{}".format(idx)
        elif torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print(
                "Cuda installed but only {} GPU(s) available! Running on GPU 0 {}!".format(
                    torch.cuda.device_count(), torch.cuda.get_device_name()
                )
            )
            device = "cuda:0"
        else:
            print("No GPU available! Running on CPU")
    return device

def set_seed(seed):
    """
    Sets a fixed value for all random seeds for reproducibility.

    Parameters:
    seed (int): Seed value for random number generators.

    Returns:
    bool: True if seeds are set successfully.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled   = False

    return True
