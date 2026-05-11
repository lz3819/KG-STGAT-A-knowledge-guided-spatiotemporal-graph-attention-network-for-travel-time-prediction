# -*- coding: utf-8 -*-
"""
Utility Functions for KG-STGAT
"""
import torch
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(predictions, labels):
    """
    Calculate evaluation metrics
    
    Args:
        predictions: numpy array of predictions
        labels: numpy array of ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    # Flatten if needed
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # MAE
    mae = mean_absolute_error(labels, predictions)
    
    # RMSE
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    
    # MAPE
    mask = labels != 0
    mape = np.mean(np.abs((labels[mask] - predictions[mask]) / labels[mask])) * 100
    
    # R2 Score
    r2 = r2_score(labels, predictions)
    
    # Correlation coefficient
    correlation = np.corrcoef(labels, predictions)[0, 1]
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'correlation': correlation
    }
    
    return metrics


def save_checkpoint(model, optimizer, epoch, loss, save_path, filename='best_model.pth'):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: current epoch
        loss: current loss
        save_path: directory to save checkpoint
        filename: name of checkpoint file
    """
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    filepath = os.path.join(save_path, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer  
        checkpoint_path: path to checkpoint file
        
    Returns:
        epoch: epoch number from checkpoint
        loss: loss from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, loss {loss:.4f}")
    
    return epoch, loss


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience: How long to wait after last improvement
            verbose: If True, prints messages
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def masked_mae(predictions, labels, null_val=np.nan):
    """
    Masked MAE loss - ignores null values
    
    Args:
        predictions: predicted values
        labels: ground truth labels
        null_val: value to mask
        
    Returns:
        MAE with masked values
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(predictions - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_rmse(predictions, labels, null_val=np.nan):
    """
    Masked RMSE loss - ignores null values
    
    Args:
        predictions: predicted values
        labels: ground truth labels
        null_val: value to mask
        
    Returns:
        RMSE with masked values
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = (predictions - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.sqrt(torch.mean(loss))


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """Print a summary of the model architecture"""
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    total_params = 0
    trainable_params = 0
    
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        if parameter.requires_grad:
            trainable_params += params
        print(f"{name:60s} {params:>10,}")
    
    print("="*70)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*70 + "\n")


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def denormalize(data, max_val, min_val):
    """
    Denormalize data
    
    Args:
        data: normalized data
        max_val: maximum value used in normalization
        min_val: minimum value used in normalization
        
    Returns:
        Denormalized data
    """
    return data * (max_val - min_val) + min_val


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # Test metrics
    predictions = np.random.randn(100)
    labels = predictions + np.random.randn(100) * 0.1
    
    metrics = calculate_metrics(predictions, labels)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, verbose=True)
    losses = [1.0, 0.9, 0.85, 0.87, 0.88, 0.89, 0.90]
    
    print("\nTesting early stopping:")
    for i, loss in enumerate(losses):
        early_stopping(loss)
        if early_stopping.early_stop:
            print(f"Early stopping at iteration {i}")
            break
