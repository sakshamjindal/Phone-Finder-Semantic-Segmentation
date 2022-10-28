import numpy as np
from collections import defaultdict


class MetricMonitor:

  def __init__(self):
    """Module to monitor traininig metric"""

    self.reset()

  def reset(self):
    self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

  def update(self, metric_name, val, counter_update = 1):
    metric = self.metrics[metric_name]

    metric["val"] += val
    metric["count"] += counter_update
    metric["avg"] = metric["val"] / metric["count"]


class EarlyStopping:

  def __init__(self, patience = 7, delta = 0, path ="best_model.pth"):
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Args:
        patience (int): How long to wait after last time validation loss improved.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0   
        path (str) : Path to save the model
    """
    self.patience = patience
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta
    self.patience_mode = False

  def __call__(self, val_loss):

    score = -val_loss

    if self.best_score is None:
        self.best_score = score
    elif score < self.best_score + self.delta:
        self.patience_mode = True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
    else:
        self.patience_mode = False
        self.best_score = score
        self.counter = 0
        self.val_loss_min = val_loss
        