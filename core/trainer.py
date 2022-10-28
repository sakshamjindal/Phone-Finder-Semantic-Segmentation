import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import time
import copy
from collections import defaultdict

from core.callbacks import EarlyStopping, MetricMonitor


class Trainer():
    
  def __init__(self, model, datamodule, device, params):

    """
    Trainer engine to run training loops
    """
      
    self.model = model,
    self.datamodule = datamodule
    self.device = device
    self.params = params

    if isinstance(self.model, tuple):
      self.model = self.model[0]

    if "ce_weights" in self.params:
      weight = self.params["ce_weights"]
      weight = torch.FloatTensor(weight)
      weight = weight.to(self.device)
      self.criterion = nn.CrossEntropyLoss(weight=weight)
    else:
      self.criterion = nn.CrossEntropyLoss()

    self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    self.train_logs = {}
    self.val_logs = {}

  def train(self, epoch):

    """
    Run one epoch of training loop
    """
    metric_monitor = MetricMonitor()
    self.model.train()
    for i, (images, target) in enumerate(self.datamodule.train_loader):
        images = images.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        output = self.model(images)
        loss = self.criterion(output, target)

        _, preds = torch.max(output, 1)
        corrects = torch.sum(preds == target.data)

        metric_monitor.update("Loss", loss.item(), 1)
        metric_monitor.update("Accuracy", corrects, torch.numel(preds))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    if (epoch - 1) % self.params["log_frequency"] == 0:
      print(
          "Training:  Loss: {loss_metric}, Pixel Accuracy: {accuracy_metric}".format(
                epoch=epoch,loss_metric=metric_monitor.metrics["Loss"]["avg"],
                accuracy_metric = metric_monitor.metrics["Accuracy"]["avg"])
      )

    self.train_logs[epoch] = {
        "epoch" : epoch,
        "loss" : metric_monitor.metrics["Loss"]["avg"],
        "pixel_accuracy" : metric_monitor.metrics["Accuracy"]["avg"]
    }

  def validate(self, epoch):

    """
    Run one epoch of validation loop
    """
      
    metric_monitor = MetricMonitor()
    self.model.eval()
    with torch.no_grad():
      for i, (images, target) in enumerate(self.datamodule.val_loader):
        images = images.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        output = self.model(images)
        loss = self.criterion(output, target)

        _, preds = torch.max(output, 1)
        corrects = torch.sum(preds == target.data)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", corrects, torch.numel(preds))

    if (epoch - 1) % self.params["log_frequency"] == 0:
      print(
          "Validation:  Loss: {loss_metric}, Pixel Accuracy: {accuracy_metric}".format(
                epoch=epoch,loss_metric=metric_monitor.metrics["Loss"]["avg"],
                accuracy_metric = metric_monitor.metrics["Accuracy"]["avg"])
      )

    self.val_logs[epoch] = {
        "epoch" : epoch,
        "loss" : metric_monitor.metrics["Loss"]["avg"],
        "pixel_accuracy" : metric_monitor.metrics["Accuracy"]["avg"]
    }

  def train_and_validate(self, num_epochs):

    """
    Run multiple epochs of training and validation over the dataset
    """

    early_stopping = EarlyStopping(patience = 7, delta = 0.005)

    for epoch in range(1, num_epochs + 1):

      if (epoch - 1) % self.params["log_frequency"] == 0:
        log = True
      else:
        log = False

      if log:
        print("---------------------------------------")
        print("Epoch:  {epoch}".format(epoch = epoch))

      if early_stopping.early_stop is False:
        self.train(epoch)
        self.validate(epoch)
        early_stopping(self.val_logs[epoch]["loss"])
        if log:
          print("Best loss till now: {}".format(-early_stopping.best_score))
        if early_stopping.patience_mode is False:
          self.save_checkpoint()
          if log:
            print("Saving checkpoint at the epoch")
        else:
          if log:
            print("Patiently waiting for the loss to decrease")
      else:
        print("Early stopping at epoch : {}".format(epoch))
        print("Best loss till now: {}".format(-early_stopping.best_score))
        break

  def save_checkpoint(self, path = "best_model.pth"):
    """Saves model when validation loss decrease."""
    torch.save(self.model.state_dict(), path)

  def load_model(self, model_path = "best_model.pth"):
    """Loads the model from model path"""
    self.model.load_state_dict(torch.load(path))