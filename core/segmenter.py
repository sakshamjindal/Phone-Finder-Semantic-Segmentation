import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
from typing import Union

from core.trainer import Trainer

class SemanticSegmenter():

  def __init__(
      self,
      model, 
      datamodule,
      params, 
      mode = "training",
      model_path = None
  ):

    """
    Initialise the model class
    Args:
        model: torch model class
        dataloader: datamodule containing train and val dataloader
        mode: "inference" or "training"
        params: dictionary of parameters for training
        model_path: path of  the model from where to load the trained model/
                    pretrained weights
    """

    self.model = model
    self.params = params

    if torch.cuda.is_available() and "cuda" in params["device"]: 
      self.device = torch.device(params["device"]) 
    else:
      self.device = torch.device("cpu")

    self.model = self.model.to(self.device)

    # Usually passed in inference mode 
    if model_path is not None:
      self.model.load_state_dict(torch.load(model_path, map_location=self.device)) 

    # Load trainer only in training mode
    if mode == "training":
      self.datamodule = datamodule
      self.semantic_trainer = Trainer(self.model, self.datamodule, self.device, self.params)

  def train_and_test(self, num_epochs : int = 5):
    """
    Train the model given the number of epochs
    """

    # here you train your model
    self.semantic_trainer.train_and_validate(num_epochs)

  def plot_train_logs(self):

    # plot train and val logs
    train_logs = self.semantic_trainer.train_logs
    val_logs = self.semantic_trainer.val_logs

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(train_logs.keys()), [x["loss"] for x in train_logs.values()], color = "blue")
    plt.title("Train Loss Plot")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(list(val_logs.keys()), [x["loss"] for x in val_logs.values()], color = "green")
    plt.title("Val Loss Plot")
    plt.grid()
    plt.show()
      
  def predict_single_image(self, image: Union[np.array, str], model_path: str = None):

    """
    Predict on a single image
    Args:
      image: numpy array of size of (H,W) or path to the image
    Returns:
      image_tensor, mask_tensor (torch.Tensor): tensors of size (H, W, 3) and (H, W)
    """

    if model_path is not None:
      self.model.load_model(model_path)

    self.model.eval()

    if isinstance(image, str):
      # open the image using PIL Image
      pil_image = Image.open(image)
    else:
      # convert to PIL image for resizing
      pil_image = Image.fromarray(image)
    
    # resize the image to dimensions of train image
    # alternatively we can pad as well
    pil_image = pil_image.resize(self.params["image_size"])
    image = np.array(pil_image)

    # convert to tensor image
    batch = T.ToTensor()(image).unsqueeze(0)
    batch = batch.to(self.device)

    with torch.no_grad():
      output = self.model(batch)

      # get probablity map
      prob_map = output.softmax(dim=1)
      prob_map = prob_map.squeeze(0)
      prob_map = prob_map.cpu().detach().numpy()

      # get output mask
      _ , pred_mask = torch.max(output,dim=1)
      pred_mask = pred_mask.cpu()

    return batch.squeeze(0).cpu(), pred_mask[0,:,:]