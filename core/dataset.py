# make a dataset from of iphone images pasted on background images

from typing import Optional, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from albumentations.pytorch import ToTensorV2
import torchvision
import torchvision.transforms as transforms

import albumentations as A

def train_augmentations(image_size=(256, 256)):
    return A.Compose([
        # Geometric Transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5),
        A.Affine(p=0.5), 
        A.Perspective(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        # Image Intensity Transforms
        # A.RandomBrightnessContrast(p=0.5, brightness_limit=(0.5, 1.5), contrast_limit=0.5),
        A.ColorJitter(brightness=(0.75, 1.25), contrast=(0.75, 1.25), saturation=(0.5, 1)),
        # Blur Transforms
        A.RandomGamma(p=0.25),
        A.Blur(p=0.5, blur_limit = 3),
        A.GaussianBlur(p = 0.5)
    ], p = 1)

def test_augmentations(image_size=(256, 256)):
    return A.Compose([
        A.RandomCrop(width = image_size[0], height = image_size[1], p=1),
    ], p = 1)

class IphoneDataset(Dataset):

  def __init__(self, dataset: List, transforms = None):

    """
    Initialized
    """
    super().__init__()

    self.dataset = dataset
    self._transforms = transforms

  def apply_normalization(self, img):
      
    # Normalisation turned off
    # Needs to be switched on after calculating dataset mean and std of dataset
    transform = T.Compose([
                    T.ToTensor(),
                    # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
      
    return transform(img)

  def __getitem__(self, index):

    """
    Get a single image / label pair.
    """
     
    # Extract image and apply augmentations
    image, mask = self.dataset[index]
    mask = mask.astype(int)
    
    augmented = self._transforms(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']

    # Apply normalisation to image
    image = self.apply_normalization(image)

    return image, torch.Tensor(mask).long()

  def __len__(self) -> int:
  
    return len(self.dataset)
