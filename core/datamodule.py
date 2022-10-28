class DataModule:

  def __init__(
      self, 
      train_dataset, 
      val_dataset,
      batch_size: int = 4,
      **kwargs
      ):

    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.batch_size = batch_size

    if "num_workers" in kwargs:
      self.num_workers = kwargs["num_workers"]
    else:
      self.num_workers = 2

  @property
  def train_loader(self):
    return self._data_loader(
            dataset = self.train_dataset,
            shuffle = True, 
            num_worker = self.num_workers,
            batch_size = self.batch_size
    )
  @property              
  def val_loader(self):
    return self._data_loader(
            dataset = self.train_dataset,
            shuffle = False, 
            num_worker = 1,
            batch_size = 1
    )
      
  def _data_loader(
      self, 
      dataset : Dataset, 
      shuffle : bool = True, 
      num_worker : int = 2,
      batch_size: int = 4
  ) -> DataLoader :
  
    return DataLoader(
        dataset = dataset,
        shuffle = shuffle,
        batch_size = batch_size,
        num_workers = num_worker
    )