import pytorch_lightning as pl
from torch.utils.data import DataLoader 

class DataModule(pl.LightningDataModule):
    def __init__(self, filepath, batch_size):
        super().__init__()
          
        self.filepath = filepath
          
        self.batch_size = batch_size
  
    def prepare_data(self):
        pass
  
    def setup(self, stage=None):
        
        #Load and format dataset
        self.train_data, self.val_data, self.test_data = None
  
    def train_dataloader(self):
        
          # Generating train_dataloader
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size)
  
    def val_dataloader(self):
        
          # Generating val_dataloader
        return DataLoader(self.val_data,
                          batch_size = self.batch_size)
  
    def test_dataloader(self):
        
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size = self.batch_size)