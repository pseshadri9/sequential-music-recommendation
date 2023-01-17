import pytorch_lightning as pl
import torch

class model(pl.LightningModule): 
    def __init__(self, args): 
        super(model, self).__init__() 
          
        # Define model architecture
        self.fc = None

        # Defining learning rate
        self.lr = None
          
        # Define loss 
        self.loss = None
    
    def forward(self, x):
        pass
    
    def configure_optimizers(self):
        # Define and return the optimizer 
        pass
    
    def training_step(self, train_batch, batch_idx): 
        
        # Defining training step for our model
        pass
    
    def validation_step(self, valid_batch, batch_idx): 
        
        # Defining validation steps for our model
        pass