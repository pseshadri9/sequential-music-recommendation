import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader 
import torch
import os
import pandas as pd
import ast
import constants

TRAIN = 'train'
TEST = 'test'
VAL = 'val'

SKIP = 'skip'
SEQUENCE = 'sequence'
NEXT_TOKEN = 'next_token'
USER_ID = 'user_id'
COLUMNS = [SEQUENCE, NEXT_TOKEN, USER_ID, SKIP]

class DataModule(pl.LightningDataModule):
    def __init__(self, filepath, batch_size):
        super().__init__()
          
        self.filepath = filepath
          
        self.batch_size = batch_size

        self.setup()
  
    def prepare_data(self, data):
        data.st.apply(lambda x: ast.literal_eval(x))
        data.skip.apply(lambda x: ast.literal_eval(x))
        st = []
        for st_it in data.st:
            st_ = self.__class__._add_special_tokens(st_it)
            st.append(st_)

        dataset = TensorDataset(torch.tensor(st),
                                torch.tensor(data.lst.tolist()),
                                torch.tensor(data.a.tolist()),
                                torch.tensor(data.buy.tolist()),
                                torch.tensor(data.skip.tolist()))
        return DataLoader(dataset, 
                          batch_size = self.batch_size)
    @staticmethod
    def _add_special_tokens(st):
        if type(st) == list:
            st_mod = [constants.CLS_ITEM]
            st_mod.extend(st)
        elif type(st) == torch.Tensor:
            st_mod = torch.cat((torch.ones(st.size(0)).unsqueeze(1) * constants.CLS_ITEM, st), dim=-1)  # add CLS token
        else:
            raise TypeError(f'Argument st should have type list or torch.Tensor but had {type(st)} instead.')
        return st_mod
  
    def setup(self, stage=None):
        
        #Load and format dataset
        dirs = {v.split['_'][1][:-3]: v for v in os.listdir(self.filepath)}

        self.train_data = self.prepare_data(pd.read_csv(dirs[TRAIN]))
        self.val_data = self.prepare_data(pd.read_csv(dirs[VAL]))
        self.test_data = self.prepare_data(pd.read_csv(dirs[TEST]))