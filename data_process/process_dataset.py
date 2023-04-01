import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, Dataset 
import torch
import os
import pandas as pd
import ast
from data_process import constants
from tqdm import tqdm

torch.manual_seed(0)

TRAIN = 'train'
TEST = 'test'
VAL = 'val'

SKIP = 'skip'
SEQUENCE = 'sequence'
NEXT_TOKEN = 'next_token'
USER_ID = 'user_id'
COLUMNS = [SEQUENCE, NEXT_TOKEN, USER_ID, SKIP]

FILE_EXT = '.csv'

#Reserved Sequence Tokens
PAD = 0
CLS = 1
MSK = 2

NUM_RESERVED_TOKENS = 3

#Reserved Skip Tokens
SKIP_PAD = 3

class LfMDataModule(pl.LightningDataModule):
    def __init__(self, filepath, batch_size):
        super().__init__()
          
        self.filepath = filepath
          
        self.batch_size = batch_size

        self.setup()
  
    def prepare_data(self, data):
        data.st = data.st.apply(lambda x: ast.literal_eval(x))
        data.skip = data.skip.apply(lambda x: ast.literal_eval(x))
        st = []
        u_ids = []
        for st_it in data.st:
            u_ids.append(st_it[0])
            st_ = self.__class__._add_special_tokens(st_it[1:])
            st.append(st_)

        dataset = TensorDataset(torch.tensor(st),
                                torch.tensor(u_ids),
                                torch.tensor(data.lst.tolist()),
                                torch.tensor(data.a.tolist()),
                                torch.tensor(data.buy.tolist()),
                                torch.tensor(data.skip.tolist()))
        return DataLoader(dataset, 
                          batch_size = self.batch_size, shuffle=True)
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
        dirs = {v.split('_')[-1][:-4]: v for v in os.listdir(self.filepath) if v.endswith(FILE_EXT)}
        print(dirs)
        self.train_data = self.prepare_data(pd.read_csv(self.filepath + dirs[TRAIN]))
        self.val_data = self.prepare_data(pd.read_csv(self.filepath + dirs[VAL]))
        self.test_data = self.prepare_data(pd.read_csv(self.filepath + dirs[TEST]))

class SpotifyDataModule(pl.LightningDataModule):
    def __init__(self, filepath, batch_size):
        super().__init__()
          
        self.filepath = filepath
          
        self.batch_size = batch_size

        self.setup()
    
    def load_data(self, filepath):
        sessions = list()
        skips = list()
        vocab = set()
        session_ids = list()
        pbar = tqdm(sorted(os.listdir(filepath)))
        for x in pbar:
            sessions_i, skips_i, vocab_i, session_ids_i = self.load_csv(os.path.join(filepath, x))

            sessions.extend(sessions_i)
            skips.extend(skips_i)
            vocab = vocab.union(vocab_i)
            session_ids.extend(session_ids_i)
            pbar.set_description(f'vocab: {len(vocab)} sessions:{len(sessions)}')

            if len(sessions) > 10000000: #stop when 10M sessions are sampled
                break
        
        vocab = {v :k + NUM_RESERVED_TOKENS for k, v in enumerate(vocab)}
        sessions = [self.append_special_tokens([vocab[x] for x in session]) for session in sessions]
        skips = [self.skip_preprocess(skip) for skip in skips]
        return sessions, skips, vocab, session_ids
    
    def zeropad(self, l, length, padding_val = PAD):
        if len(l) >= length:
            return l
        return l + [padding_val] * (length - len(l))
    
    def append_special_tokens(self, l):
        return [CLS] + self.zeropad(l, 20) #size of each sequence
    
    def skip_preprocess(self, l, binary=True): #do not consider weak skips + pad. binary=True ignores severity of skip
        if binary:
            return [SKIP_PAD] + self.zeropad([1 if x > 1 else 0 for x in l], 20, padding_val=SKIP_PAD)

        return [SKIP_PAD] + self.zeropad([x - 1 if x != 0 else 0 for x in l], 20, padding_val=SKIP_PAD)

    def load_csv(self, f):
        cols = ['session_position', 'track_id_clean', 'skip_level', 'session_id'] #+ ['skip_1', 'skip_2', 'skip_3']
        df = pd.read_csv(f)

        #0 = no skip, 3 = strongest skip
        df['skip_level'] = ((df['skip_1'].astype(int) + df['skip_2'].astype(int) + df['skip_3'].astype(int)))
        df = df[cols]

        groups = df.groupby(['session_id'])
        df = df.loc[groups['skip_level'].transform('max') > 1,:].sort_values(by = 
                ['session_id','session_position'], axis=0) #Do not consider sessions with no skips
        groups = df.groupby(['session_id'])

        sessions = groups['track_id_clean'].apply(list)
        skips = groups['skip_level'].apply(list)
        vocab = set(df['track_id_clean'].to_list())
        session_ids = [name for name,unused_df in groups] #preserve ordering of session_ids

        '''
        c = 0
        for s, df_s in tqdm(df.groupby(['session_id'], sort=False), leave=False):
            df_s = df_s.sort_values(by = ['session_position'], axis=0)


            sessions_2 = df_s['track_id_clean'].to_list()
            skips_2 = df_s['skip_level'].to_list()
            #vocab = vocab.union(set(sessions[-1]))
            #session_ids.append(s)
            check_sessions = all([x == y for x, y in zip(sessions_2, sessions[c])])
            check_skips = all([x == y for x, y in zip(skips_2, skips[c])]) 
            check_id = session_ids[c] == s
            c += 1

            print(check_sessions and check_skips and check_id)
        '''
        return sessions, skips, vocab, session_ids
  
    def setup(self, stage='TRAIN', select_criterion=None):
        
        #Load and format dataset
        if select_criterion:
            sessions, skips, vocab, session_ids = select_criterion(*self.load_data(self.filepath))
        else:
            sessions, skips, vocab, session_ids = self.load_data(self.filepath)
        
        if stage == 'TRAIN':
            self.train_data, self.val_data, self.vocab = self.split_data(sessions, skips, vocab, stage=stage)
        else:
            self.test_data, self.vocab = self.split_data(sessions, skips, vocab, stage=stage)
        
    
    def split_data(self, sessions, skips, vocab, stage='TRAIN'):
        sessions = torch.Tensor(sessions) #N x 20
        skips = torch.Tensor(skips)
        vocab = torch.Tensor(sorted(vocab.values()))
        targets = sessions[:, -1] #last element
        if stage == 'TEST':
            return TensorDataset(sessions[:, :-1], skips[:, :-1], targets), vocab
        else:
            indices = torch.randperm(sessions.shape[0])
            val = TensorDataset(sessions[indices[:2 * sessions.shape[0] // 10], :-1], 
                skips[indices[:2 * sessions.shape[0] // 10], :-1], targets[:2 * sessions.shape[0] // 10])

            train = TensorDataset(sessions[indices[2 * sessions.shape[0] // 10:], :-1], 
                skips[indices[2 * sessions.shape[0] // 10:], :-1], targets[2 * sessions.shape[0] // 10:])
            
            return train, val, vocab
    
    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.batch_size)

class SeqDataset(Dataset):
    def __init__(self, *tensors):
        tensors = tensors
        self.shape = (len(tensors), *(x.shape for x in tensors))
    def __get_item__(self, idx):
        return (x[idx] for x in tensors)
