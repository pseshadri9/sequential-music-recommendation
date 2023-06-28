import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
from random import Random
import os
import pandas as pd
import ast
from data_process import constants
from tqdm import tqdm

torch.manual_seed(0)
r = Random(0)

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
MSK = 3

NUM_RESERVED_TOKENS = 3

#Reserved Skip Tokens
SKIP_PAD = 2

'''
class TensorDataset(Dataset):
    def __init__(self, *tensors):
        tensors = tensors
        self.shape = (len(tensors), *(x.shape for x in tensors))
    def __get_item__(self, idx):
        return (x[idx] for x in tensors)
    def __len__(self):
        return 
'''

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
    def __init__(self, filepath, batch_size, max_seq_len=20, preprocess=None):
        super().__init__()
          
        self.filepath = filepath
        self.preprocess = preprocess
          
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.setup()
    
    def load_data(self, filepath):
        sessions = list()
        skips = list()
        vocab = set()
        session_ids = list()
        total_interactions = 0
        #pbar = tqdm(sorted(os.listdir(filepath)))
        files = os.listdir(filepath)
        r.shuffle(files)
        pbar = tqdm(files)
        for x in pbar:
            sessions_i, skips_i, vocab_i, session_ids_i = self.load_csv(os.path.join(filepath, x))
            total_interactions += sum([len(x) for x in sessions_i])
            sessions.extend(sessions_i)
            skips.extend(skips_i)
            vocab = vocab.union(vocab_i)
            session_ids.extend(session_ids_i)
            pbar.set_description(f'vocab: {len(vocab)} sessions:{len(sessions)} interactions: {total_interactions}')

            if total_interactions > 2000000: #stop when 10M sessions are sampled
                break
        
        return self.preprocess_data(sessions, skips, vocab, session_ids)
    
    def preprocess_data(self, sessions, skips, vocab, session_ids):
        vocab = {v :k + NUM_RESERVED_TOKENS for k, v in enumerate(vocab)}
        if self.preprocess is None:
            sessions = [[vocab[x] for x in session] for session in sessions]
        elif self.preprocess == 'positive':
            temp = list()
            for session, skip in zip(sessions, skips):
                temp.append(list())
                for idx, (_ , _) in enumerate(zip(session, skip)):
                    if skip[idx] > 1:
                        temp[-1].append(vocab[session[idx]])
            sessions = temp
        elif self.preprocess == 'contrastive':
            sessions = [[vocab[x] for x in session] for session in sessions]
            for idx, skip in enumerate(skips):
                for idx, _ in enumerate(skip):
                    count = 1
                    while (idx + count) < len(skip) and skip[idx + count] <= 1:
                        count += 1
                    skip[idx] = idx + count
                skips[idx] = skip
                    
        skips = [self.skip_preprocess(skip) for skip in skips]

        return sessions, skips, vocab, session_ids



    def zeropad(self, l, length, padding_val = PAD):
        if len(l) >= length:
            return l
        return l + [padding_val] * (length - len(l))
    
    def append_special_tokens(self, l):
        return self.zeropad(l, self.max_seq_len) #size of each sequence
    
    def skip_preprocess(self, l, binary=True, unidirectional = False): #do not consider weak skips + pad. binary=True ignores severity of skip
        if binary:
           seq = self.zeropad([0 if x > 1 else 1 for x in l], self.max_seq_len, padding_val=SKIP_PAD) #0 = SKIP, 1 = POSITIVE
        elif self.preprocess == 'contrastive':
            seq = self.zeropad(l, self.max_seq_len, padding_val=SKIP_PAD)
        else:
            seq = self.zeropad([x - 1 if x != 0 else 0 for x in l], self.max_seq_len, padding_val=SKIP_PAD)
        return seq #+ [SKIP_PAD] if unidirectional else [SKIP_PAD] + seq
    
    def next_positive_hit(self, skip):
        pass

    def load_csv(self, f):
        cols = ['session_position', 'track_id_clean', 'skip_level', 'session_id'] #+ ['skip_1', 'skip_2', 'skip_3']
        df = pd.read_csv(f)

        #0 = no skip, 3 = strongest skip
        df['skip_level'] = ((df['skip_1'].astype(int) + df['skip_2'].astype(int) + df['skip_3'].astype(int)))
        df = df[cols]

        groups = df.groupby(['session_id'])
        df = df.loc[(groups['skip_level'].transform('max') > 1) & (groups['session_id'].transform('size') > 5) #& (groups['skip_level'].transform(lambda x: x[-3]) <2)
                    ,:].sort_values(by = ['session_id','session_position'], axis=0) #Do not consider sessions with no skips & (groups['skip_level'].transform(lambda x: max(*x[-3:])) <2)
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
        
        self.train_data, self.val_data, self.test_data, self.vocab = self.split_data(sessions, skips, vocab)
    
        
    
    def split_data(self, sessions, skips, vocab):
        test_targets = list()
        val_targets = list()
        last_train_targets = list()
        for x in sessions:
            test_targets.append(x[-1])
            val_targets.append(x[-2])
            last_train_targets.append(x[-3])
        train_targets = [self.zeropad(x[1:-2], self.max_seq_len) for x in sessions]
        sessions = [self.append_special_tokens(x[:-3]) for x in sessions]
        sessions = torch.tensor(sessions).int() #N x 20
        skips = torch.tensor(skips, dtype=torch.int8) #[0,0,0,0,1,0,1,0,0]
        vocab = torch.tensor(sorted(vocab.values()) + [x for x in range(NUM_RESERVED_TOKENS)]).int()
        train_targets = torch.tensor(train_targets).int()
        test_targets = torch.tensor(test_targets).int()
        val_targets = torch.tensor(val_targets).int()
        last_train_targets = torch.tensor(last_train_targets).int()

        train = TensorDataset(sessions, train_targets, skips)

        val = TensorDataset(sessions, last_train_targets, val_targets, skips)

        test = TensorDataset(sessions, last_train_targets, val_targets, test_targets, skips)

            
        return train, val, test, vocab
    
    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, num_workers = os.cpu_count(), shuffle=True, pin_memory = False)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size, num_workers = os.cpu_count(), pin_memory = False)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.batch_size, num_workers= os.cpu_count(), pin_memory = False)

class DataSampler:
    def __init__(self, filepath, n=1e7,seed=1, path='Sampled_{}', 
        name='data-{}.csv', chunk_size=1e6):
        self.save_path = os.path.join(filepath, path.format(str(n)))
        self.name = name
        self.seed = seed
        self.n = n
        self.cols = ['session_position', 'track_id_clean', 'skip_level', 'session_id']
        self.data = self.load_n_samples(filepath)

        self.save_data(self.data, chunk_size=chunk_size)
    
    def load_n_samples(self, filepath):
        pbar = tqdm(sorted(os.listdir(filepath)))
        print(f'working on path :{filepath}')
        df_all = pd.DataFrame()
        dfs = list()
        for x in pbar:
            df = pd.read_csv(os.path.join(filepath, x))
            df['skip_level'] = ((df['skip_1'].astype(int) + df['skip_2'].astype(int) + df['skip_3'].astype(int)))
            #df_all = pd.concat((df_all, df[self.cols]))
            dfs.append(df[self.cols])
            pbar.set_description(f'Size: {len(df_all)}')
        
        while len(dfs) > 0:
            df_all = pd.concat((df_all, dfs[-1]))
            del dfs[-1]
        
        return df_all.groupby('session_id').sample(n=self.n, random_state=self.seed)
    
    def save_data(self, data, num_chunks = 10):
        chunk_size = int(len(data) / num_chunks)
        path = os.path.join(self.save_path, self.name)
        print(f'saving to path: {self.save_path}')
        for start in tqdm(range(0, df.shape[0], chunk_size)):
            df.iloc[start:start + chunk_size].to_csv(path.format(str(start // chunk_size)))
