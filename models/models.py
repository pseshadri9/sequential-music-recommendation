import pytorch_lightning as pl
import torch
from torchmetrics import AUROC
from torchmetrics.retrieval import RetrievalMAP
from tqdm import tqdm

from utility import InfoNCE
from data_process import MASKING_ITEM, PADDING_ITEM

class VanillaTransformer(pl.LightningModule): 
    def __init__(self, max_seq_len = None, vocab_size = None, h_dim = None, 
                lr = 0.005, nhead = 4, token_dim = None, dropout = 0.2, nEncoders = 1,
                k = [1, 5, 10, 50, 100], return_skip = False, bidirectional = False):
        super(VanillaTransformer, self).__init__()
        self.save_hyperparameters()
          
        # Define model architecture

        self.max_seq_len = max_seq_len + 1 if bidirectional else max_seq_len
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.h_dim = h_dim
        self.k = k
        self.pe = torch.nn.Embedding(self.max_seq_len, token_dim)
        self.vocab = torch.nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.return_skip = return_skip
        self.bidirectional = bidirectional

        encoder_layers = torch.nn.TransformerEncoderLayer(token_dim, nhead, h_dim, dropout, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layers, nEncoders)

        #self.decoder = torch.nn.Sequential(torch.nn.Linear(h_dim, vocab_size), torch.nn.LogSoftmax(dim=-1))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(h_dim, token_dim), torch.nn.GELU())
        #self.decoder_bias = torch.nn.parameter.Parameter(torch.randn(vocab_size)).to(self.device)
        self.softmax = torch.nn.LogSoftmax(dim=-1)


        if return_skip:
            self.fc_skip = None #torch.nn.Linear(token_dim, self.max_seq_len)

        self.mask = None #self._generate_square_subsequent_mask(self.max_seq_len)
        
        # Defining learning rate
        self.lr = lr
          
        # Define loss 
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        #self.loss = torch.nn.NLLLoss(ignore_index=0)
        if self.return_skip:
            self.skip_loss = InfoNCE(negative_mode='paired', ignore_index=PADDING_ITEM) #torch.nn.CrossEntropyLoss(ignore_index=2)
            #self.sigma_target = torch.nn.parameter.Parameter(torch.rand(1)).to(self.device)
            #self.sigma_skip = torch.nn.parameter.Parameter(torch.rand(1)).to(self.device)
        
        if self.bidirectional:
            self.mask_token = MASKING_ITEM
            self.loss = torch.nn.NLLLoss(ignore_index = PADDING_ITEM)
            self.mask_amt = int((self.max_seq_len - 3) // (1/0.15)) #max_seq_len - 3 items in training sequence, 0.15 mask proportion


        #Cache Validation outputs for Top - K
        self.val_outs = list()
        self.skip_outs = list()

        #Define metrics
        self.auroc_target = AUROC(task="multiclass", num_classes=vocab_size, ignore_index=0,thresholds= 5, average='weighted', validate_args=False)
        self.auroc_skip = AUROC(task="multilabel", num_labels=max_seq_len, thresholds= 5, ignore_index=2, average='weighted', validate_args=False)
        self.MAP = RetrievalMAP(top_k=10)
        self.MAP_outs = list()

    def init_weights(self):
        initrange = 0.02
        nn.init.trunc_normal_(self.encoder.weight, a=-initrange, b=initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.zeros_(self.decoder_bias)
        nn.init.trunc_normal_(self.decoder.weight, a=-initrange, b=initrange)
        nn.init.trunc_normal_(self.vocab.weight, a=-initrange, b=initrange)
        nn.init.trunc_normal_(self.pe.weight, a=-initrange, b=initrange)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(self.device)
        return mask.bool()
    
    def _mask_tracks(self, X, TRAIN=True):
        X = X.detach().clone()
        if TRAIN:
            dist = torch.ones(X.shape).to(self.device)
            dist[X == PADDING_ITEM] = 0
            #indices = torch.clamp(torch.multinomial(dist, int(X.shape[1] // (1/self.mask_prop))), min=torch.ones((X.shape[0], 1)).to(self.device), 
                                #max=(last_item - 1).unsqueeze(dim=-1)).long().to(self.device)      
            #indices[:, 0] -= 1

            indices = torch.multinomial(dist, self.mask_amt).to(self.device)
            batch_idx = torch.tile(torch.linspace(0, X.shape[0] - 1, steps = X.shape[0]), (indices.shape[1],1)).long().T.to(self.device)

            X[batch_idx, indices] = self.mask_token

        last_item = torch.argmin(X, dim=1).to(self.device)
        X[range(X.shape[0]), last_item] = self.mask_token
        X = torch.cat((X, torch.zeros(X.shape[0], self.max_seq_len - X.shape[1]).long().to(self.device)), axis=1)

        return X
    
    def sampled_softmax(self, X, y, num=1000):
        observed_tracks, indices = torch.unique(y, return_inverse=True)

        observed_tracks = observed_tracks.to(self.device)
        
        tracks = torch.linspace(0,self.vocab_size - 1, self.vocab_size).int().to(self.device)

        tracks[observed_tracks] = 0

        idx = torch.randperm(self.vocab_size - observed_tracks.shape[0] - 1).to(self.device)

        negative_samples = tracks[tracks != 0][idx[:num]]


        return self.softmax(torch.cat((X[:, :, observed_tracks], X[:, :, negative_samples]), axis=2)), indices
    
    def _mask_unseen_tracks(self, y, num=1000):
        pass


    def forward(self, x):
        if not self.bidirectional:
            self.mask = self._generate_square_subsequent_mask(self.max_seq_len)

        #embs = torch.cat((self.vocab(x[:,:-1]), torch.ones((x.shape[0], 1, self.token_dim)).to(self.device)), axis=1) #N, 20, token_dim, append CLS token embedding
        embs = self.vocab(x)

        embs = self.pe(torch.linspace(0,self.max_seq_len - 1, self.max_seq_len).int().to(self.device)) + embs

        enc_out = self.encoder(embs, self.mask)

        return self.decode(enc_out) #(self.decode(enc_out), None) if not self.return_skip else (self.decode(enc_out), enc_out) 
    
    def decode(self, x):
        x_ = self.decoder(x)
        output = torch.matmul(x_, self.vocab.weight.transpose(0, 1)) #+ self.decoder_bias
        return output, x_

    def configure_optimizers(self):
        # Define and return the optimizer 
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr = self.lr)
    
    def training_step(self, train_batch, batch_idx):

        # Defining training step for our model
        sessions, targets, skip = train_batch
        skip = skip.long()

        if self.bidirectional:
            masked_sessions = self._mask_tracks(sessions)
            masked_targets = sessions[masked_sessions[:, :-1] == self.mask_token].view(sessions.shape[0], self.mask_amt + 1)
            last_item = torch.argmin(sessions, dim=1) - 1
            masked_targets[:, -1] = targets[range(targets.shape[0]), last_item]
            output, x_ = self.forward(masked_sessions)

            output = output[masked_sessions == self.mask_token].view(output.shape[0], self.mask_amt + 1, -1)
            targets = masked_targets

        else:
            output, x_ = self.forward(sessions)

        targets = targets.long()

        output, targets = self.sampled_softmax(output, targets)
        #output = self.softmax(output)


        output = output.transpose(1, 2)
        
        #closest_samples = self._closest_pos_sample(x_, skip).long()
        #targets[:, :-1] = closest_samples[:, :-1]
        target_loss = self.loss(output, targets)

        if self.return_skip:
            if self.bidirectional:
                #sessions = sessions[:, :-1]
                x_ = x_[:, :-1, :]
                #raise Exception(x_.shape, skip.shape, sessions.shape)
            neg_targets, pos_query, pos_key = self.get_negative_samples(x_, skip, sessions=sessions)
            
            skip_loss = self.skip_loss(pos_query, pos_key, negative_keys=neg_targets)

            train_loss = skip_loss + target_loss
            #train_loss = (1/(2 * self.sigma_target ** 2)) * target_loss \
                #+ (1/(self.sigma_skip ** 2)) * skip_loss + torch.log(self.sigma_skip) + torch.log(self.sigma_target)
        else:
            train_loss = target_loss #+ skip_loss #+ 0.1 * skip_loss

        self.log("train_loss", train_loss.detach(), prog_bar=True, sync_dist=True)

        if self.return_skip:
            try:
                self.log("target_loss", target_loss.detach(), prog_bar=True, sync_dist=True)
            except:
                pass
            self.log("skip_loss", skip_loss.detach(), prog_bar=True, sync_dist=True)

            #self.log("target_weight", (1/(2 * self.sigma_target ** 2)).detach(), prog_bar=True, sync_dist=True)
            #self.log("skip_weight", (1/(self.sigma_skip ** 2)).detach(), prog_bar=True, sync_dist=True)
        #top_k, total = self.test_top_k([(output.detach().cpu(), targets.detach().cpu())], k = [5])
        #self.log_dict({f'top-{k_i} HR': v / total for k_i, v in top_k.items()})
        
        return train_loss

    
    def validation_step(self, valid_batch, batch_idx):
        # Defining validation steps for our model
        sessions, targets = self.get_val_batch(valid_batch[:-1])
        skips = valid_batch[-1].long()

        if self.bidirectional:
            sessions = self._mask_tracks(sessions, TRAIN=False)

        output, x_ = self.forward(sessions)
        #output = self.softmax(output)
        targets = targets.long()

        #output, targets = self.sampled_softmax(output, targets.contiguous())
        #output = output.view(-1, self.vocab_size)

        output, targets = self.sampled_softmax(output, targets)
        #output = self.softmax(output)
        idx = torch.argmin(sessions, dim=1) - 1       
        output = output[range(output.shape[0]), idx, :]



        target_loss = self.loss(output, targets)
        if self.return_skip:
            if self.bidirectional:
                sessions = sessions[:, :-1]
                x_ = x_[:, :-1, :]
            neg_targets, pos_query, pos_key = self.get_negative_samples(x_, skips, sessions=sessions)
            #raise Exception(neg_targets.shape, pos_query.shape, pos_key.shape)
            
            skip_loss = self.skip_loss(pos_query, pos_key, negative_keys=neg_targets)

            loss = skip_loss + target_loss

            #loss = (1/(2 * self.sigma_target ** 2)) * target_loss \
                #+ (1/(self.sigma_skip ** 2)) * skip_loss + torch.log(self.sigma_skip) + torch.log(self.sigma_target)
        else:
            loss = target_loss

        indexes = torch.arange(output.shape[0]).unsqueeze(dim=1).tile((output.shape[1]))
        MAP_target = torch.zeros_like(indexes).bool()
        MAP_target[range(targets.shape[0]), targets] = True
        self.MAP.update(output.detach().cpu(), MAP_target.detach().cpu(), indexes=indexes.detach().cpu())
        #self.auroc_target(output, targets)
        #self.compute_MAP(output.detach(), targets.detach(), skips.detach())

        index = torch.Tensor(range(output.shape[0])).long().to(self.device).unsqueeze(dim=-1)
        #self.auroc_skip(output[index, torch.cat((sessions[:, :-1], targets.unsqueeze(dim=-1)), axis = 1)], skips.long())

        self.log("val_loss", loss.detach(), prog_bar=True, sync_dist=True)
        self.val_outs.append(self.test_top_k([(output.detach().cpu(), targets.detach().cpu())]))

        if self.return_skip:
            self.log("val target_loss", target_loss.detach(), prog_bar=True, sync_dist=True)
            self.log("val skip_loss", skip_loss.detach(), prog_bar=True, sync_dist=True)
        
        return loss

    def compute_MAP(self, preds, sessions, skips):
        with torch.no_grad():
            observed_tracks, indices = torch.unique(sessions, return_inverse=True)
            sessions[:, 1:][(skips[:, :-1] != 1)] = 0
            tgt_skip = torch.zeros_like(preds).to(self.device)
            tgt_skip[torch.tile(torch.linspace(0, sessions.shape[1] - 1, steps = sessions.shape[1]), (sessions.shape[0],1)).long().to(self.device)
                     , sessions] = 1
            idx = torch.arange(sessions.shape[0]).unsqueeze(dim=-1).tile(1, preds.shape[1]).long().to(self.device)
            self.MAP_outs.append((self.MAP(preds, tgt_skip.bool(), indexes=idx).detach().cpu().item(), preds.shape[0]))
    
    def MAP_val(self):
        res = 0
        tot = 0
        for map, bs in self.MAP_outs:
            res += map * bs
            tot += bs
        
        self.MAP_outs.clear()
        return res / tot




    def test_step(self, batch, batch_idx):
        # Defining validation steps for our model
        sessions, targets = self.get_test_batch(batch[:-1])
        skips = batch[-1]

        if self.bidirectional:
            sessions = self._mask_tracks(sessions, TRAIN=False)

        output, x_ = self.forward(sessions, )
        #output = output.view(-1, self.vocab_size)
        targets = targets.long()
        #output = self.softmax(output)
        output, targets = self.sampled_softmax(output, targets)
        #output = self.softmax(output)

        idx = torch.argmin(sessions, dim=1) - 1
        output = output[range(output.shape[0]), idx, :]

        #z = self.get_skip_one_hot(sessions, targets, skips)
        target_loss = self.loss(output, targets)

        if self.return_skip:
            if self.bidirectional:
                sessions = sessions[:, :-1]
                x_ = x_[:, :-1, :]
            neg_targets, pos_query, pos_key = self.get_negative_samples(x_, skips, sessions=sessions)
            
            skip_loss = self.skip_loss(pos_query, pos_key, negative_keys=neg_targets)

            loss = skip_loss + target_loss
            #loss = (1/(2 * self.sigma_target ** 2)) * target_loss \
                #+ (1/(self.sigma_skip ** 2)) * skip_loss + torch.log(self.sigma_skip) + torch.log(self.sigma_target)
        else:
            
            loss = target_loss

        #self.auroc_target(output, targets)
        #self.compute_MAP(output.detach(), targets.detach(), skips.detach())
        
        index = torch.Tensor(range(output.shape[0])).long().to(self.device).unsqueeze(dim=-1)
        #self.auroc_skip(output[index, torch.cat((sessions[:, :-1], targets.unsqueeze(dim=-1)), axis = 1)], skips.long())
        indexes = torch.arange(output.shape[0]).unsqueeze(dim=1).tile((output.shape[1]))
        MAP_target = torch.zeros_like(indexes).bool()
        MAP_target[range(targets.shape[0]), targets] = True
        self.MAP.update(output.detach().cpu(), MAP_target.detach().cpu(), indexes=indexes.detach().cpu())


        self.log("test_loss", loss.detach(), prog_bar=True, sync_dist=True)
        self.val_outs.append(self.test_top_k([(output.detach().cpu(), targets.detach().cpu())]))

        if self.return_skip:
            self.log("test target_loss", target_loss.detach(), prog_bar=True, sync_dist=True)
            self.log("test skip_loss", skip_loss.detach(), prog_bar=True, sync_dist=True)
        
        return loss

    def get_val_batch(self, valid_batch):
        sessions, last_target, target = valid_batch
        idx = torch.argmin(sessions, dim=1)
        sessions[range(sessions.shape[0]), idx] = last_target

        return (sessions, target)

    def get_test_batch(self, test_batch):
        last_target = test_batch[-1]
        return self.get_val_batch((*self.get_val_batch(test_batch[:-1]), last_target))
    
    def get_negative_samples(self, a, skip, sessions=None):
        if sessions is None:
            keys = a
        else:
            keys = self.vocab(sessions)
        neg_mask = skip == 1
        pos_mask = skip == 0


        n = torch.zeros_like(a)
        n[neg_mask] = keys[neg_mask]
        
        p = torch.zeros_like(a)
        p_key = torch.zeros_like(a)
        
        #ignore last item which does not have a closest positive sample
        p_key[:, :-1][pos_mask[:, :-1]] = keys[torch.arange(a.shape[0]).unsqueeze(-1),
                                    self._closest_pos_sample(a, skip)][:, :-1][pos_mask[:, :-1]]
        p[:, :-1][pos_mask[:, :-1]] = a[:, :-1][pos_mask[:, :-1]]

        n_size = self.max_seq_len - 1 if self.bidirectional else self.max_seq_len
                                                                        
        return n.tile(n_size, 1, 1), p.view(-1, *p.shape[2:]), p_key.view(-1, *p_key.shape[2:])
    
    def _closest_pos_sample(self, a, skip):
        m = torch.iinfo(skip.dtype).max

        z = torch.linspace(0, a.shape[1] - 1, a.shape[1]).unsqueeze(dim=-1)
        z = z.tile((a.shape[0],)).transpose(0, 1)

        b = z.detach().clone().unsqueeze(dim=-1)

        z[skip != 0] = m

        z = z.unsqueeze(dim=-1).tile((1, 1, a.shape[1])).transpose(1, 2)

        diff = (z - b)

        diff[diff <= 0] = m

        return diff.argmin(dim=2)



    def get_skip_one_hot(self, sessions, targets, skip):
        s = torch.cat((sessions[:, :-1], targets.unsqueeze(dim=-1)), axis = 1)
        z = torch.ones((sessions.shape[0], self.vocab_size)).long().to(self.device) * 2
        z[range(s.shape[0]), s.unsqueeze(dim=-1).long()] = skip.unsqueeze(dim=-1).long()
        return z
    
    def on_validation_epoch_end(self):
        #top_k = self.test_top_k(self.val_outs)
        top_k = {k_i: 0 for k_i in self.k}
        total = 0
        for d, t in self.val_outs:
            for k, v in d.items():
                top_k[k] += v
            total += t
        
        top_k = {f'Val top-{k_i}': v / total for k_i, v in top_k.items()}

        for k, v in top_k.items():
            self.log(k,v, prog_bar=True, sync_dist=True)
        
        #self.log('Val AUROC Target', self.auroc_target, on_epoch=True)
        #self.log('Val AUROC Skip', self.auroc_skip, on_epoch=True)
        self.log('Val MAP', self.MAP.compute().item(), on_epoch=True)
        self.MAP.reset()

        self.val_outs.clear()  # free memory
        self.skip_outs.clear()
    
    def on_test_epoch_end(self):
        #top_k = self.test_top_k(self.val_outs)
        top_k = {k_i: 0 for k_i in self.k}
        total = 0
        for d, t in self.val_outs:
            for k, v in d.items():
                top_k[k] += v
            total += t
        
        top_k = {f'Test top-{k_i}': v / total for k_i, v in top_k.items()}

        for k, v in top_k.items():
            self.log(k,v, prog_bar=True, sync_dist=True)
        
        #self.log('Test AUROC Target', self.auroc_target, on_epoch=True)
        #self.log('Test AUROC Skip', self.auroc_skip, on_epoch=True)
        self.log('Test MAP', self.MAP.compute().item(), on_epoch=True)
        self.MAP.reset()

        self.val_outs.clear()  # free memory
        self.skip_outs.clear()
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)[0]
    
    def test_top_k(self, batches, k = None):
        if k is None:
            k = self.k
        top_k_rate = {k_i: 0 for k_i in k}
        total = 0

        #for batch in tqdm(batches, desc='Top-K Val'):
        for batch in batches:
            X, y = batch #(N, vocab_size) , (N, )
            X = X.to(self.device)
            y = torch.unsqueeze(y, dim=-1).to(self.device)

            for k_i in k:
                _, topK = torch.topk(X, k_i, dim=1)

                z = torch.sum(torch.eq(topK, y).int())

                top_k_rate[k_i] += z
            
            total += X.shape[0]
        
        #return {f'top-{k_i} HR': v / total for k_i, v in top_k_rate.items()}
        return top_k_rate, total
