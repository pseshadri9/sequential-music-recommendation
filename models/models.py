import pytorch_lightning as pl
import torch
from tqdm import tqdm

class VanillaTransformer(pl.LightningModule): 
    def __init__(self, max_seq_len = None, vocab_size = None, h_dim = None, 
                lr = 0.005, nhead = 4, token_dim = None, dropout = 0.2, nEncoders = 1,
                k = [1, 5, 10, 50, 100]):
        super(VanillaTransformer, self).__init__()
          
        # Define model architecture

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.k = k
        self.pe = torch.nn.Embedding(max_seq_len, token_dim, padding_idx=0)
        self.vocab = torch.nn.Embedding(vocab_size, token_dim)

        encoder_layers = torch.nn.TransformerEncoderLayer(token_dim, nhead, h_dim, dropout, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layers, nEncoders)

        #self.decoder = torch.nn.Sequential(torch.nn.Linear(h_dim, vocab_size), torch.nn.LogSoftmax(dim=-1))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(h_dim, token_dim), torch.nn.GELU())
        self.decoder_bias = torch.nn.parameter.Parameter(torch.randn(vocab_size)).to(self.device)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.mask = self._generate_square_subsequent_mask(self.max_seq_len)
        
        # Defining learning rate
        self.lr = lr
          
        # Define loss 
        self.loss = torch.nn.NLLLoss()

        #Cache Validation outputs for Top - K
        self.val_outs = list()
    
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

    def forward(self, x):
        if not self.mask is None:
            self.mask = self._generate_square_subsequent_mask(self.max_seq_len)

        embs = torch.cat((self.vocab(x[:,:-1]), torch.ones((x.shape[0], 1, self.token_dim)).to(self.device)), axis=1) #N, 20, token_dim, append CLS token embedding

        embs = self.pe(torch.linspace(0,self.max_seq_len - 1, 1).long().to(self.device)) + embs

        enc_out = self.encoder(embs, self.mask)

        return self.decode(enc_out[:, -1]), enc_out
    
    def decode(self, x):
        output = torch.mm(self.decoder(x), self.vocab.weight.transpose(0, 1)) + self.decoder_bias
        return self.softmax(output)

    def configure_optimizers(self):
        # Define and return the optimizer 
        return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def training_step(self, train_batch, batch_idx): 
        
        # Defining training step for our model
        sessions, skip , targets = train_batch 
        output, _ = self.forward(sessions)
        #print(output.shape)
        #output = output.view(-1, self.vocab_size)
        loss = self.loss(output, targets)

        self.log("train_loss", loss.detach(), prog_bar=True)
        #top_k, total = self.test_top_k([(output.detach().cpu(), targets.detach().cpu())], k = [5])
        #self.log_dict({f'top-{k_i} HR': v / total for k_i, v in top_k.items()})
        
        return loss

    
    def validation_step(self, valid_batch, batch_idx): 
        
        # Defining validation steps for our model
        sessions, skip , targets = valid_batch 
        output, _ = self.forward(sessions)
        #print(output.shape)
        #output = output.view(-1, self.vocab_size)
        loss = self.loss(output, targets)

        self.log("val_loss", loss.detach(), prog_bar=True)
        self.val_outs.append(self.test_top_k([(output.detach().cpu(), targets.detach().cpu())]))
        
        return loss
    
    def on_validation_epoch_end(self):
        #top_k = self.test_top_k(self.val_outs)
        top_k = {k_i: 0 for k_i in self.k}
        total = 0
        for d, t in self.val_outs:
            for k, v in d.items():
                top_k[k] += v
            total += t
        
        top_k = {f'top-{k_i}': v / total for k_i, v in top_k.items()}


        self.log_dict(top_k, prog_bar=True)

        self.val_outs.clear()  # free memory
    
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

                z = torch.sum(torch.eq(topK, y).long())

                top_k_rate[k_i] += z
            
            total += X.shape[0]
        
        #return {f'top-{k_i} HR': v / total for k_i, v in top_k_rate.items()}
        return top_k_rate, total
