import pytorch_lightning as pl
import torch

class VanillaTransformer(pl.LightningModule): 
    def __init__(self, max_seq_len = None, vocab_size = None, h_dim = None, 
                lr = 0.005, nhead = 4, token_dim = None, dropout = 0.5, nEncoders = 1): 
        super(model, self).__init__()
          
        # Define model architecture

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.pe = torch.nn.Embedding(max_seq_len, token_dim)
        self.vocab = torch.nn.Embedding(vocab_size, token_dim)

        encoder_layers = TransformerEncoderLayer(token_dim, nhead, h_dim, dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layers, nEncoders)

        self.decoder = nn.Sequential(nn.Linear(h_dim, vocab_size), nn.LogSoftmax(dim=-1))

        self.mask = _generate_square_subsequent_mask(self, max_seq_len)
        
        # Defining learning rate
        self.lr = lr
          
        # Define loss 
        self.loss = torch.nn.NLLLoss()
    
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        embs = self.pe(torch.linspace(0,self.max_seq_len - 1, 1)) + self.vocab(x)

        return self.fc(self.encoder(embs, self.mask))

        
    
    def configure_optimizers(self):
        # Define and return the optimizer 
        return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def training_step(self, train_batch, batch_idx): 
        
        # Defining training step for our model
        sessions, _ , targets = train_batch 
        output = self.forward(sessions)
        output = output.view(-1, self.vocab_size)
        return self.loss(output, targets)

    
    def validation_step(self, valid_batch, batch_idx): 
        
        # Defining validation steps for our model
        sessions, _ , targets = train_batch 
        output = self.forward(sessions)
        output = output.view(-1, self.vocab_size)
        return self.loss(output, targets)