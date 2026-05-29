import pytorch_lightning as pl
import torch
from torchmetrics import AUROC
from torchmetrics.retrieval import RetrievalMAP
from tqdm import tqdm

from utility import InfoNCE
from data_process import MASKING_ITEM, PADDING_ITEM

class VanillaTransformer(pl.LightningModule):
    def __init__(self, max_seq_len=None, vocab_size=None, h_dim=None,
                 lr=0.005, nhead=4, token_dim=None, dropout=0.2, nEncoders=1,
                 k=[1, 5, 10, 20], return_skip=False, bidirectional=False):
        super(VanillaTransformer, self).__init__()
        self.save_hyperparameters()



        self.max_seq_len = max_seq_len + 1 if bidirectional else max_seq_len
        self.vocab_size  = vocab_size
        self.token_dim   = token_dim
        self.h_dim       = h_dim
        self.k           = k
        self.pe    = torch.nn.Embedding(self.max_seq_len, token_dim)
        self.vocab = torch.nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.return_skip   = return_skip
        self.bidirectional = bidirectional

        encoder_layers = torch.nn.TransformerEncoderLayer(
            token_dim, nhead, h_dim, dropout, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layers, nEncoders)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(h_dim, token_dim), torch.nn.GELU()
        )
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.mask = None

        self.lr   = lr
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

        if self.return_skip:
            self.fc_skip  = None
            self.skip_loss = InfoNCE(negative_mode='paired',
                                     ignore_index=PADDING_ITEM)

        if self.bidirectional:
            self.mask_token = MASKING_ITEM
            self.loss       = torch.nn.NLLLoss(ignore_index=PADDING_ITEM)
            self.mask_amt   = int((self.max_seq_len - 3) // (1 / 0.15))

        # Cache validation outputs for Top-K
        self.val_outs  = []
        self.skip_outs = []

        # Metrics
        self.auroc_target = AUROC(
            task="multiclass", num_classes=vocab_size, ignore_index=0,
            thresholds=5, average='weighted', validate_args=False
        )
        self.auroc_skip = AUROC(
            task="multilabel", num_labels=max_seq_len, thresholds=5,
            ignore_index=2, average='weighted', validate_args=False
        )
        # MAP is computed over the FULL vocabulary (not the sampled subset),
        # so inflated values from the small sampled-softmax universe are avoided.
        self.MAP      = RetrievalMAP(top_k=10)
        self.MAP_outs = []

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (mask.float()
                    .masked_fill(mask == 0, float('-inf'))
                    .masked_fill(mask == 1, float(0.0))
                    .to(self.device))
        return mask.bool()

    def _mask_tracks(self, X, TRAIN=True):
        X = X.detach().clone()
        if TRAIN:
            dist = torch.ones(X.shape).to(self.device)
            dist[X == PADDING_ITEM] = 0
            indices   = torch.multinomial(dist, self.mask_amt).to(self.device)
            batch_idx = (torch.tile(
                torch.linspace(0, X.shape[0] - 1, steps=X.shape[0]),
                (indices.shape[1], 1)
            ).long().T.to(self.device))
            X[batch_idx, indices] = self.mask_token

        last_item = torch.argmin(X, dim=1).to(self.device)
        X[range(X.shape[0]), last_item] = self.mask_token
        X = torch.cat(
            (X, torch.zeros(X.shape[0], self.max_seq_len - X.shape[1])
             .long().to(self.device)),
            axis=1
        )
        return X

    # ------------------------------------------------------------------
    # Sampled softmax (training / loss only)
    # ------------------------------------------------------------------

    def sampled_softmax(self, X, y, num=1000):
        """
        Returns log-probs over (observed targets ∪ random negatives) and
        re-mapped target indices.  Used *only* for loss computation during
        training and validation; MAP is computed on the full-vocab logits.
        """
        observed_tracks, indices = torch.unique(y, return_inverse=True)
        observed_tracks = observed_tracks.to(self.device)

        tracks = torch.linspace(0, self.vocab_size - 1,
                                self.vocab_size).int().to(self.device)
        tracks[observed_tracks] = 0

        idx              = torch.randperm(
            self.vocab_size - observed_tracks.shape[0] - 1
        ).to(self.device)
        negative_samples = tracks[tracks != 0][idx[:num]]

        sampled_logits = torch.cat(
            (X[:, :, observed_tracks], X[:, :, negative_samples]), axis=2
        )
        return self.softmax(sampled_logits), indices

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        if not self.bidirectional:
            self.mask = self._generate_square_subsequent_mask(self.max_seq_len)

        embs = self.vocab(x)
        embs = (self.pe(torch.linspace(0, self.max_seq_len - 1,
                                       self.max_seq_len).int().to(self.device))
                + embs)
        enc_out = self.encoder(embs, self.mask)
        return self.decode(enc_out)

    def decode(self, x):
        x_     = self.decoder(x)
        output = torch.matmul(x_, self.vocab.weight.transpose(0, 1))
        return output, x_

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, train_batch, batch_idx):
        sessions, targets, skip = train_batch
        skip = skip.long()

        if self.bidirectional:
            masked_sessions = self._mask_tracks(sessions)
            masked_targets  = sessions[
                masked_sessions[:, :-1] == self.mask_token
            ].view(sessions.shape[0], self.mask_amt + 1)
            last_item = torch.argmin(sessions, dim=1) - 1
            masked_targets[:, -1] = targets[range(targets.shape[0]), last_item]
            output, x_ = self.forward(masked_sessions)
            output  = output[masked_sessions == self.mask_token].view(
                output.shape[0], self.mask_amt + 1, -1
            )
            targets = masked_targets
        else:
            output, x_ = self.forward(sessions)

        targets = targets.long()
        output_sampled, targets_sampled = self.sampled_softmax(output, targets)
        output_sampled  = output_sampled.transpose(1, 2)
        target_loss     = self.loss(output_sampled, targets_sampled)

        if self.return_skip:
            if self.bidirectional:
                x_ = x_[:, :-1, :]
            neg_targets, pos_query, pos_key = self.get_negative_samples(
                x_, skip, sessions=sessions
            )
            skip_loss  = self.skip_loss(pos_query, pos_key,
                                        negative_keys=neg_targets)
            train_loss = skip_loss + target_loss
        else:
            train_loss = target_loss

        self.log("train_loss", train_loss.detach(), prog_bar=True, sync_dist=True)
        if self.return_skip:
            try:
                self.log("target_loss", target_loss.detach(),
                         prog_bar=True, sync_dist=True)
            except Exception:
                pass
            self.log("skip_loss", skip_loss.detach(),
                     prog_bar=True, sync_dist=True)

        return train_loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, valid_batch, batch_idx):
        sessions, targets = self.get_val_batch(valid_batch[:-1])
        skips = valid_batch[-1].long()

        if self.bidirectional:
            sessions = self._mask_tracks(sessions, TRAIN=False)

        output, x_ = self.forward(sessions)
        targets     = targets.long()

        # --- loss (sampled softmax universe) ---
        output_sampled, targets_sampled = self.sampled_softmax(output, targets)
        idx    = torch.argmin(sessions, dim=1) - 1
        out_s  = output_sampled[range(output_sampled.shape[0]), idx, :]
        target_loss = self.loss(out_s, targets_sampled)

        if self.return_skip:
            if self.bidirectional:
                x_ = x_[:, :-1, :]
            neg_targets, pos_query, pos_key = self.get_negative_samples(
                x_, skips, sessions=sessions
            )
            skip_loss = self.skip_loss(pos_query, pos_key,
                                       negative_keys=neg_targets)
            loss = skip_loss + target_loss
        else:
            loss = target_loss

        # --- Top-K HR (over sampled universe, same as before) ---
        self.val_outs.append(
            self.test_top_k([(out_s.detach().cpu(),
                              targets_sampled.detach().cpu())])
        )

        # --- MAP over sampled-negative universe ---
        full_logits = output[range(output.shape[0]), idx, :]
        self._update_map(full_logits, targets)

        self.log("val_loss", loss.detach(), prog_bar=True, sync_dist=True)
        if self.return_skip:
            self.log("val target_loss", target_loss.detach(),
                     prog_bar=True, sync_dist=True)
            self.log("val skip_loss", skip_loss.detach(),
                     prog_bar=True, sync_dist=True)

        return loss

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        sessions, targets = self.get_test_batch(batch[:-1])
        skips = batch[-1]

        if self.bidirectional:
            sessions = self._mask_tracks(sessions, TRAIN=False)

        output, x_ = self.forward(sessions)
        targets     = targets.long()

        output_sampled, targets_sampled = self.sampled_softmax(output, targets)
        idx   = torch.argmin(sessions, dim=1) - 1
        out_s = output_sampled[range(output_sampled.shape[0]), idx, :]
        target_loss = self.loss(out_s, targets_sampled)

        if self.return_skip:
            if self.bidirectional:
                x_ = x_[:, :-1, :]
            neg_targets, pos_query, pos_key = self.get_negative_samples(
                x_, skips, sessions=sessions
            )
            skip_loss = self.skip_loss(pos_query, pos_key,
                                       negative_keys=neg_targets)
            loss = skip_loss + target_loss
        else:
            loss = target_loss

        self.val_outs.append(
            self.test_top_k([(out_s.detach().cpu(),
                              targets_sampled.detach().cpu())])
        )

        # MAP over sampled-negative universe
        full_logits = output[range(output.shape[0]), idx, :]
        self._update_map(full_logits, targets)

        self.log("test_loss", loss.detach(), prog_bar=True, sync_dist=True)
        if self.return_skip:
            self.log("test target_loss", target_loss.detach(),
                     prog_bar=True, sync_dist=True)
            self.log("test skip_loss", skip_loss.detach(),
                     prog_bar=True, sync_dist=True)

        return loss

    # ------------------------------------------------------------------
    # MAP helper – sampled-negative evaluation (memory-safe)
    # ------------------------------------------------------------------

    def set_eval_negatives(self, num_neg=999):
        """
        Call once after the vocab is known (e.g. in main.py before trainer.fit)
        to fix the negative-sample pool used during val/test MAP evaluation.
        Draws `num_neg` items uniformly from [1, vocab_size) excluding padding (0),
        and registers them as a buffer so they move with the model to the right device.

            model.set_eval_negatives(num_neg=999)

        The true item is appended per-sample at eval time, giving a ranked list
        of (num_neg + 1) candidates — consistent with the standard SASRec / BERT4Rec
        evaluation protocol and memory-safe even for large vocabularies.
        """
        neg_pool = torch.randperm(self.vocab_size - 1)[:num_neg] + 1  # skip idx 0 (PAD)
        self.register_buffer('eval_neg_pool', neg_pool)

    def _update_map(self, full_logits, targets):
        """
        Rank the ground-truth item against `eval_neg_pool` negatives.

        full_logits : (N, vocab_size) – raw (non-normalised) scores for every item.
        targets     : (N,)            – ground-truth item indices in vocab space.

        RetrievalMAP expects flat (N*(num_neg+1),) tensors.
        The true item is always placed first in the candidate list (index 0 per
        query) so the boolean target is simply True at position 0 and False for
        the rest.
        """
        if not hasattr(self, 'eval_neg_pool'):
            # Fallback: sample on-the-fly if set_eval_negatives was not called.
            neg_pool = (torch.randperm(self.vocab_size - 1)[:999] + 1).to(full_logits.device)
        else:
            neg_pool = self.eval_neg_pool.to(full_logits.device)

        N        = full_logits.shape[0]
        num_neg  = neg_pool.shape[0]
        cand_len = num_neg + 1          # true item + negatives

        # Scores for the true item: (N, 1)
        true_scores = full_logits[torch.arange(N), targets].unsqueeze(1)

        # Scores for the shared negative pool: (N, num_neg)
        neg_scores  = full_logits[:, neg_pool]

        # Candidate scores: true item first, then negatives → (N, cand_len)
        cand_scores = torch.cat([true_scores, neg_scores], dim=1)

        # Relevance: only position 0 is relevant per query
        map_target  = torch.zeros(N, cand_len, dtype=torch.bool)
        map_target[:, 0] = True

        # Group index: one integer per query, repeated cand_len times
        indexes = torch.arange(N).unsqueeze(1).expand(N, cand_len).reshape(-1)

        self.MAP.update(
            cand_scores.reshape(-1).detach().cpu(),
            map_target.reshape(-1).detach().cpu(),
            indexes=indexes.detach().cpu()
        )

    # ------------------------------------------------------------------
    # Epoch-end callbacks
    # ------------------------------------------------------------------

    def _log_epoch_metrics(self, split):
        top_k = {k_i: 0 for k_i in self.k}
        total = 0
        for d, t in self.val_outs:
            for k, v in d.items():
                top_k[k] += v
            total += t

        for k_i, v in top_k.items():
            self.log(f'{split} top-{k_i}', v / total,
                     prog_bar=True, sync_dist=True)

        map_val = self.MAP.compute().item()
        self.log(f'{split} MAP', map_val, on_epoch=True)
        self.MAP.reset()

        self.val_outs.clear()
        self.skip_outs.clear()

    def on_validation_epoch_end(self):
        self._log_epoch_metrics('Val')

    def on_test_epoch_end(self):
        self._log_epoch_metrics('Test')

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def get_val_batch(self, valid_batch):
        sessions, last_target, target = valid_batch
        idx = torch.argmin(sessions, dim=1)
        sessions[range(sessions.shape[0]), idx] = last_target
        return sessions, target

    def get_test_batch(self, test_batch):
        last_target = test_batch[-1]
        return self.get_val_batch((*self.get_val_batch(test_batch[:-1]), last_target))

    # ------------------------------------------------------------------
    # Contrastive / skip helpers
    # ------------------------------------------------------------------

    def get_negative_samples(self, a, skip, sessions=None):
        keys    = self.vocab(sessions) if sessions is not None else a
        neg_mask = skip == 1
        pos_mask = skip == 0

        n     = torch.zeros_like(a)
        p     = torch.zeros_like(a)
        p_key = torch.zeros_like(a)

        n[neg_mask] = keys[neg_mask]

        closest = self._closest_pos_sample(a, skip)
        p_key[:, :-1][pos_mask[:, :-1]] = \
            keys[torch.arange(a.shape[0]).unsqueeze(-1), closest][:, :-1][pos_mask[:, :-1]]
        p[:, :-1][pos_mask[:, :-1]] = a[:, :-1][pos_mask[:, :-1]]

        n_size = self.max_seq_len - 1 if self.bidirectional else self.max_seq_len
        return n.tile(n_size, 1, 1), p.view(-1, *p.shape[2:]), p_key.view(-1, *p_key.shape[2:])

    def _closest_pos_sample(self, a, skip):
        m = torch.iinfo(skip.dtype).max
        z = torch.linspace(0, a.shape[1] - 1, a.shape[1]).unsqueeze(-1)
        z = z.tile((a.shape[0],)).transpose(0, 1)
        b = z.detach().clone().unsqueeze(-1)
        z[skip != 0] = m
        z    = z.unsqueeze(-1).tile((1, 1, a.shape[1])).transpose(1, 2)
        diff = z - b
        diff[diff <= 0] = m
        return diff.argmin(dim=2)

    # ------------------------------------------------------------------
    # Top-K Hit Rate
    # ------------------------------------------------------------------

    def test_top_k(self, batches, k=None):
        if k is None:
            k = self.k
        top_k_rate = {k_i: 0 for k_i in k}
        total = 0
        for X, y in batches:
            X = X.to(self.device)
            y = torch.unsqueeze(y, dim=-1).to(self.device)
            for k_i in k:
                _, topK = torch.topk(X, k_i, dim=1)
                top_k_rate[k_i] += torch.sum(torch.eq(topK, y).int())
            total += X.shape[0]
        return top_k_rate, total

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)[0]
