#infonce, AUROC 
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['InfoNCE', 'info_nce']

class SkipContrastiveLoss(nn.Module):
    def __init__(self, ignore_index = 0):
        super().__init__()
        self.CE = InfoNCE(negative_mode='paired', ignore_index=ignore_index)
        self.ignore_index = ignore_index
    
    def forward(self, x_, skips, sessions):
        neg_targets, pos_query, pos_key = self.get_negative_samples(x_, skips, sessions=sessions)
        return self.CE(pos_query, pos_key, negative_keys=neg_targets)
    
    def get_negative_samples(self, a, skip, sessions=None):
        '''
        retrieves negative sample embeddings for each track in each session. 
        
        params:
            a: batch of listening sessions with track embedding, shape: [N, h_dim, session_length]
            skip: skip labels for each session, shape: [N, session_length]
            sessions: 
        
        '''
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
        
        #ignore last sample which does not have a closest positive sample
        p_key[:, :-1][pos_mask[:, :-1]] = keys[torch.arange(a.shape[0]).unsqueeze(-1),
                                    self._closest_pos_sample(a, skip)][:, :-1][pos_mask[:, :-1]]
        p[:, :-1][pos_mask[:, :-1]] = a[:, :-1][pos_mask[:, :-1]]

        n_size = self.max_seq_len - 1 if self.bidirectional else self.max_seq_len
                                                                        
        return n.tile(n_size, 1, 1), p.view(-1, *p.shape[2:]), p_key.view(-1, *p_key.shape[2:])
    
    def _closest_pos_sample(self, a, skip):
        '''
        vectorized implementation to compute the closest positive sample for each track in each batch

        params:
            a: batch of sessions with track embeddings [N, embedding_dim, session_length]
        '''
        m = torch.iinfo(skip.dtype).max

        z = torch.linspace(0, a.shape[1] - 1, a.shape[1]).unsqueeze(dim=-1)
        z = z.tile((a.shape[0],)).transpose(0, 1)

        b = z.detach().clone().unsqueeze(dim=-1)

        z[skip != 0] = m

        z = z.unsqueeze(dim=-1).tile((1, 1, a.shape[1])).transpose(1, 2)

        diff = (z - b)

        diff[diff <= 0] = m

        return diff.argmin(dim=2)

class InfoNCE(nn.Module):
    """
    The following implementation is adapted from: https://github.com/RElbers/info-nce-pytorch

    
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', ignore_index = 0):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.ignore_index = ignore_index

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode,
                        ignore_index=self.ignore_index)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired',
             ignore_index=0):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
