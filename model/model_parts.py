import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


# Todo later, relative positional encoding

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, n_heads):
        super(MultiHeadAttention, self).__init__()

        assert model_dimension % n_heads == 0

        self.model_dimension = model_dimension
        self.n_heads = n_heads
        self.head_dimension = model_dimension // n_heads

        # From https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
        # However, the important thing to understand is that this is a logical split only. 
        # The Query, Key, and Value are not physically split into separate matrices, 
        # one for each Attention head. A single data matrix is used for the Query, Key, and Value, respectively, 
        # with logically separate sections of the matrix for each Attention head. 
        # Similarly, there are not separate Linear layers, one for each Attention head. 
        # All the Attention heads share the same Linear layer 
        # but simply operate on their ‘own’ logical section of the data matrix.

        self.w_q = nn.Linear(model_dimension, model_dimension)
        self.w_k = nn.Linear(model_dimension, model_dimension)
        self.w_v = nn.Linear(model_dimension, model_dimension)

        self.w_o = nn.Linear(model_dimension, model_dimension)


    def split_heads(self, x):
        batch_size, seq_len, model_dimension = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.head_dimension).transpose(1, 2)
    
    def merge_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_len, self.model_dimension)
    
    def forward(self, x):
        Q = self.split_heads(self.w_q(x))
        K = self.split_heads(self.w_k(x))
        V = self.split_heads(self.w_v(x))

        attn_scores = torch.matmul(Q, K.transpose(2,3)) 
        attn_scores = attn_scores / torch.sqrt(self.head_dimension)
        attn_probs = torch.softmax(attn_scores)

        output = torch.matmul(attn_probs, V)
        output = self.w_o(self.combine_heads(output))

        return output


