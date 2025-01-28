import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


# Todo later, relative positional encoding

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim, n_heads):
        super(MultiHeadAttention, self).__init__()

        assert model_dim % n_heads == 0

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads

        # From https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
        # However, the important thing to understand is that this is a logical split only. 
        # The Query, Key, and Value are not physically split into separate matrices, 
        # one for each Attention head. A single data matrix is used for the Query, Key, and Value, respectively, 
        # with logically separate sections of the matrix for each Attention head. 
        # Similarly, there are not separate Linear layers, one for each Attention head. 
        # All the Attention heads share the same Linear layer 
        # but simply operate on their ‘own’ logical section of the data matrix.

        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)

        self.w_o = nn.Linear(model_dim, model_dim)


    def split_heads(self, x):
        batch_size, seq_len, model_dim = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    
    def merge_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_len, self.model_dim)
    
    def forward(self, x):
        Q = self.split_heads(self.w_q(x))
        K = self.split_heads(self.w_k(x))
        V = self.split_heads(self.w_v(x))

        attn_scores = torch.matmul(Q, K.transpose(2,3)) 
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        output = self.w_o(self.merge_heads(output))

        return output


class FeedForwardBlock(nn.Module):

    def __init__(self, model_dim, ff_dim):
        super(FeedForwardBlock, self).__init__()

        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class TransBlock(nn.Module):

    # todo: add dropout
    def __init__(self, model_dim, n_heads, ff_dim=None):
        super(TransBlock, self).__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads

        self.ff_dim = model_dim if ff_dim == None else ff_dim

        self.attention_block = MultiHeadAttention(self.model_dim, self.n_heads)
        self.ff_block = FeedForwardBlock(self.model_dim, self.ff_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):

        # Here using prenorm
        x = self.norm1(x)
        x = self.attention_block(x)
        x = self.norm2(x)
        x = self.ff_block(x)

        return x


# https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000) / model_dim))
        pos_enc = torch.zeros(max_len, 1, model_dim)
        pos_enc[:, 0, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 0, 1::2] = torch.cos(position * div_term)

        # todo: what does this do?
        self.register_buffer('pe', pos_enc)

    def forward(self, x):
        x = x + self.pe[:x.size()[0]]
        return self.dropout(x)



    




# test = MultiHeadAttention(12, 4)
        pos_enc[:, 0, 0::2] = torch.sin(position * div_term)
# 
# tnsr = torch.rand(1, 2, 12)
# 
# print(tnsr)
# 
# 
# print(test(tnsr))

# test = TransBlock(12, 4, 30)

# tnsr = torch.rand(2, 3, 12)

# test(tnsr)

tmp = PositionalEncoding(10, max_len=5)