
import torch
import torch.nn as nn

from model import model_parts as mp

CUSTOM_ALPHABET = {
    '-': 0,
    'A': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'V': 18,
    'W': 19,
    'Y': 20,
    'm': 21, # oxidized methionine
}
ALPHABET_SIZE = len(CUSTOM_ALPHABET)


class TransMS2Predictor(nn.Module):
    
    def __init__(self, ):
        super(TransMS2Predictor, self).__init__()

        self.embedding_dim = 256
        self.n_transblocks = 8
        self.penult_dim = 512

        self.output_units = 174

        # Metadata Encoder (charge + method + machine + energy)
        self.metadata_encoder = nn.Linear(6+2+3+1, 2*self.embedding_dim)

        # Embedding (onehot to embedding)
        self.peptide_embedder = nn.Linear(ALPHABET_SIZE, self.embedding_dim)

        # Positional Encoding
        self.pos_enc = mp.PositionalEncoding(self.embedding_dim)

        self.trans_blocks = [
            # todo: add variables
            mp.TransBlock(self.embedding_dim, 8, 128)
            for _ in range(self.n_transblocks)
        ]

        self.penult_linear = nn.Linear(self.embedding_dim, self.penult_dim)
        self.penult_norm = nn.LayerNorm(self.penult_dim)
        self.relu = nn.ReLU()
        self.final_linear = nn.Linear(self.penult_dim, self.output_units)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, metadata):
        
        # Metadata processing
        metadata = self.self.metadata_encoder(metadata)     # bs, 2*emb_dim
        metadata = metadata[:, None, :]                     # bs, 1, 2*emb_dim  
        gamma, beta = torch.chunk(metadata, 2, dim=-1)      # bs, 1, emb_dim

        # Peptide processing
        x = self.peptide_embedder(x)
        x = self.pos_enc(x)                                 # bs, seq_len, emb_dim

        # Integrate metadata
        x = x * gamma + beta

        for i in range(len(self.trans_blocks)):
            trans_block = self.trans_blocks[i]
            x = trans_block(x)

        x = self.penult_linear(x)                           # bs, seq_len, penult_dim
        x = self.penult_norm(x)
        x = self.relu(x)

        x = self.final_linear(x)                            # bs, seq_len, output_units
        x = self.sigmoid(x)

        x = torch.mean(x, dim=1, keepdim=True)              # bs, 1, output_units

        return x







# test = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_test.parquet"




# import pandas as pd
# tmp = pd.read_parquet(test, engine='pyarrow')

# print(tmp)


