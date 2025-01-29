
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

    def forward(self, x):
        
        print(x.size())
        x = self.peptide_embedder(x)

        print(x.size())





# test = "/cmnfs/data/proteomics/Prosit_PTMs/Transformer_Train/no_aug_test.parquet"




# import pandas as pd
# tmp = pd.read_parquet(test, engine='pyarrow')

# print(tmp)


