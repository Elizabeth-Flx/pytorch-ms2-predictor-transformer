
import torch
import torch.nn as nn

import model_parts as mp


class TransMS2Predictor(nn.Module):
    
    def __init__(self, ):
        super(TransMS2Predictor, self).__init__()

        self.embedding_dim = 256
        self.n_transblocks = 8
        self.penult_dim = 512

        self.output_units = 174

        # Embedding (onehot to embedding)
        self.peptide_embedder = nn.Linear(self.embedding_dim)

        self.trans_blocks = [
            # todo: add variables
            mp.TransBlock()
            for _ in range(self.n_transblocks)
        ]

        self.penult_linear = nn.Linear(self.penult_dim)
        self.penult_norm = nn.LayerNorm(self.penult_dim)
        self.relu = nn.ReLU()
        self.final_linear = nn.Linear(self.output_units)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pass






