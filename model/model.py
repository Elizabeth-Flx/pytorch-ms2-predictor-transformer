
import torch
import torch.nn as nn


class TransMS2Predictor(nn.Module):
    
    def __init__(self, ):
        super(TransMS2Predictor, self).__init__()

        self.embedding_dim = 256
        self.n_transblocks = 8

        # Embedding (onehot to embedding)
        self.peptide_embedder = nn.Linear(self.embedding_dim)

        self.trans_blocks = [
            


            for _ in range(n_transblocks)
        ]






