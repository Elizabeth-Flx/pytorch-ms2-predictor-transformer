import torch 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from model.model import CUSTOM_ALPHABET



class ProteomeToolsDataset(Dataset):
    def __init__(self, paths):
        self.data = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        metadata = np.concatenate([
            row["charge_oh"], 
            row["method_nr_oh"],
            row["machine_oh"],
            np.array(  [row["collision_energy"]]  )
        ])

        metadata = torch.tensor(metadata).float()

        sequence = row["prosit_sequence"]
        sequence = sequence + "-"*(30-len(sequence))
        indexes = [CUSTOM_ALPHABET[aa] for aa in sequence]

        sequence_oh = torch.tensor(np.eye(len(CUSTOM_ALPHABET))[indexes]).float()

        intensities = torch.tensor(row["intensities_raw"]).float()

        return sequence_oh, metadata, intensities
