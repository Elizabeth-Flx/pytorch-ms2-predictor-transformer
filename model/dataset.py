import torch 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from model.model import CUSTOM_ALPHABET



class ProteomeToolsDataset(Dataset):

    def __init__(self, paths):
        data = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)

        self.sequence_oh = data["prosit_sequence"].apply(lambda x: 
            torch.tensor(
                np.eye(len(CUSTOM_ALPHABET))[ [CUSTOM_ALPHABET[aa] for aa in x.ljust(30, '-')] ]
            ).float()
        ).to_numpy()

        self.metadata = data.apply(lambda row:
            torch.tensor(                                    
            np.concatenate([
                row["charge_oh"], 
                row["method_nr_oh"],
                row["machine_oh"],
                np.array(  [row["collision_energy"]]  )
            ])).float(), axis=1
        ).to_numpy()

        self.intensities = data["intensities_raw"].apply(lambda x: torch.tensor(x).float()).to_numpy()

    
    def __len__(self):
        return len(self.sequence_oh)
    
    def __getitem__(self, idx):
        return self.sequence_oh[idx], self.metadata[idx], self.intensities[idx]

