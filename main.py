# import packages
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math

class SarcasmDataset(Dataset):
    def __init__(self):
        self.train_data = pd.read_csv("./archive/train-balanced-sarcasm.csv")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        return self.train_data.iloc[item]  # turn to tensor before returning (using Doc2Vec)