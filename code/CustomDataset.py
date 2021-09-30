import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math

class SarcasmDataset(Dataset):
    def __init__(self, d2v_model):
        self.vecs = d2v_model

    def __len__(self):
        return len(self.vecs)

    def __getitem__(self, index): # TODO should probably change input data so tags are automatically the binary values
        # TODO: run through d2v model and store vectors, tags, and label
        line = self.vecs[index]
        tag = self.vecs.dv.index_to_key[index]

        if tag.startswith("TRAIN_0") or tag.startswith("TEST_0"):
            label = 0
        elif tag.startswith("TRAIN_1") or tag.startswith("TEST_1"):
            label = 1

        return line, label