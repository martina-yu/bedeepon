import numpy as np
import pandas as pd
import random
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
import seaborn as sns

class BatchSampler(Sampler):
    def __init__(self, data):
        idx_list = []
        for ref, sub_df in data.groupby('Reference'):
            idx_list.append(sub_df.index.tolist())
        self.sampler = SubsetRandomSampler(idx_list)
    
    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for idx in self.sampler:
            yield idx

class gRNADataset:
    def __init__(self, df):
        self.df = df
        self.indices = self.df['index'].tolist()
        self.offsets = self.df['offsets'].tolist()
        self.cnt = self.df['Count'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        indices = self.indices[idx]
        offsets = self.offsets[idx]
        cnt = self.cnt[idx]
        return indices, offsets, cnt

class testDataset:
    def __init__(self, df):
        self.df = df
        self.indices = self.df['index'].tolist()
        self.offsets = self.df['offsets'].tolist()
        self.cnt = self.df['True_Proportion'].tolist()
        self.ref = self.df['Reference'].tolist()
        self.otm = self.df['Outcomes'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        indices = self.indices[idx]
        offsets = self.offsets[idx]
        cnt = self.cnt[idx]
        ref = self.ref[idx]
        otm = self.otm[idx]
        return indices, offsets, cnt, ref, otm
