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

# class BiLSTM_Attention(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):

#         super(BiLSTM_Attention, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.embedding = nn.EmbeddingBag(input_dim, embedding_dim) ## try nn.Embedding, it might be a more natural fit
        
#         '''
#         torch.nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, mode='mean', sparse=False, _weight = None, include_last_offset=False, padding_idx=None, device=None, dtype=None)
#         '''
#         self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = n_layers, bidirectional = True, batch_first = True)
#         self.fc = nn.Linear(hidden_dim * 2, 1)
#         self.dropout = nn.Dropout(dropout)

#     def attention_net(self, x, query, mask=None):

#         d_k = query.size(-1)
#         scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k + 1e-8)
#         alpha_n = F.softmax(scores, dim=-1)
#         context = torch.matmul(alpha_n, x).sum(1)

#         return context, alpha_n

#     def forward(self, seq, offset, batch_size):
#         emb = self.dropout(self.embedding(seq, offset))
#         emb_v = emb.view(batch_size, 20, -1) ## seq_length = 20
#         emb_vt = emb_v.transpose(1,0)
#         out, (hidden, _) = self.rnn(emb_vt)
#         out = out.permute(1, 0, 2)
#         '''
#         out = out.permute(1, 0, 2) ---> (seq_length, batch_size, hidden_dim), to rearrange the dimensions of LSTM output, let "seq_length, batch_size" swap position. It can be update with setting "batch_first = True"
#         '''
#         query = self.dropout(out)
#         attn_output, alpha_n = self.attention_net(out, query)
#         logit = F.leaky_relu(self.fc(attn_output)) ## also try ReLU, tanh and *sigmoid(might be better)

#         return logit