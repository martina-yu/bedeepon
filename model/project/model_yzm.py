import numpy as np
import pandas as pd
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
from project.utils_yzm import BatchSampler, gRNADataset, testDataset
from project.functions_yzm import *


class BiLSTMTrainable(tune.Trainable):
    def setup(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file_path = config["file_path"]
        self.storage_path = config["storage_path"]
        self.trained_model_path = config["trained_model_path"]
        self.basename = config["basename"]
        self.epoch = config["epochs"]
        self.criterion_KLD = nn.KLDivLoss(reduction='none')
        self.criterion_mse = nn.MSELoss(reduction='none')

        # Load Training dataset
        self.train_data = pd.read_csv(f'{self.file_path}.csv', usecols=['Reference', 'Outcomes', 'Count'])
        self.train_data = self.train_data.groupby('Reference').filter(lambda x: x['Count'].sum() > 0)
        self.train_data = self.train_data.sort_values(by='Reference')
        self.train_data = self.train_data.reset_index(drop=True)

        self.train_dataset, self.list_gRNA, self.grp_df = process_data(self.train_data)

        self.kf = KFold(n_splits=5, shuffle=True, random_state=2024)
        
        self.fold_list = []
        self.epoch_list = []
        self.train_list = []
        self.valid_list = []
        self.fold = 0
        self.best_valid_losses = []
        self.best_valid_loss = float('inf')

        self.model = BiLSTM_Attention(4, config["emb_dim"], config["hid_dim"], config["layers"], config["dropout"]).to(self.device)
        self.model.apply(init_weights)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    def step(self):
        config = self.config

        if self.fold >= 5:
            raise StopIteration
        
        train_index, valid_index = list(self.kf.split(self.list_gRNA))[self.fold]
        
        train_fold_data = self.train_data.loc[train_index].copy().reset_index(drop=True)
        valid_fold_data = self.train_data.loc[valid_index].copy().reset_index(drop=True)

        train_dataset = gRNADataset(train_fold_data)
        valid_dataset = gRNADataset(valid_fold_data)
        
        train_dataloader = DataLoader(train_dataset, batch_sampler=BatchSampler(train_fold_data), collate_fn=generate_batch)
        valid_dataloader = DataLoader(valid_dataset, batch_sampler=BatchSampler(valid_fold_data), collate_fn=generate_batch)

        train_loss_list, valid_loss_list = train_model(self.model, config["epochs"], config["clips"], self.fold, config["trained_model_path"], train_dataloader, valid_dataloader, self.criterion_KLD, self.criterion_mse, self.device, self.basename, self.optimizer)

        self.best_valid_loss = min(valid_loss_list)
        self.best_valid_losses.append(self.best_valid_loss)
        self.fold_best_valid_loss = min(self.best_valid_losses)
        self.best_fold = self.best_valid_losses.index(min(self.best_valid_losses)) + 1

        print(f"Fold {self.fold + 1}: best valid loss = {fold_best_valid_loss}")
        
        self.fold += 1
        
        return {"fold_best_valid_loss": self.fold_best_valid_loss, "best_fold": self.best_fold}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        torch.save({
            'fold': self.fold,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_valid_losses': self.best_valid_losses
        }, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.fold = checkpoint['fold']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_valid_losses = checkpoint['best_valid_losses']