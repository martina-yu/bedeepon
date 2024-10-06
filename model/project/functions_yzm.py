import numpy as np
import pandas as pd
import random
import math
import os
import sys
import pkbar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
import seaborn as sns
from project.utils_yzm import gRNADataset, testDataset, BatchSampler
from ray import tune

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('finishing random seed setting!')

def generate_encoding(ref, out):
    STOI = str.maketrans('ACGT', '0123')
    idx = [int(nuc) for pair in zip(ref.translate(STOI), out.translate(STOI)) for nuc in pair]
    ofs = list(range(0, len(idx), 2))
    return idx, ofs

def generate_batch(batch):
    indices = [idx for item in batch for idx in item[0]]
    offsets = [ofs + i * len(item[0]) for i, item in enumerate(batch) for ofs in item[1]]
    counts = [item[2] for item in batch]    
    return torch.LongTensor(indices), torch.LongTensor(offsets), torch.FloatTensor(counts)

def generate_test_batch(batch):
    indices = [idx for item in batch for idx in item[0]]
    offsets = [ofs + i * len(item[0]) for i, item in enumerate(batch) for ofs in item[1]]
    counts = [item[2] for item in batch]
    ref = [item[3] for item in batch]
    otm = [item[4] for item in batch]
    return torch.LongTensor(indices), torch.LongTensor(offsets), torch.FloatTensor(counts), ref, otm

def init_weights(m):
    '''
    Optimized: to prevent early-stage vanishing gradients in LSTMs, can let model learn long-term dependencies better.
    '''
    for name, param in m.named_parameters():
        if 'rnn.weight_' in name:
            nn.init.orthogonal_(param.data)
        elif 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        elif 'bias' in name:
            if 'rnn.bias_' in name:
                nn.init.constant_(param.data, 1)
            else:
                nn.init.constant_(param.data, 0)


def train_model(model, epochs, clips, fold, trained_model_path, train_dataloader, valid_dataloader, criterion_KLD, criterion_mse, device, basename, optimizer):
    
    train_loss_list = []
    valid_loss_list = []
    
    best_valid_loss = np.inf

    for epoch in range(epochs):
        model.train()
        train_per_epoch = len(train_dataloader)
        kbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=epochs, width=8, always_stateful=False)
        train_loss = 0
        for i, batch in enumerate(train_dataloader):
            indices, offsets, counts = batch
            indices = indices.to(device)
            offsets = offsets.to(device)
            counts = counts.to(device)
            all_counts = counts.sum() + 1e-6
            y_true = counts / all_counts
            batch_size = int(len(indices) / 40)
            
            optimizer.zero_grad()
            outputs = model(indices, offsets, batch_size)
            
            y_pred_1 = F.log_softmax(outputs, 0).view(-1)
            loss_temp_1 = criterion_KLD(y_pred_1, y_true)
            loss_temp_1 = torch.abs(loss_temp_1 * y_true) * 100
            
            y_pred_2 = F.softmax(outputs, 0).view(-1)
            loss_temp_2 = criterion_mse(y_pred_2, y_true)
            loss_temp_2 = loss_temp_2 * y_true * 100
            
            loss1 = loss_temp_1.mean()
            loss2 = loss_temp_2.mean()
            
            loss = 0.25 * loss1 + 0.75 * loss2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            kbar.update(i, values=[("training loss", train_loss/(i + 1)),("loss1",loss1),("loss2",loss2),("lr", optimizer.defaults['lr'])])
        
        train_loss_ = train_loss / len(train_dataloader)
        train_loss_list.append(train_loss_)
        print(f"\nbegin to validate model: ...\n")
  
        ######## switch to validation mode: #########
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for batch in valid_dataloader:
                indices, offsets, counts = batch
                indices = indices.to(device)
                offsets = offsets.to(device)
                counts = counts.to(device)
                all_counts = counts.sum() + 1e-6
                y_true = counts / all_counts
                batch_size = int(len(indices) / 40)
    
                outputs = model(indices, offsets, batch_size)
                
                pred_y_1 = F.log_softmax(outputs, dim=0).view(-1)
                loss_temp_1 = criterion_KLD(pred_y_1, y_true)
                loss_temp_1 = torch.abs(loss_temp_1 * y_true) * 100

                pred_y_2 = F.softmax(outputs, dim=0).view(-1)
                loss_temp_2 = criterion_mse(pred_y_2, y_true) * 100
                loss_temp_2 = loss_temp_2 * y_true
                
                loss1 = loss_temp_1.mean()
                loss2 = loss_temp_2.mean()

                loss = 0.25 * loss1 + 0.75 * loss2
                
                valid_loss += loss.item()
                
        valid_loss /= len(valid_dataloader)
        valid_loss_list.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{trained_model_path}/{basename}/{fold+1}_{best_valid_loss}.pth')
            print(f"\nNew best model saved for fold {fold+1} with validation loss {best_valid_loss:.6f}")


        print(f'\nFold {fold + 1}, Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss_:.6f}, Validation Loss: {best_valid_loss:.6f}\n')
    
    return train_loss_list, valid_loss_list

def test_model(test_dataloader, model, criterion_KLD, criterion_mse, device):
    model.eval()
    test_loss = 0
    lst_pred = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            indices, offsets, counts, ref, otm = batch
            indices = indices.to(device)
            offsets = offsets.to(device)
            counts = counts.to(device)
            all_counts = counts.sum() + 1e-6
            y_true = counts / all_counts
            batch_size = int(len(indices) / 40)

            outputs = model(indices, offsets, batch_size)
            
            pred_y_1 = F.log_softmax(outputs, dim=0).view(-1)
            loss_temp_1 = criterion_KLD(pred_y_1, y_true)
            loss_temp_1 = torch.abs(loss_temp_1 * y_true) * 100
            
            pred_y_2 = F.softmax(outputs, dim=0).view(-1)
            loss_temp_2 = criterion_mse(pred_y_2, y_true) * 100
            loss_temp_2 = loss_temp_2 * y_true
            
            loss1 = loss_temp_1.mean()
            loss2 = loss_temp_2.mean()
            loss = loss1 * 0.5 + loss2 * 0.5
            test_loss += loss.item()

            df_on = pd.DataFrame({'Reference': ref,'Outcomes': otm})
            df_on['true'] = y_true.cpu().detach().numpy()
            df_on['pred'] = pred_y_2.cpu().detach().numpy()
            lst_pred.append(df_on)
            
    test_loss /= len(test_dataloader)
    df = pd.concat(lst_pred)
    return test_loss,df

def loss_plot(fold_list, epoch_list, train_list, valid_list, file_path, basename, mean_valid_loss, base):

    result_file = f"{file_path}.csv"
    df = pd.DataFrame({
        'Fold': fold_list,
        'Epoch': epoch_list,
        'Train loss': train_list,
        'Valid loss': valid_list
    })
    df.to_csv(result_file, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(data=df, x='Epoch', y='Train loss', hue='Fold', ax=ax1)
    ax1.set_title(f'Training Loss Over Epochs {base}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')

    sns.lineplot(data=df, x='Epoch', y='Valid loss', hue='Fold', ax=ax2)
    ax2.set_title(f'Validation Loss Over Epochs {base}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    plt.savefig(f'{file_path}.png')
    plt.tight_layout()

def process_data(df):
    df['index'], df['offsets'] = zip(*df.apply(lambda row: generate_encoding(row['Reference'], row['Outcomes']), axis = 1)) ## for each rows
    df = df[['Reference','Outcomes','Count','index','offsets']]
    grp_df = dict(list(df.groupby('Reference')))
    list_gRNA = df['Reference'].unique()
    df_dataset = gRNADataset(df[['index','offsets','Count']])
    return df_dataset,list_gRNA, grp_df

def process_data_test(df):
    df['index'], df['offsets'] = zip(*df.apply(lambda row: generate_encoding(row['Reference'], row['Outcomes']), axis = 1)) ## for each rows
    df = df[['index','offsets','True_Proportion','Reference','Outcomes']]
    grp_df = dict(list(df.groupby('Reference')))
    list_gRNA = df['Reference'].unique()
    df_dataset = testDataset(df) ## testDataset
    return df_dataset,list_gRNA, grp_df

def correlation(df):
    grp = df.groupby('Reference')
    
    df_sprm = df[df['Reference'] != df['Outcomes']]
    df_sprm['All_True'] = df_sprm.groupby('Reference')['true'].transform('sum')
    df_sprm['true'] = df_sprm['true'] / df_sprm['All_True']
    df_sprm['All_Pred'] = df_sprm.groupby('Reference')['pred'].transform('sum')
    df_sprm['pred'] = df_sprm['pred'] / df_sprm['All_Pred']
    
    grp_sprm = df_sprm.groupby('Reference')
    
    all_sprm = df_sprm['true'].corr(df_sprm['pred'], method='spearman')
    all_prs = df['true'].corr(df['pred'], method='pearson')

    lst_sprm = []
    lst_prs = []

    for k in grp_sprm.groups.keys():
        df_sprm_grp = grp_sprm.get_group(k)
        spearman_corr = df_sprm_grp['true'].corr(df_sprm_grp['pred'], method='spearman')
        lst_sprm.append({'Reference': k, 'Spearman_Correlation': spearman_corr})

    for k in grp.groups.keys():
        df_grp = grp.get_group(k)
        pearson_corr = df_grp['true'].corr(df_grp['pred'], method='pearson')
        lst_prs.append({'Reference': k, 'Pearson_Correlation': pearson_corr})

    sprm_df = pd.DataFrame(lst_sprm)
    prs_df = pd.DataFrame(lst_prs)
    
    return all_sprm, all_prs, sprm_df, prs_df

def correlation(df):
    grp = df.groupby('Reference')
    df = df[df['true'] >= 1e-5]
    all_sprm = df['true'].corr(df['pred'], method='spearman')
    all_prs = df['true'].corr(df['pred'], method='pearson')

    lst_sprm = []
    lst_prs = []
    
    for k,df_grp in grp:
        if len(df_grp) < 2:
            continue
        
        df_grp = grp.get_group(k)
        pearson_corr = df_grp['true'].corr(df_grp['pred'], method='pearson')
        spearman_corr = df_grp['true'].corr(df_grp['pred'], method='spearman')
        lst_prs.append({'Reference': k, 'Pearson_Correlation': pearson_corr})
        lst_sprm.append({'Reference': k, 'Spearman_Correlation': spearman_corr})

    sprm_df = pd.DataFrame(lst_sprm)
    prs_df = pd.DataFrame(lst_prs)

    print(f'Spearman: {all_sprm}\nPearson: {all_prs}')

    return all_sprm, all_prs, sprm_df, prs_df