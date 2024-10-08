import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import haplotype
from haplotype.dataset import *
from haplotype.data_preprocess import *
from haplotype.utilities import *
from haplotype.predict_model import BEDICT_HaplotypeModel


def generate_model_input(df, unedited):
    '''
    `df`: 'Input' and 'Output'
    `model input`: need 'seq_id','seq'
    '''
    print('length of df:', len(df))
    df['mksure'] = 0
    if unedited=='edited':
        if len(df.iloc[1,1]) == 20:
            for i in range(len(df)):
                seq1 = df.loc[i, 'Input']
                seq2 = df.loc[i, 'Output']
                if seq1 == seq2:
                    df.loc[i,'mksure'] = 2
                else:
                    df.loc[i,'mksure'] = 1
    
        else:
            for i in range(len(df)):
                '''
                check if edit happens in window 3 -> 10:
                '''
                seq1 = df.loc[i, 'Input'][0:2]+ df.loc[i, 'Input'][10:20] #### Need to adjust to the length of seqence
                seq2 = df.loc[i, 'Output'][0:2]+ df.loc[i, 'Output'][10:20] #### Need to adjust to the length of seqence
                if seq1 == seq2:
                    df.loc[i,'mksure'] = 1
                else:
                    df.loc[i,'mksure'] = 2
            
        length = len(df[df['mksure'] == 2])
        
        if length != 0:
            print("edit didn't happen on window 3 -> 10")
        else:
            print('check window 3 -> 10 successfully!')
    else:
        df_sub = df.copy()
        # df_sub = df_sub.rename(columns = {'Input': ''})
        print('What columns they have:',df_sub.columns)
        
        group_ids = {group: f"seq_{i}" for i, group in enumerate(df_sub['Input'].unique())}
        df_sub['seq_id'] = df_sub['Input'].map(group_ids)
        df_ABE = df_sub[['seq_id','Input']]
        
        print('length of df_ABE:',len(df_ABE))
        df_ABE_ = df_ABE.drop_duplicates(subset = 'Input')
        print('length of df_ABE_ with dropping duplicates:',len(df_ABE_))
    
        return df_ABE_


def pred(df_ABE_):
    
    print('report available cuda devices:',report_available_cuda_devices())

    device = get_device(True, 0)
    print('device:', device)
    
    seqconfig_dataproc = SeqProcessConfig(20, (1,20), (3,10), 1)
    print(seqconfig_dataproc)
    
    teditor = 'ABEmax'
    cnv_nucl = ('A', 'G')
    seq_processor = HaplotypeSeqProcessor(teditor, cnv_nucl, seqconfig_dataproc)
    
    seqconfig_datatensgen = SeqProcessConfig(20, (1,20), (3, 20), 1)
    
    bedict_haplo = BEDICT_HaplotypeModel(seq_processor, seqconfig_datatensgen, device)

    dloader = bedict_haplo.prepare_data(df_ABE_,
                                    ['seq_id','Input'],
                                    outpseq_col=None,
                                    outcome_col=None,
                                    renormalize=False,
                                    batch_size=500)

    print('dloader work finished! Begin generating prediction')
    
    num_runs = 5 
    pred_df_lst = [] 
    for run_num in range(num_runs):
        path = '/home/user/notebooks/yuzm/BEDEEPON/BE_DICT/model_crispr/trained_models/bystander/ABEmax/train_val' ## need to change!!!!!!!!!!!!!!!
        model_dir = os.path.join(path, f'run_{run_num}')
        print('run_num:', run_num)
        print('model_dir:', model_dir)
    
        pred_df = bedict_haplo.predict_from_dloader(dloader,
                                                    model_dir, 
                                                    outcome_col=None)
        pred_df['run_num'] = run_num
        pred_df_lst.append(pred_df)
        
    pred_df_unif = pd.concat(pred_df_lst, axis=0, ignore_index=True)
    check_na(pred_df_unif)
    agg_df = bedict_haplo.compute_avg_predictions(pred_df_unif)
    check_na(agg_df)

    return agg_df

def check_and_return_prop(agg_df):

    '''
    To check if numbers of unedited is equal to numbers of 'df library":
    `df`: with column 'Input', 'Output', length = 20
    '''
    agg_df['unedited'] = 2
    df_lib = agg_df.drop_duplicates(subset = 'Inp_seq')
    
    for i in range(len(agg_df)):
        if agg_df.loc[i, 'Inp_seq'] == agg_df.loc[i, 'Outp_seq']:
            agg_df.loc[i,'unedited'] = 0
        else:
            agg_df.loc[i, 'unedited'] = 1
    
    print(agg_df[agg_df['unedited'] == 0]) 

    
    a = len(agg_df[agg_df['unedited'] == 0])
    b = len(df_lib)
    
    print('length of unedited', a)
    print('length of lib', b)

    if a == b:
        return agg_df[agg_df['unedited'] == 1]
    else:
        return None


def correlation_of_prop(df, name):
    '''
    generate correlation

    `sub_df`: the input df!
    `name`: author name
    
    '''
    df_ABE_ = generate_model_input(df)
    
    agg_df = pred(df_ABE_)
    # agg_df.to_csv('./benchmark/' + name + '_ABE_freq.csv')
    
    check_df = check_and_return_prop(agg_df)

    ### Process Proportion
    if check_df is None:
        print('procedure in prediction wrong!')
        return None
    else:
        df_ABE_prop = check_df.copy()
        df_ABE_prop = df_ABE_prop.reset_index(drop = True)
        df_ABE_prop = df_ABE_prop[['seq_id', 'Inp_seq', 'Outp_seq', 'pred_score']]
        df_ABE_prop['True_pred'] = df_ABE_prop.groupby('Inp_seq')['pred_score'].transform('sum')
        df_ABE_prop['True_pred'] = df_ABE_prop['pred_score'] / df_ABE_prop['True_pred']
        df_ABE_prop.to_csv('./benchmark_new/' + name + '_ABE_prop.csv')
        # cmpr = pd.read_csv('./data/song_prop_ABE_for_DICT.csv', index_col = 0)
        
        cmpr = df.copy()
        print(cmpr.columns)
        cmpr = cmpr.rename(columns = {'Input':'Inp_seq','Output':'Outp_seq'})

        cor = pd.merge(cmpr, df_ABE_prop, how = 'outer', on = ['Inp_seq','Outp_seq'])

        print('merge finished!')
        pearson_corr = cor['True_Proportion'].corr(cor['True_pred'])
        print("Pearson's correlation:", pearson_corr)
        
        spearman_corr = cor['True_Proportion'].corr(cor['True_pred'], method='spearman')
        print("Spearman's correlation:", spearman_corr)
        
        return pearson_corr, spearman_corr, cor
