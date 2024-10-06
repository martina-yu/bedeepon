agg_df = bench_ABEmax.pred(df_ABE_)
df_CBE_prop = agg_df.copy()
df_CBE_prop = df_CBE_prop.reset_index(drop = True)
df_CBE_prop = df_CBE_prop[['seq_id', 'Inp_seq', 'Outp_seq', 'pred_score']]
df_CBE_prop['True_pred'] = df_CBE_prop.groupby('Inp_seq')['pred_score'].transform('sum')
df_CBE_prop['True_pred'] = df_CBE_prop['pred_score'] / df_CBE_prop['True_pred']

cmpr = df.copy()
print(cmpr.columns)
cmpr = cmpr.rename(columns = {'Input':'Inp_seq','Output':'Outp_seq'})

cor = pd.merge(cmpr, df_CBE_prop, how = 'outer', on = ['Inp_seq','Outp_seq'])

print('merge finished!')
pearson_corr = cor['True_Proportion'].corr(cor['True_pred'])
print("Pearson's correlation:", pearson_corr)

spearman_corr = cor['True_Proportion'].corr(cor['True_pred'], method='spearman')
print("Spearman's correlation:", spearman_corr)

if length != 0:
    print("edit didn't happen on window 3 -> 10")
else:
    print('check window 3 -> 10 successfully!')

    df_sub = df_.copy()
    # df_sub = df_sub.rename(columns = {'Input': ''})
    print('What columns they have:',df_sub.columns)
    
    group_ids = {group: f"seq_{i}" for i, group in enumerate(df_sub['Input'].unique())}
    df_sub['seq_id'] = df_sub['Input'].map(group_ids)
    df_ABE = df_sub[['seq_id','Input']]
    
    print('length of df_ABE:',len(df_ABE))
    df_ABE_ = df_ABE.drop_duplicates(subset = 'Input')
    print('length of df_ABE_ with dropping duplicates:',len(df_ABE_))
    
agg_df = bench_ABEmax.pred(df_ABE_)

bench_ABEmax.check_and_return_prop(agg_df)

df_sub = df_sub.drop(columns = ['count', 'all_count'])
df_sub['seq_id'] = 1

df_sub.columns = ['Inp_seq', 'Outp_seq', 'true_score', 'seq_id']
df_sub = df_sub.reindex(columns = list(['Editor', 'Inp_seq', 'Outp_seq', 'true_score', 'seq_id']))
print(df_sub)
df_sub['Editor'] = df_sub.fillna(value = 'ABEmax').copy()

csv_dir = create_directory(os.path.join(curr_pth, 'sample_data', 'predictions_haplo'))
report_available_cuda_devices()
for gr_name, gr_df in df_sub.groupby(by=['Editor']):
    print(gr_name)
    display(gr_df)
    
seqconfig_dataproc = SeqProcessConfig(20, (1,20), (3,10), 1)
print(seqconfig_dataproc)

teditor = 'ABEmax'
cnv_nucl = ('A', 'G')
seq_processor = HaplotypeSeqProcessor(teditor, cnv_nucl, seqconfig_dataproc)
bedict_haplo = BEDICT_HaplotypeModel(seq_processor, seqconfig_datatensgen, device)
dloader = bedict_haplo.prepare_data(df_sub,
                                    ['seq_id','Inp_seq'],
                                    outpseq_col=None,
                                    outcome_col=None,
                                    renormalize=False,
                                    batch_size=500)
num_runs = 5
pred_df_lst = []
for run_num in range(num_runs):
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
tseqids = df_sub['seq_id']

res_html = bedict_haplo.visualize_haplotype(agg_df, 
                                            tseqids, 
                                            ['seq_id','Inp_seq'], 
                                            'Outp_seq', 
                                            'pred_score', 
                                            predscore_thr=0.)
for seq_id in res_html:
    display(HTML(res_html[seq_id]))
    

vf = HaplotypeVizFile(os.path.join(curr_pth, 'BE-DICT', 'model_crispr', 'haplotype', 'viz_resources'))
for seq_id in res_html:
    display(HTML(res_html[seq_id]))
    vf.create(res_html[seq_id], csv_dir, f'{teditor}_{seq_id}_haplotype')