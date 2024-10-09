def process_data(file_path):
    pandarallel.initialize(nb_workers=20)
    data_list = []
    for data in pd.read_csv(file_path, header=0, sep='\t',chunksize=1e5):
        # data = pd.pivot_table(data, values='count', index=['gRNA', 'target'], columns=['is_edit'], aggfunc='sum').fillna(0)
        data = data[(data['is_edit'] != -1) & (data['tgt_len'] == 23) & (data['is_mis_synthesis'] == 0)]
        data_list.append(data)
    data = pd.concat(data_list,ignore_index=True)
    data.columns = ['gRNA', 'target','tgt_len','is_mis_synthesis','is_edit']
    data['gRNA'] = data['gRNA'].str[:-3]
    data['target'] = data['target'].str[:-3]
        
    file_path = os.path.basename(file_path)

    if 'ABE' in file_path:
        file_df = data[data.parallel_apply(check_canonical_ABE, axis=1)]
    else:
        file_df = data[data.parallel_apply(check_canonical_CBE, axis=1)]
        
    file_df['count'] = file_df.groupby(['gRNA', 'target'])['gRNA'].transform('count')
    file_df = file_df.drop_duplicates(subset=['gRNA', 'target'], keep='first') # 去除重复的行
    file_df['all_count'] = file_df.groupby(['gRNA'])['count'].transform('sum')
    file_df['unedited_count'] = file_df.groupby(['gRNA','target'])['count'].transform('sum')
    file_df['unedited_proportion'] = file_df['unedited_count'] / file_df['all_count']
    file_df['overall_efficiency'] = 1 - file_df['unedited_proportion'][file_df['is_edit'] == 0]

    groups = file_df.groupby(['gRNA'])
    for name, group in groups:
        equal_rows = group[group['gRNA'] == group['target']]
        if not equal_rows.empty:
            c_value = equal_rows.iloc[0]['overall_efficiency']
            file_df.loc[(file_df['gRNA'] == name) & (file_df['gRNA'] != file_df['target']), 'overall_efficiency'] = c_value

    cano_edited = file_df[file_df['gRNA'].str[2:10] != file_df['target'].str[2:10]].copy()
    cano_edited["deno"] = cano_edited.groupby('gRNA')['count'].transform('sum')
    cano_edited["proportion"] = cano_edited["count"] / cano_edited["deno"]
    unedited = (file_df['gRNA'] == file_df['target'])
    cano_unedited = file_df[unedited]
    processed_cano_seq = pd.concat([cano_edited, cano_unedited], axis=0)
    processed_cano_seq = processed_cano_seq.fillna(0)

    processed_cano_seq = processed_cano_seq.sort_values(by=['gRNA', 'target'])
    processed_cano_seq = processed_cano_seq.reset_index(drop=True)
    processed_cano_seq = processed_cano_seq.drop(columns=['tgt_len','is_mis_synthesis','is_edit','unedited_count','unedited_proportion','deno'])
    
    processed_cano_seq.to_csv(file_path[0:-4] + '_processed.csv', header=True, encoding='utf-8', index=False)
    

def check_canonical_ABE(file_data):
    gRNA = file_data['gRNA']
    target = file_data['target']
    for i in range(len(gRNA)):
        if gRNA[i] != target[i]:
            if not (gRNA[i] == 'A' and target[i] == 'G'):
                return False
    return True

def check_canonical_CBE(file_data):
    gRNA = file_data['gRNA']
    target = file_data['target']
    for i in range(len(gRNA)):
        if gRNA[i] != target[i]:
            if not (gRNA[i] == 'C' and target[i] == 'T'):
                return False
    return True



