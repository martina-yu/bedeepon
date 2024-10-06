device = "cuda" if torch.cuda.is_available() else "cpu"
bystander_model.init_model(base_editor='ABE', celltype='HEK293T')
model = bystander_model.model

if model is not None:
    model.to(device)
else:
    print("Model instance is not available.")
    
pd.set_option('display.max_colwidth', None)


def generate_mutations(sequence, base, a, b):
    positions = range(a, b)
    print(f"Sequence: {sequence}, Length: {len(sequence)}, Positions: {positions}")
    if base =='C':
        trans = 'T'
    elif base == 'A':
        trans = 'G'
    else:
        raise ValueError("Unsupported base for mutation. Only 'C' and 'A' are supported.")
    
    # print(f'{base}->{trans}')

    base_positions = [pos for pos in positions if sequence[pos] == base]
    
    all_combinations = []
    for r in range(1, len(base_positions) + 1):
        combinations = itertools.combinations(base_positions, r)
        all_combinations.extend(combinations)
    
    mutated_sequences = set()
    mutated_sequences.add(sequence) # !!!!!!!add this if we consider its unedited outcomes!!!!!!!!!!
    for combination in all_combinations:
        mutated_sequence = list(sequence)
        for pos in combination:
            mutated_sequence[pos] = trans
        mutated_sequences.add("".join(mutated_sequence))
    
    return list(mutated_sequences)

def apply_mutations(df, base, a, b):
    start_time = time.time()
    results = []
    
    for index, row in df.iterrows():
        sequence = row['Input']
        id = row['ID']
        mutated_sequences = generate_mutations(sequence, base, a, b)

        for mutated_sequence in mutated_sequences:
            results.append({
                'ID': id,
                'Input': sequence,
                'Output': mutated_sequence
            })
    
    result_df = pd.DataFrame(results)
    end_time = time.time()
    execution_time = end_time - start_time
    print('execution_time:', execution_time)
    return result_df

def generate_sequence(original_sequence, mutations, i):
    start_position = i
    mutated_sequence = list(original_sequence)
    # print(f'mutated_sequence:{mutated_sequence}')
    
    for col, base in mutations.items():
        if '-' in col:
            pos = - int(col.split('-')[1])
            # print(f'pos:{pos}')
            index = pos - start_position
            # print(f'index:{index}')
        else:
            pos = int(col[1:])
            # print(f'esle pos:{pos}')
            index = pos - start_position
            # print(f'esle index:{index}')
        
        mutated_sequence[index] = base
    
    return ''.join(mutated_sequence)

def generate_mutated_sequences(df, original_sequence, i):
    start_position = i
    mutated_sequences = []
    frequencies = []
    
    for idx, row in df.iterrows():
        mutations = row.drop('Predicted frequency').to_dict()
        mutated_sequence = generate_sequence(original_sequence, mutations, start_position)
        mutated_sequences.append(mutated_sequence)
        frequencies.append(row['Predicted frequency'])
    
    result_df = pd.DataFrame({
    'Output': mutated_sequences,
    'Predicted Frequency': frequencies
})
    return result_df

def process_group(lib_ABE, df_input):
    print('begin use process_group function...')
    start_time = time.time()
    merg_df = []
    grps = lib_ABE.groupby('Input')
    
    for group_name, grp in grps:
        # seq = grp.iloc[0,0]
        seq = grp.iloc[0,1]
        
        pred_df, stat = bystander_model.predict(seq)
        result_df_tst = generate_mutated_sequences(pred_df, seq, -19)
        tst = df_input[df_input['Input'] == seq]

        merg = tst.merge(result_df_tst, how = 'left', on = 'Output')
        
        merg_df.append(merg)
    
    print('finish loop')
    end_time = time.time()
    exe_time = end_time - start_time
    print('execute time:',exe_time)
    
    final_df = pd.concat(merg_df, ignore_index=True)

    df_grouped = final_df.groupby('Input')['Predicted Frequency'].transform('sum')
    final_df['Pred_proc'] = final_df['Predicted Frequency'] / df_grouped
    # final_df = final_df[['Input','Output','True_Proportion', 'Pred_proc']]

    pearson_corr = final_df['True_Proportion'].corr(final_df['Pred_proc'])
    print("Pearson's correlation:", pearson_corr)
    
    spearman_corr = final_df['True_Proportion'].corr(final_df['Pred_proc'], method='spearman')
    print("Spearman's correlation:", spearman_corr)
    
    return final_df


def benchmark(df, base):
    
    '''
    `df`: dataframe with 'Input, output, Proportion (no wild-type), Prediction (no wild-type)'
    `base`: 'A' or 'C'
    '''
    
    df_ = df[['Input']] # df_: only keep column 'Input', length = 50
    lib_df = df_.drop_duplicates(subset = 'Input', ignore_index = False) # lib_df: column 'Input' with no duplicated, length = 50
    df_BE = apply_mutations(lib_df, base, 22, 30)
    print(df_BE)

    
    C_true = df.iloc[:,:4] # C_true: keep 'Input, Output, Prediction, Proportion'

    
    df_input = C_true.merge(df_BE, how = 'right', on = ['Input','Output'])
    df_input = df_input.fillna(0)

    df_grouped = df_input.groupby('Input')['Proportion (no wild-type)'].transform('sum')
    df_input['True_Proportion'] = df_input['Proportion (no wild-type)'] / df_grouped
    df_input = df_input[['Input','Output','Prediction (no wild-type)', 'True_Proportion']]
    
    final_df = process_group(lib_df,df_input)
    
    return final_df


def generate_own_50bp(Lib, df):
    df = df.copy()
    
    lib_seqs = Lib['Seq'].tolist()
    lib_seqs_len = len(lib_seqs)
    
    for i in range(len(df)):
        for j in range(lib_seqs_len):
            if df.iloc[i, 0] == lib_seqs[j][20:40]:
                df.loc[i, 'Input'] = lib_seqs[j]
                df.loc[i, 'Output'] = lib_seqs[j][0:20] + df.iloc[i, 1] + lib_seqs[j][40:50]
                break   
                
    return df


def benchmark_own(df, Lib, base):
    print('Lib input: \n',Lib)
    # `df`: test set with 'Input,	Output,	True_Proportion', length = 20
    df = df.drop_duplicates(subset = 'Input')
    df = df.reset_index(drop = True)
    lib_df = df[['Input']]
    df_ = generate_zcd_50bp(Lib, df)

    filter_df = df_[df_['Input'].apply(len) != 50]
    
    df_BE = apply_mutations(df_, base, 22, 30) # generate enumarated sequences
    
    df_input = df_.merge(df_BE, how = 'right', on = ['Input','Output'])
    df_input = df_input.fillna(0)
    

    df_grouped = df_input.groupby('Input')['True_Proportion'].transform('sum')
    df_input['True_Proportion'] = df_input['True_Proportion'] / df_grouped
    
    df_input['True_Proportion'] = df_input['True_Proportion'].fillna(0)
    df_input = df_input[['Input','Output', 'True_Proportion']]
    
    final_df = process_group(lib_df,df_input)

    return final_df

