def generate_mutations(sequence, base, a, b):
    positions = range(a, b)
    
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
    # mutated_sequences.add(sequence)
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
        mutated_sequences = generate_mutations(sequence, base, a, b)

        for mutated_sequence in mutated_sequences:
            results.append({
                'Input': sequence,
                'Output': mutated_sequence
            })
    
    result_df = pd.DataFrame(results)
    end_time = time.time()
    execution_time = end_time - start_time
    print('execution_time:', execution_time)
    return result_df


def generate_fasta_file(name):
    df_ = pd.read_csv('./../../DEEPBE_'+ name + '_prop_ABE.csv')
    df_lib = df_[['Sequence ID','Input']]
    df_lib = df_lib.drop_duplicates(subset = 'Input')
    df_lib.to_csv('DEEPBE_'+ name + '_ABE_lib.csv')

    fasta_file = 'DEEPBE_' + name + '_ABE.fasta'

    with open(fasta_file, 'w') as f:
        for index, row in df_lib.iterrows():
            seq_id = row['Sequence ID']
            sequence = row['Input']
            f.write(f'>seq{seq_id}\n') # for only with number or different name
            # f.write(f'>{seq_id}\n') # for seq with alphabet
            f.write(f'{sequence}\n')
    
    print(f'FASTA file has been saved as {fasta_file}')
    return

def benchmark_DEEPABE(name):
    input_file = './outputs/'+ name + '_ABE_prop.txt'
    output_file = './outputs/'+ name + '_ABE_prop_fixed.txt'
    df_lib = pd.read_csv('DEEPBE_'+ name + '_ABE_lib.csv')
    
    with open(input_file, 'r') as file:
        lines = file.readlines()

######### ================= for arbab's dataset ==========================
    # with open(output_file, 'w') as file:
    #     for line in lines:
    #         line = line.replace('Poly Tsatmut', 'satmut')
    
    #         parts = line.split('\t')
    #         new_line = []
            
    #         for part in parts:
    #             if 'satmut_6mer' in part or 'satmut_4mer' in part:
    #                 if new_line:
    #                     file.write('\t'.join(new_line) + '\n')
    #                 new_line = [part]
    #             else:
    #                 new_line.append(part)
            
    #         if new_line:
    #             file.write('\t'.join(new_line) + '\n')
        # with open(input_file, 'r') as file:
    #     lines = file.readlines()


######### ================= for song's dataset ==========================
    
    with open(output_file, 'w') as file:
        for line in lines:   
            parts = line.split('\t')
            new_line = []
            
            for part in parts:
                if 'seq' in part:
                    if new_line:
                        file.write('\t'.join(new_line) + '\n')
                    new_line = [part]
                else:
                    new_line.append(part)
            
            if new_line:
                file.write('\t'.join(new_line) + '\n')
    
    df = pd.read_csv(output_file, delim_whitespace=True)
    df = pd.read_csv('./outputs/'+ name + '_ABE_prop_fixed.txt', delim_whitespace=True)


    df = df.sort_values(by = 'ID')
    df = df.reset_index(drop = True)
    df = df.iloc[:,:10]
    df.to_csv('./outputs/' + name + '_ABE_prop_outcome.csv')

    result_df = apply_mutations(df_lib,'A',7,14) #### seq length = 30

    df = df[['ID', 'Guide Sequence (20bp)','Outcome Sequence (26bp)','ABE Efficiency','ABE Proportion']]
    df = df.rename(columns = {'Guide Sequence (20bp)':'Input','Outcome Sequence (26bp)':'Output'})
    df['Output'] = df['Output'].str[3:23]
    result_df['Input'] = result_df['Input'].str[4:24]
    result_df['Output'] = result_df['Output'].str[4:24]
    pred_df = df.merge(result_df, on = ['Input', 'Output'])
    
    df_['Input'] = df_['Input'].str[4:24]
    df_['Output'] = df_['Output'].str[4:24]
    df_ = df_.iloc[:,:6]
    enu_df = result_df.merge(df_, how = 'left', on = ['Input', 'Output'])

    enu_df = enu_df[['Input', 'Output', 'Proportion (no wild-type)']] ###### for without prediction!!!!!!!
    enu_df['Proportion (no wild-type)'] = enu_df['Proportion (no wild-type)'].fillna(0)
    
    name_df = pred_df.merge(enu_df, on = ['Input', 'Output'])
    print(f'{name}_ABE_proportion has {len(name_df)} rows')
    print(name_df.dtypes)
    name_df['ABE Proportion'] = pd.to_numeric(name_df['ABE Proportion'], errors='coerce')
    
    final_df = name_df.copy()
    df_grouped = final_df.groupby('Input')['ABE Proportion'].transform('sum')
    final_df['ABE Proportion'] = final_df['ABE Proportion'] / df_grouped
    final_df['ABE Proportion'] = final_df['ABE Proportion'].fillna(0)
    df_grouped = final_df.groupby('Input')['Proportion (no wild-type)'].transform('sum')
    final_df['Proportion (no wild-type)'] = final_df['Proportion (no wild-type)'] / df_grouped
    final_df['Proportion (no wild-type)'] = final_df['Proportion (no wild-type)'].fillna(0)
    
    pearson_corr = final_df['ABE Proportion'].corr(final_df['Proportion (no wild-type)'])
    print(f'{name}_ABE_proportion Pearson correlation:{pearson_corr}')
    
    spearman_corr = final_df['ABE Proportion'].corr(final_df['Proportion (no wild-type)'], method='spearman')
    print(f'{name}_ABE_proportion Spearman correlation:{spearman_corr}')
    return final_df





