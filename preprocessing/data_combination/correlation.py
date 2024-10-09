# upload files
def correlation(file_path1, file_path2):
    file_path1 = pd.read_csv(file_path1,sep=',')
    file_path2 = pd.read_csv(file_path2,sep=',')
    # drop out rows in file_path1&2 with overall_efficiency == 0
    file_path1 = file_path1[(file_path1['overall_efficiency']!= 0) & (file_path1['all_count'] >= 100)]
    file_path2 = file_path2[(file_path2['overall_efficiency']!= 0) & (file_path2['all_count'] >= 100)]
    merged_ABE = pd.merge(file_path1, file_path2, on=['gRNA','target'], how='outer')
    merged_ABE.reset_index(drop=True, inplace=True) 
    merged_ABE.dropna(inplace=True)
    spearman_correlation = spearmanr(merged_ABE['overall_efficiency_x'], merged_ABE['overall_efficiency_y'])[0]
    paerson_correlation = pearsonr(merged_ABE['overall_efficiency_x'], merged_ABE['overall_efficiency_y'])[0]

    print(f'The spearman correlation between file_path1 and file_path2 is: {spearman_correlation}')
    print(f'The pearson correlation between file_path1 and file_path2 is: {paerson_correlation}')
    
def combine_eff(file_path1, file_path2):
    merge = pd.merge(file_path1, file_path2, on = ['gRNA','target'], how='outer')
    merge.fillna(0, inplace=True)
    merge['count'] = merge['count_x'] + merge['count_y']
    merge['all_count'] = merge['all_count_x'] + merge['all_count_y']
    merge['efficiency'] = merge['count'] / merge['all_count']
    merge = merge.iloc[:,[0,1,-1]]

    edited = merge[merge['gRNA'] != merge['target']].copy()
    edited['proportion'] = edited.groupby('gRNA')['efficiency'].transform('sum')    

correlation(file_path1, file_path2)
combine_eff(file_path1, file_path2)

