def correlation(file_path1, file_path2):
    file_path1 = file_path1[file_path1['all_count'] >= 100]
    file_path2 = file_path2[file_path2['all_count'] >= 100]
    
    merged_BE = pd.merge(file_path1, file_path2, on=['gRNA','target'], how='outer')
    merged_BE.reset_index(drop=True, inplace=True) 
    merged_BE.dropna(inplace=True)
    spearman_correlation = spearmanr(merged_BE['outcome_efficiency_x'], merged_BE['outcome_efficiency_y'])[0]
    pearson_correlation = pearsonr(merged_BE['outcome_efficiency_x'], merged_BE['outcome_efficiency_y'])[0]

    print(f'The spearman correlation between file_path1 and file_path2 is: {spearman_correlation}')
    print(f'The pearson correlation between file_path1 and file_path2 is: {pearson_correlation}')
        
    plt.figure(figsize=(8, 6))
    sns.regplot(x='outcome_efficiency_x', y='outcome_efficiency_y', data=merged_BE, scatter_kws={'s': 3, 'alpha': 0.4},color='steelblue')
    sns.regplot(x='outcome_efficiency_x', y='outcome_efficiency_y', data=merged_BE, scatter=False, color='red', line_kws={'linestyle': '--'})
    plt.xlabel('Outcome Efficiency (x)')
    plt.ylabel('Outcome Efficiency (y)')
    plt.title('Hela ABE correlation\nSpearman: {:.2f}, Pearson: {:.2f}'.format(spearman_correlation, pearson_correlation))
    plt.grid(True)
    plt.tight_layout()
    plt.annotate(f'Spearman: {spearman_correlation:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='red')
    plt.annotate(f'Pearson: {pearson_correlation:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, color='red')
    
def combine_eff(file_path1, file_path2):
    df_both = pd.merge(file_path1, file_path2, on=['gRNA','target'], how='outer')
    df_both['count_x'] = df_both['count_x'].fillna(0)
    df_both['count_y'] = df_both['count_y'].fillna(0)
    df_both['all_count_x'] = df_both.groupby('gRNA')['count_x'].transform('sum')
    df_both['all_count_y'] = df_both.groupby('gRNA')['count_y'].transform('sum')
    df_both['count'] = df_both['count_x'] + df_both['count_y']
    df_both['all_count'] = df_both['all_count_x'] + df_both['all_count_y']
    df_both['efficiency'] = df_both['count'] / df_both['all_count']
    df_both = df_both[['gRNA','target','count','all_count','efficiency']]

    df1_only = file_path1.merge(file_path2, on='gRNA', how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    df2_only = file_path2.merge(file_path1, on='gRNA', how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    df1_only.fillna(0, inplace=True)
    df2_only.fillna(0, inplace=True)
    df1_only['count'] = df1_only['count_x'] + df1_only['count_y']
    df2_only['count'] = df2_only['count_x'] + df2_only['count_y']
    df1_only['all_count'] = df1_only['all_count_x'] + df1_only['all_count_y']    
    df2_only['all_count'] = df2_only['all_count_x'] + df2_only['all_count_y']
    df1_only['efficiency'] = df1_only['count_x'] / df1_only['all_count_x']
    df2_only['efficiency'] = df2_only['count_x'] / df2_only['all_count_x']
    df1_only.rename(columns={'target_x':'target'}, inplace=True)
    df2_only.rename(columns={'target_x':'target'}, inplace=True)
    df1_only = df1_only[['gRNA','target','count','all_count','efficiency']]
    df2_only = df2_only[['gRNA','target','count','all_count','efficiency']]
    df_only = pd.concat([df1_only, df2_only])
    merged = pd.concat([df_both, df_only])
    merged = merged[(merged['all_count'] >= 100)]
    merged = merged.drop_duplicates(subset=['gRNA','target'])
    merged.sort_values('gRNA', inplace=True)
    merged.to_csv('./get_plots/combine_Hela_ABE.csv',index=False, float_format='%.6g')   
    
if __name__ == '__main__':
    file_path1 = './get_plots/.csv'
    file_path2 = './get_plots/.csv'
    file_path1 = pd.read_csv(file_path1,sep=',')
    file_path2 = pd.read_csv(file_path2,sep=',')
    correlation(file_path1, file_path2)
    combine_eff(file_path1, file_path2)
