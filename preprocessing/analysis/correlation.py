def correlation(file_path1, file_path2):
    file_path1 = file_path1[(file_path1['outcome_efficiency'] != 0) & (file_path1['all_count'] >= 100)]
    file_path2 = file_path2[(file_path2['outcome_efficiency'] != 0) & (file_path2['all_count'] >= 100)]
    
    merged_ABE = pd.merge(file_path1, file_path2, on=['gRNA', 'target'], how='outer')
    merged_ABE.reset_index(drop=True, inplace=True) 
    merged_ABE.dropna(inplace=True)
    
    spearman_correlation = spearmanr(merged_ABE['outcome_efficiency_x'], merged_ABE['outcome_efficiency_y'])[0]
    pearson_correlation = pearsonr(merged_ABE['outcome_efficiency_x'], merged_ABE['outcome_efficiency_y'])[0]
    
    plt.figure(figsize=(10, 6))
    
    sns.regplot(x='outcome_efficiency_x', y='outcome_efficiency_y', data=merged_ABE, ci=95, line_kws={'color': 'blue'}, scatter=False)
    
    scatter = plt.scatter(merged_ABE['outcome_efficiency_x'], merged_ABE['outcome_efficiency_y'], 
                          c=merged_ABE['outcome_efficiency_y'], s=60, cmap='plasma')
    
    plt.colorbar(scatter, label='Outcomes Efficiency')

    plt.text(0.1, 0.9, f'Pearson r = {pearson_correlation:.2f}\nSpearman r = {spearman_correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, color='black')

    plt.xlim([merged_ABE['outcome_efficiency_x'].min() - 0.1, merged_ABE['outcome_efficiency_x'].max() + 0.1])
    plt.ylim([merged_ABE['outcome_efficiency_y'].min() - 0.1, merged_ABE['outcome_efficiency_y'].max() + 0.1])
    plt.xlabel(f'{celltype} {base} Replicates Correlation')
    plt.ylabel('')
    
    print(f'The Spearman correlation between file_path1 and file_path2 is: {spearman_correlation}')
    print(f'The Pearson correlation between file_path1 and file_path2 is: {pearson_correlation}')
    
    plt.show()