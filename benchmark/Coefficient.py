test_data = pd.read_csv(f'{file_path}.csv',index_col=0)

all_sprm, all_prs, sprm_df, prs_df = correlation(test_data)

print('Pearson',prs_df['Pearson_Correlation'].mean())
print('Spearman', sprm_df['Spearman_Correlation'].mean())

merged_df = pd.merge(prs_df, sprm_df, left_index=True, right_index=True, suffixes=('_Pearson', '_Spearman'))
plot_df = pd.melt(merged_df, value_vars=['Pearson_Correlation', 'Spearman_Correlation'], 
                  var_name='Type', value_name='Correlation')
sns.violinplot(x='Type', y='Correlation', data=plot_df, ax=axes, palette=palette)
pearson_mean = merged_df['Pearson_Correlation'].mean()
pearson_std = merged_df['Pearson_Correlation'].std()
spearman_mean = merged_df['Spearman_Correlation'].mean()
spearman_std = merged_df['Spearman_Correlation'].std()
axes.text(0, 1, f'{pearson_mean:.2f} ± {pearson_std:.3f}', ha='center', va='bottom', transform=axes.transAxes, fontsize=12)
axes.text(1, 1, f'{spearman_mean:.2f} ± {spearman_std:.3f}', ha='center', va='bottom', transform=axes.transAxes, fontsize=12)
plt.show()