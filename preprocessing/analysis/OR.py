def get_odds_ratio(file_path):
    df = pd.read_csv(file_path,sep=',')
    df = df[df['edited_sum'] == 1]
    filtered_dataframes = []
    for i in range(4, 10):
        filtered_data = df[df['gRNA'].str[i - 1] != df['target'].str[i - 1]]
        filtered_data = filtered_data.copy()
        filtered_data['site'] = i
        filtered_dataframes.append(filtered_data)
    all_filtered_data = pd.concat(filtered_dataframes)
    df = all_filtered_data[['gRNA','target','efficiency','site']]
    df = df[(df['site'] >= 4) & (df['site']<= 7)]
    pos1_list = []
    pos2_list = []
    base_list = []
    odds_list = []
    for pos, sub_df in df.groupby('site', sort=False):
        cutoff1 = sub_df['efficiency'].quantile(0.20)
        cutoff2 = sub_df['efficiency'].quantile(0.80)
        df1 = sub_df[sub_df['efficiency'] <= cutoff1].copy()
        df2 = sub_df[sub_df['efficiency'] >= cutoff2].copy()
        df1 = pd.merge(df1, df1['gRNA'].str.split('', expand=True),left_index=True, right_index=True)
        df2 = pd.merge(df2, df2['gRNA'].str.split('', expand=True),left_index=True, right_index=True)
        df1.drop(columns=[0, 21], inplace=True) 
        df2.drop(columns=[0, 21], inplace=True) 
        for i in range (1, 21):
            base_list1 = df1[i].unique()
            base_list2 = df2[i].unique()
            if i == pos:
                continue 
            else:
                for base in ['A', 'C', 'G', 'T']:
                    cnt1 = df1[i].value_counts()[base] if base in base_list1 else 0
                    cnt2 = df1.shape[0] - cnt1
                    cnt3 = df2[i].value_counts()[base] if base in base_list2 else 0
                    cnt4 = df2.shape[0] - cnt3 
                    odds = None
                    if cnt1 != 0 and cnt3 != 0:
                        odds = np.log2((cnt3/cnt4)/(cnt1/cnt2))
                        
                    pos1_list.append(pos)
                    pos2_list.append(i)
                    base_list.append(base)
                    odds_list.append(odds)
    df = pd.DataFrame({'pos1': pos1_list, 'pos2': pos2_list,'base': base_list,'odds': odds_list})
    df.to_csv(file_path[:-4] +'_odds.csv', index=False)

def OR_plot(file_path):
    df = pd.read_csv(file_path,sep=',')
    
    pos4 = df[df['pos1'] == 4]
    pos5 = df[df['pos1'] == 5]
    pos6 = df[df['pos1'] == 6]
    pos7 = df[df['pos1'] == 7]
    
    for base in ['A', 'T', 'C', 'G']:
        base_empty_row = pd.DataFrame([[4, 4, base, np.nan]], columns=df.columns)
        pos4 = pd.concat([pos4, base_empty_row]).sort_values(by=['pos2']).reset_index(drop=True)
    
    for base in ['A', 'T', 'C', 'G']:
        base_empty_row = pd.DataFrame([[5, 5, base, np.nan]], columns=df.columns)
        pos5 = pd.concat([pos5, base_empty_row]).sort_values(by=['pos2']).reset_index(drop=True)
        
    for base in ['A', 'T', 'C', 'G']:
        base_empty_row = pd.DataFrame([[6, 6, base, np.nan]], columns=df.columns)
        pos6 = pd.concat([pos6, base_empty_row]).sort_values(by=['pos2']).reset_index(drop=True)
        
    for base in ['A', 'T', 'C', 'G']:
        base_empty_row = pd.DataFrame([[7, 7, base, np.nan]], columns=df.columns)
        pos7 = pd.concat([pos7, base_empty_row]).sort_values(by=['pos2']).reset_index(drop=True)
        
    matplotlib.style.use('seaborn-white') 
    fig, axes = plt.subplots(4, 4, figsize=(20,10))
    figure_labels = ['Adenine','Cytosine','Guanine','Thymine']
    colors = ['blue','orange','green','red']

    for i, base in enumerate(['A', 'C', 'G', 'T']):
        pos4_i = pos4[pos4['base'] == base]
        ax = axes[i][0]
        c = pos4_i['odds'].apply(lambda x: colors[i] if abs(x) >= np.log2(1.2) else 'gray').values
        pos4_i['odds'].plot(ax=ax, kind='bar', color=c, sharex=False, legend=False, rot=0)
        ax.set_title(f'Odds for {figure_labels[i]}')
        ax.set_xlabel('pos2')
        ax.set_ylabel('Odds')
        ax.set_xticks(np.arange(0, 21, 3)) 
        ax.set_yticks(np.arange(-6, 6, 2))
        ax.set_xticklabels(np.arange(1, 22, 3)) 
        
    for i, base in enumerate(['A', 'C', 'G', 'T']):
        pos5_i = pos5[pos5['base'] == base]
        ax = axes[i][1]
        c = pos5_i['odds'].apply(lambda x: colors[i] if abs(x) >= np.log2(1.2) else 'gray').values
        pos5_i['odds'].plot(ax=ax, kind='bar', color=c, sharex=False, legend=False, rot=0)
        ax.set_xticks(np.arange(0, 21, 3)) 
        ax.set_yticks(np.arange(-6, 6, 2))
        ax.set_xticklabels(np.arange(1, 22, 3)) 

    for i, base in enumerate(['A', 'C', 'G', 'T']):
        pos6_i = pos6[pos6['base'] == base]
        ax = axes[i][2]
        c = pos6_i['odds'].apply(lambda x: colors[i] if abs(x) >= np.log2(1.2) else 'gray').values
        pos6_i['odds'].plot(ax=ax, kind='bar', color=c, sharex=False, legend=False, rot=0)
        ax.set_xticks(np.arange(0, 21, 3)) 
        ax.set_yticks(np.arange(-6, 6, 2))
        ax.set_xticklabels(np.arange(1, 22, 3)) 

    for i, base in enumerate(['A', 'C', 'G', 'T']):
        pos7_i = pos7[pos7['base'] == base]
        ax = axes[i][3]
        c = pos7_i['odds'].apply(lambda x: colors[i] if abs(x) >= np.log2(1.2) else 'gray').values
        pos7_i['odds'].plot(ax=ax, kind='bar', color=c, sharex=False, legend=False, rot=0)
        ax.set_xticks(np.arange(0, 21, 3)) 
        ax.set_yticks(np.arange(-6, 6, 2))
        ax.set_xticklabels(np.arange(1, 22, 3)) 
    
    
if __name__ == '__main__':
    file_path1 = './get_plots/.csv'
    file_path2 = './get_plots/.csv'
    get_odds_ratio(file_path1)
    OR_plot(file_path2)