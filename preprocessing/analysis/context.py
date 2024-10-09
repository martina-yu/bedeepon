def context_plot(file_path):
    df = pd.read_csv(file_path,sep=',')
    df = df[df['edited_sum'] == 1]
    filtered_dataframes = []
    for i in range(4, 10):
        filtered_data = df[df['gRNA'].str[i - 1] != df['target'].str[i - 1]]
        filtered_data = filtered_data.copy()
        filtered_data['site'] = i
        filtered_dataframes.append(filtered_data)
    all_filtered_data = pd.concat(filtered_dataframes)

    site_data_list = []
    for i in range(4, 11):
        site_data = all_filtered_data[all_filtered_data['site'] == i].copy()
        site_data['context'] = site_data['gRNA'].apply(lambda x: x[i - 2:i + 1])
        site_data['mean'] = site_data.groupby('context')['efficiency'].transform('mean')
        site_data_list.append(site_data)
    site_data_list = pd.concat(site_data_list)
        
    site_data_list = site_data_list.copy()
    site_data_list['con_mean'] = site_data_list.groupby('context')['mean'].transform('mean')
    site_df = site_data_list[['site','context','mean','con_mean']]
    site_df = site_df.drop_duplicates(subset='mean')

    con_df = site_df.drop_duplicates(subset='context')
    df_sorted = con_df.sort_values(by='con_mean')
    context_list = df_sorted['context'].to_list()

    sites = site_df['site'].unique()
    fig, ax = plt.subplots(figsize=(20, 8))

    for i in sites:
        df = site_df[site_df['site'] == i].copy()
        context_map = {context: index for index, context in enumerate(context_list)}
        df['context_order'] = df['context'].map(context_map)
        df = df.sort_values('context_order')
        ax.plot(df['context'], df['mean'], marker='o',markersize=15, label=f'Site {i}',linewidth=4)

    ax.set_xlabel('Context', fontsize=18)
    ax.set_ylabel('Mean Value',fontsize=18)
    ax.set_title('Mean Values of Different Contexts for HEK293T ABE', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xticks(range(len(context_list)))  
    ax.set_xticklabels(context_list, rotation=45)  
    ax.legend(title='Site', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
if __name__ == '__main__':
    file_path = './get_plots/.csv'
    context_plot(file_path)