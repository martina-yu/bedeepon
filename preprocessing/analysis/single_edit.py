def single_edit(file_path):
    df = pd.read_csv(file_path,sep=',')
    df['efficiency'] = df['efficiency'] * 100
    df['edited_sum'] = 0
    for i in range(len(df)):
        for j in range(len(df.iloc[i]['gRNA'])):
            if df.iloc[i]['gRNA'][j]!= df.iloc[i]['target'][j]:
                df.at[i, 'edited_sum'] += 1
    df.to_csv(file_path[:-4] + '_site.csv', index=False)

if __name__ == '__main__':
    file_path = './get_plots/.csv'
    single_edit(file_path)
    
df = pd.read_csv('./.csv',sep=',')
df = df[df['edited_sum'] == 1]
filtered_dataframes = []
for i in range(4, 10):
    filtered_data = df[df['gRNA'].str[i - 1] != df['target'].str[i - 1]]
    filtered_data = filtered_data.copy()
    filtered_data['site'] = i
    filtered_dataframes.append(filtered_data)
all_filtered_data = pd.concat(filtered_dataframes)