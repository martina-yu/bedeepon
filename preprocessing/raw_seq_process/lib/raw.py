if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Base edit analysis.')
    parser.add_argument('-f', '--fastq-path', help='Fastq path.')
    parser.add_argument('-n', '--sample-name', help='Sample name.')
    parser.add_argument('-o', '--output-path', help='Output path.')
    parser.add_argument('-t', '--editor-type', help='Editor type.', 
        choices=['Off', 'On'])
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    prefix = f'{args.output_path}/{args.sample_name}'
    read1, read2 = preprocess.get_read_path(args.fastq_path)
    fastp_out = preprocess.do_fastp(read1, read2, prefix)
    seqtk_out = preprocess.do_seqtk(fastp_out, prefix)
    dedup_out = preprocess.do_dedup(seqtk_out, prefix)
    
    if args.editor_type == 'Off':
        if args.sample_name.startswith('A'):
            ref = '/lustre/wangyongming/ForZCD/BECode/data/AG_Lib_Final.txt'
        elif args.sample_name.startswith('C'):
            ref = '/lustre/wangyongming/ForZCD/BECode/data/CT_Lib_Final.txt'
        usecols = ['gRNA', 'True_gRNA', 'type', 'barcode2', 'target']
        ref_df = pd.read_csv(ref, sep='\t', usecols=usecols)
        ref_df.rename(columns={'target': 'source', 'type': 'offtype'}, 
            inplace=True)
        ref_df['gRNA'] = ref_df['gRNA'].str.upper() + 'GG'
        ref_df['True_gRNA'] = ref_df['True_gRNA'].str.upper()
        ref_df['barcode2'] = ref_df['barcode2'].str.upper()
        ref_df['source'] = ref_df['source'].str.upper()
    else:
        ref = '/lustre/wangyongming/ForZCD/BECode/data/OT_Lib_Final.txt'
        usecols = ['gRNA_seq', 'prefix', 'suffix', 'target_gRNA_PAM']
        ref_df = pd.read_csv(ref, sep='\t', usecols=usecols)
        columns = {'target_gRNA_PAM': 'gRNA'}
        ref_df.rename(columns=columns, inplace=True)
        ref_df['prefix'] = ref_df['prefix'].str.upper()
        ref_df['suffix'] = ref_df['suffix'].str.upper()

    pandarallel.initialize(nb_workers=20)
    df_list = []
    with pd.read_csv(dedup_out, names=['info'], chunksize=1e6) as reader:
        for df in reader:
            df['info'] = df.parallel_apply(lambda x: x['info'].strip(), axis=1)
            df[['count', 'read']] = df.parallel_apply(
                lambda x: x['info'].split(' '), axis=1, result_type='expand')
            df.drop(columns='info', inplace=True)
            df['count'] = df['count'].astype(int)

            if args.editor_type == 'Off':
                df[['True_gRNA', 'barcode2', 'target']] = df.parallel_apply(
                    lambda x: cutadapter.off_read_info(x['read']), axis=1, 
                    result_type='expand')
                df.dropna(inplace=True)
                if not len(df):
                    continue
                df = pd.merge(df, ref_df, on=['True_gRNA', 'barcode2'])
                if not len(df):
                    continue
                df.drop(columns=['True_gRNA', 'read'], inplace=True)
                df['is_edit'] = df.parallel_apply(
                    lambda x: editcheck.off_check(x['target'], x['source']), 
                    axis=1)
                df = df[df['is_edit'] != -1]
                df.reset_index(drop=True, inplace=True)
                if not len(df):
                    continue
            else:
                df['gRNA_seq'] = df.parallel_apply(
                    lambda x: cutadapter.on_read_info(x['read']), axis=1)
                df.dropna(inplace=True)
                df = pd.merge(df, ref_df, on='gRNA_seq')
                if not len(df):
                    continue
                df['target'] = df.parallel_apply(
                    lambda x: cutadapter.on_read_info(x['read'], x['prefix'], 
                    x['suffix']), axis=1)
                df.dropna(inplace=True)
                if not len(df):
                    continue
                df['tgtlen'] = df.parallel_apply(
                    lambda x: len(x['target']), axis=1)
                df = df[df['tgtlen'] == 23].reset_index(drop=True)
                columns = ['gRNA_seq', 'prefix', 'suffix', 'tgtlen', 'read']
                df.drop(columns=columns, inplace=True)
                if not len(df):
                    continue
                df[['is_mis_synthesis', 'is_edit']] = df.parallel_apply(
                    lambda x: editcheck.on_check(x['target'], x['gRNA']), 
                    axis=1, result_type='expand')
                df = df[(df['is_edit'] != -1) & (df['is_mis_synthesis'] == 0)]
                df.reset_index(drop=True, inplace=True)
                df.drop(columns='is_mis_synthesis', inplace=True)
                if not len(df):
                    continue
            df_list.append(df)
    df = pd.concat(df_list)
    df.to_csv(f'{prefix}.info', sep='\t', index=False)

    if args.editor_type == 'Off':
        df = pd.pivot_table(df, values='count', index=['gRNA', 'source', 
            'offtype', 'barcode2'], columns='is_edit', aggfunc=np.sum)
        df.reset_index(inplace=True)
        df.fillna(0, inplace=True)
        df['count'] = df[0] + df[1]
        df['efficiency'] = df[1] / df['count']
        df['count'] = df['count'].astype(int)
        df = df[['gRNA', 'source', 'offtype', 'barcode2', 'count', 'efficiency']]
        df.to_csv(f'{prefix}.eff', sep='\t', index=False, float_format='%.6f')
    else:
        df = pd.read_csv(f'{prefix}.info', sep='\t')
        df1 = pd.pivot_table(df, values='count', index=['gRNA', 'target'], 
            columns='is_edit', aggfunc=np.sum)
        df2 = pd.pivot_table(df, values='count', index='gRNA', 
            columns='is_edit', aggfunc=np.sum)
        df1.reset_index(inplace=True)
        df1.fillna(0, inplace=True)
        df2.reset_index(inplace=True)
        df2.fillna(0, inplace=True)
        df2['count'] = df2[0] + df2[1]
        df2['efficiency'] = df2[1] / df2['count']
        df2['count'] = df2['count'].astype(int)
        df2 = df2[['gRNA', 'count', 'efficiency']]
        df2.to_csv(f'{prefix}.eff2', sep='\t', index=False, float_format='%.6f')
        df1 = pd.merge(df1, df2, on='gRNA')
        df1.drop_duplicates(inplace=True)
        df1 = df1[df1[1] > 0]
        df1['efficiency'] = df1[1] / df1['count']
        df1 = df1[['gRNA', 'target', 'count', 'efficiency']]
        df1.to_csv(f'{prefix}.eff1', sep='\t', index=False, float_format='%.6f')
