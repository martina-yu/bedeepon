def do_alignment(df, ref, prefix):
    '''Use BWA to align gRNA.'''
    gRNA_df = df[['gRNA']].copy()
    gRNA_df.drop_duplicates(inplace=True)
    gRNA_df.reset_index(inplace=True, drop=True)
    gRNA_df['name'] = '>' + gRNA_df.index.astype(str)
    gRNA_df.index = range(1 , len(gRNA_df) * 2, 2)
    name_df = gRNA_df[['name']]
    gRNA_df.drop(columns='name', inplace=True)
    gRNA_df.rename(columns={'gRNA': 'name'}, inplace=True)
    name_df.index = range(0 , len(gRNA_df) * 2, 2)
    r_df = pd.concat([name_df, gRNA_df])
    r_df.sort_index(inplace=True)
    r_df.to_csv(f'{prefix}.fa', sep='\t', index=False, header=None)
    cmd = f'{bwa} aln -n 0 -o 0 -l 19 -k 0 -d 0 -i 0 -O 10 -E 5 -N -t 20 ' \
        + f'{ref} {prefix}.fa | {bwa} samse {ref} - {prefix}.fa | ' \
        + f'{samtools} view ' + '| awk \' BEGIN {OFS=":"} {print $3,$10}\''
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, 
        executable='/bin/bash')
    out, _err = p.communicate()
    if len(out): 
        aln_df = pd.read_csv(BytesIO(out), sep=':', names=['ref', 'gRNA'])
        aln_df = aln_df[aln_df['ref'] != '*'].reset_index(drop=True)
    else:
        aln_df = pd.DataFrame(columns=['ref', 'gRNA'])
    df = pd.merge(df, aln_df, on='gRNA')
    return df
