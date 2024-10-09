def do_fastp(read1, read2, prefix):
    '''Merge pair-end fastq.'''
    out = f'{prefix}.fq'
    cmd = f'fastp -w 16 -A -c -m -i {read1} -I {read2} --merged_out {out} \
            -j {prefix}.json -h {prefix}.html'
    check_call(cmd, shell=True)
    return out


def do_seqtk(merged_fq, prefix):
    '''Convert fastq to fastq.'''
    out = f'{prefix}.fa'
    cmd = f'seqtk seq -a Q 33 -q 10 -n N {merged_fq} > {out}'
    check_call(cmd, shell=True)
    return out


def get_read_path(fq_path):
    '''Get read1 and read2 path.'''
    read1, read2 = None, None
    for fq in glob.glob(f'{fq_path}/*.fastq.gz'):
        if 'R1' in fq:
            read1 = fq
        elif 'R2' in fq:
            read2 = fq
    return read1, read2


def on_read_info(read, prefix, suffix):
    '''Get read barcode and target sequence.'''
    parameters = {'max_error_rate': 0.2, 'min_overlap': 20,
                  'read_wildcards': False, 'adapter_wildcards': False,
                  'indels': False}
    front_adapter = FrontAdapter(prefix, **parameters)
    back_adapter = BackAdapter(suffix, **parameters)
    linked_adapter = LinkedAdapter(front_adapter, back_adapter,
                                   front_required=True, back_required=True,
                                   name='target_region_recognition')
    r = linked_adapter.match_to(read)
    if r:
        target_range = (r.front_match.rstop,
                        r.back_match.rstart + r.front_match.rstop)
        target = read[target_range[0]:target_range[1]]
    else:
        target = None
    return target


def on_check(source, target):
    '''Check read is edit or not.'''
    if len(target) != 23:
        return 0, -1
    parameters = {'max_error_rate': 0.2, 'min_overlap': 10,
                  'read_wildcards': False, 'adapter_wildcards': False,
                  'indels': False}
    adapter = FrontAdapter(target[10:20], **parameters)
    r = adapter.match_to(source[10:20])
    if r:
        is_mis_synthesis = 0
    else:
        is_mis_synthesis = 1
    if 'N' in target:
        is_edit = -1
    elif source[1:21] in target:
        is_edit = 0
    elif source[:-3] != target[:-3]:
        is_edit = 1
    else:
        is_edit = -1
    return is_mis_synthesis, is_edit


def parser_rname(rname):
    rname_list = rname.split('|')
    grna = rname_list[0]
    prefix = rname_list[1]
    suffix = rname_list[2]
    a_pos = rname_list[3]
    c_pos = rname_list[4]
    t_pos = rname_list[5]
    g_pos = rname_list[6]
    return grna, prefix, suffix, a_pos, c_pos, t_pos, g_pos


def canonical_check(source, target, is_abe):
    is_canonical = 1
    if source == target:
        return is_canonical
    if is_abe:
        for a, b in zip(source, target):
            if a == 'A':
                if b in ['C', 'T']:
                    is_canonical = 0
                    break
            else:
                if a != b:
                    is_canonical = 0
                    break
    else:
        for a, b in zip(source, target):
            if a == 'C':
                if b in ['A', 'G']:
                    is_canonical = 0
                    break
            else:
                if a != b:
                    is_canonical = 0
                    break
    return is_canonical


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Base edit analysis.')
    parser.add_argument('-f', '--fastq-path', help='Fastq path.')
    parser.add_argument('-n', '--sample-name', help='Sample name.')
    parser.add_argument('-o', '--output-path', help='Output path.')
    parser.add_argument('-r', '--reference', help='gRNA reference.')
    parser.add_argument('-l', '--library', help='gRNA library.')
    args = parser.parse_args()
    
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    prefix = f'{args.output_path}/{args.sample_name}'
    read1, read2 = get_read_path(args.fastq_path)
    fastp_out = do_fastp(read1, read2, prefix)
    seqtk_out = do_seqtk(fastp_out, prefix)
    
    
    usecols = ['gRNA_seq', 'target_gRNA_PAM']
    ref_df = pd.read_csv(args.library, sep='\t', usecols=usecols)
    ref_df.columns = ['grna', 'source']
    
    
    pandarallel.initialize(nb_workers=20)
    df_list = []
    with pd.read_csv(seqtk_out, names=['read'], chunksize=1e6) as reader:
        for df in reader:
            df_name = df[df.index % 2 == 0].copy().reset_index(drop=True)
            df_read = df[df.index % 2 == 1].copy().reset_index(drop=True)
            
                        
            df_name['read'] = df_name['read'].apply(
                lambda x: x.split(' ')[0][1:])
            read_df = pd.merge(df_name, df_read, left_index=True,
                               right_index=True)
            read_df.columns = ['qname', 'read']
            
            
            df_read['read'] = df_read['read'].str[27:47]
            df_read.dropna(inplace=True)
            if df_read.shape[0] == 0:
                continue
            df_name = df_name.loc[df_read.index, :]
            df_len = df_read.shape[0]
            df_name.index = [2 * i for i in range(df_len)]
            df_read.index = [2 * i + 1 for i in range(df_len)]
            df = pd.concat([df_name, df_read])
            df.sort_index(inplace=True)
            fa = f'{prefix}.gRNA.fa'
            
            df.to_csv(fa, index=False, header=None)
            ref = args.reference
            cmd = f'bwa aln -t 20 -n 0 -o 0 -l 19 -k 0 -d 0 -i 0 -O 10 -E 5 -N \
                {ref} {fa} | bwa samse {ref} - {fa} | samtools view -S |  \
                cut -f 1-3 > {prefix}.gRNA.sam'
            check_call(cmd, shell=True)
            
            cols = ['qname', 'flag', 'rname']
            aln_df = pd.read_csv(f'{prefix}.gRNA.sam', sep='\t', names=cols,
                                 skiprows=81331)
            aln_df = aln_df[aln_df['flag'] == 0]
            if aln_df.shape[0] == 0:
                continue
            aln_df['qname'] = '>' + aln_df['qname']
            read_df = read_df.merge(aln_df)
            
            cols = ['grna', 'prefix', 'suffix', 'a_pos', 'c_pos', 't_pos',
                    'g_pos']
            read_df[cols] = read_df.parallel_apply(lambda x: parser_rname(
                x['rname']), axis=1, result_type='expand')
            read_df = read_df.merge(ref_df)
            
            read_df['target'] = read_df.parallel_apply(
                lambda x: on_read_info(x['read'], x['prefix'], x['suffix']),
                axis=1)
            read_df.dropna(inplace=True)
            if read_df.shape[0] == 0:
                continue
            read_df['tgt_len'] = read_df.parallel_apply(
                lambda x: len(x['target']), axis=1)
            read_df = read_df[['source', 'target', 'tgt_len']]
            read_df[['is_mis_synthesis', 'is_edit']] = read_df.parallel_apply(
                lambda x: on_check(x['source'], x['target']), axis=1,
                result_type='expand')
            df_list.append(read_df)
    df = pd.concat(df_list)
    df.to_csv(f'{prefix}.info', sep='\t', index=False)

    is_abe = True if 'ABE' in args.sample_name else False
    df_list = []
    for df in pd.read_csv(f'{prefix}.info', sep='\t', chunksize=1e6):
        df = df[(df['tgt_len'] == 23) & (df['is_mis_synthesis'] == 0) 
                & (df['is_edit'] != -1)]
        if df.shape[0] == 0:
            continue
        df['source'] = df['source'].str[:-3]
        df['target'] = df['target'].str[:-3]
        df['count'] = 1
        tab = pd.pivot_table(df, index=['source', 'target'], values='count', 
                             aggfunc='sum')
        tab.reset_index(inplace=True)
        tab['is_canonical'] = tab.parallel_apply(
            lambda x: canonical_check(x['source'], x['target'], is_abe), axis=1)
        tab = tab[tab['is_canonical'] == 1]
        if tab.shape[0] == 0:
            continue
        df_list.append(tab)
    df = pd.concat(df_list)
    tab = pd.pivot_table(df, index=['source', 'target'], values='count', 
                         aggfunc='sum')
    tab.reset_index(inplace=True)
    tab.columns = ['Reference', 'Outcome', 'Outcome_Count']
    tab['Total_Count'] = tab.groupby('Reference')['Outcome_Count'].transform('sum')
    tab['Proportion'] = tab['Outcome_Count'] / tab['Total_Count']
     # unedited
    un_df = tab[tab['Reference'] == tab['Outcome']].copy()
    un_df['Overall_Efficiency'] = 1 - un_df['Proportion']
    un_df = un_df[['Reference', 'Overall_Efficiency']]
    # 3-10 editied
    ed_df = tab[tab['Reference'].str[2:10] != tab['Outcome'].str[2:10]].copy()
    df_list = []
    for ref, sub_df in ed_df.groupby('Reference', sort=False):
        total_count = sub_df['Outcome_Count'].sum()
        sub_df['Outcome_Proportion'] = sub_df['Outcome_Count'] / total_count
        df_list.append(sub_df)
    ed_df = pd.concat(df_list)
    ed_df.drop(columns=['Proportion', 'Outcome_Count', 'Total_Count'], 
               inplace=True)
    df = ed_df.merge(un_df, on=['Reference'], how='outer')
    df['Outcome'] = df['Outcome'].fillna('-')
    df['Outcome_Proportion'] = df['Outcome_Proportion'].fillna(0)
    df.to_csv(f'{prefix}.eff', index=False, float_format='%.6g')
