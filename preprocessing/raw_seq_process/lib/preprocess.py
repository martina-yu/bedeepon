def do_fastp(read1, read2, prefix):
    '''Merge pair-end fastq.'''
    fastp = '/lustre/path/fastp'
    out = f'{prefix}.fq'
    cmd = f'{fastp} -w 16 -A -c -m -i {read1} -I {read2} --merged_out {out}'
    # check_call(cmd, shell=True)
    return out


def do_seqtk(merged_fq, prefix):
    '''Convert fastq to fastq.'''
    seqtk = '/lustre/path/miniconda3/bin/seqtk'
    out = f'{prefix}.fa'
    cmd = f'{seqtk} seq -a -Q 33 -q 10 -n N {merged_fq} > {out}'
    # check_call(cmd, shell=True)
    return out


def do_dedup(fasta, prefix):
    '''Remove duplicate reads and count.'''
    out = f'{prefix}.read'
    cmd = f'grep -v ">" {fasta} | sort --parallel=20 | uniq -c > {out}'
    # check_call(cmd, shell=True)
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
