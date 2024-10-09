def off_read_info(read):
    '''Get read barcode and target sequence.'''
    gRNA = None
    barcode = None
    target = None
    designed_prefix = 'AAGGACGAAACACC'
    designed_suffix = 'GTTTTAGAGCTAGAAATAG'
    parameters= {'max_error_rate': 0.2, 'min_overlap': 33, 
                 'read_wildcards': False, 'adapter_wildcards': False, 
                 'indels': False}
    front_adapter = FrontAdapter(designed_prefix, **parameters)
    back_adapter = BackAdapter(designed_suffix, **parameters)
    linked_adapter = LinkedAdapter(front_adapter, back_adapter, 
                                   front_required=True, back_required=True, 
                                   name='target_region_recognition')
    r = linked_adapter.match_to(read)
    if r:
        suffix_range = (r.back_match.rstart + r.front_match.rstop,  
                        r.back_match.rstop + r.front_match.rstop, 
                        r.back_match.errors)
        target_range = (r.front_match.rstop, 
                        r.back_match.rstart + r.front_match.rstop)
        gRNA = read[target_range[0]:target_range[1]]
        barcode_start = suffix_range[0] + 82 + 8
        barcode_end = barcode_start + 4 + 15 + 5
        barcode_region = read[barcode_start:barcode_end]
        if barcode_region[:4] == 'CTCC' and barcode_region[-5:] == 'GTACT':
            barcode = read[barcode_start + 4:barcode_end - 5]
            target_start = suffix_range[0] + 82 + 8 + 4 + 15
            target_end = suffix_range[0] + 82 + 8 + 4 + 15 + 5 + 25 + 20
            target_region = read[target_start:target_end]
            designed_prefix = 'GTACT'
            designed_suffix = 'CTTGGCGTAACTAGATCT'
            parameters= {'max_error_rate': 0.2, 'min_overlap': 23, 
                        'read_wildcards': False, 'adapter_wildcards': False, 
                        'indels': False}
            front_adapter = FrontAdapter(designed_prefix, **parameters)
            back_adapter = BackAdapter(designed_suffix, **parameters)
            linked_adapter = LinkedAdapter(front_adapter, back_adapter, 
                                           front_required=True, 
                                           back_required=True, 
                                           name='target_region_recognition')
            r = linked_adapter.match_to(target_region)
            if r:
                target_range = (r.front_match.rstop, 
                                r.back_match.rstart + r.front_match.rstop)
                target = target_region[target_range[0]:target_range[1]]    
    return gRNA, barcode, target

def on_read_info(read, prefix='AGCCTTGTTT', suffix='GTTTTAGAGC'):
    '''Get read barcode and target sequence.'''
    parameters= {'max_error_rate': 0.2, 'min_overlap': 20, 
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
