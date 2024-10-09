def on_check(target, source):
    '''Check read is edit or not.'''
    parameters= {'max_error_rate': 0.2, 'min_overlap': 10, 
                 'read_wildcards': False, 'adapter_wildcards': False, 
                 'indels': False}
    adapter = FrontAdapter(source[10:20], **parameters)
    r = adapter.match_to(target[10:20])
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
