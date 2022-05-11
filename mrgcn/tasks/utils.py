#!/usr/bin/env python

import logging

logger = logging.getLogger(__name__)

def optimizer_params(model, optim_config):
    opt_params = list()
    for module_name_numbered, module in model.module_dict.items():
        module_name = module_name_numbered.split('_')[0]
        module_params = {'params': module.parameters()}

        if module_name == 'RGCN':
            # use default lr
            opt_params.append(module_params)
            continue

        for datatype, params in optim_config:
            if datatype in ['xsd.boolean', 'xsd.numeric']:
                mod_name = 'FC'
            elif datatype in ['xsd.date', 'xsd.dateTime', 'xsd.gYear']:
                mod_name = 'FC'
            elif datatype in ['xsd.string', 'xsd.anyURI']:
                mod_name = 'CharCNN'
            elif datatype in ['ogc.wktLiteral']:
                mod_name = 'GeomCNN'
            elif datatype in ['blob.image']:
                mod_name = 'ImageCNN'
            else:
                raise Exception('Unsupported datatype: %s' % datatype)

            if module_name == mod_name:
                module_params.update(params)
                break

        opt_params.append(module_params)

    return opt_params

#def mkbatches(mat, node_idx, num_batches=1):
#    """ split N x * array in batches
#    """
#    n = mat.shape[0]  # number of samples
#    num_batches = min(n, num_batches)
#    idc = np.arange(n, dtype=np.int32)
#
#    if num_batches <= 1:
#        logger.debug("Full batch mode")
#
#    idc_assignments = np.array_split(idc, num_batches)
#    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
#                        for slce in idc_assignments]
#
#    return list(zip(idc_assignments, node_assignments))
#
#def mkbatches_varlength(sequences, node_idx, seq_length_map,
#                        num_batches=1):
#    n = len(sequences)
#    num_batches = min(n, num_batches)
#    if num_batches <= 1:
#        logger.debug("Full batch mode")
#
#    # sort on length
#    idc = np.arange(n, dtype=np.int32)
#    _, sequences_sorted_idc = zip(*sorted(zip(seq_length_map, idc)))
#
#    seq_assignments = np.array_split(sequences_sorted_idc, num_batches)
#    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
#                        for slce in seq_assignments]
#
#    return list(zip(seq_assignments, node_assignments))

