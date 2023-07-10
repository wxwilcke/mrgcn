#!/usr/bin/env python

from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

def optimizer_params(model, optim_config):
    opt_params = [{"params": list()}, {"params": list()}]
    opt_params_index = {"default": 0, "gates": 1}

    for param_name, param in model.named_parameters(recurse=True):
        if not param.requires_grad:
            # filter parameters of frozen layers
            continue

        param_name_lst = param_name.split('.')
        if param_name_lst[0] != 'module_dict':
            # use default settings
            if param_name_lst[0] == "gate_weights":
                # change differently fron default optimizer
                opt_params[opt_params_index["gates"]]["params"].append(param)

                gate_weights = optim_config['gate_weights']
                opt_params[opt_params_index["gates"]].update(gate_weights)
            else:
                opt_params[opt_params_index["default"]]["params"].append(param)
            continue

        module_name = param_name_lst[1]
        datatype = '.'.join(module_name.split('_')[:2])
        if datatype not in opt_params_index.keys():
            i = len(opt_params)
            opt_params_index[datatype] = i
            opt_params.append({"params": list()})

        i = opt_params_index[datatype]
        opt_params[i]["params"].append(param)
        opt_params[i].update(optim_config[datatype])

    return opt_params

class EarlyStop:
    stop = None
    tolerance = -1
    patience = -1
    _patience_default = -1
    best_score = -1
    best_weights = None
    best_optim = None
    delay = -1

    def __init__(self, patience=7, tolerance=0.01, delay=10):
        self.tolerance = tolerance
        self.delay = delay
        self._patience_default = patience

        self.reset_counter()

    def record(self, score, weights, optim):
        if self.delay > 0:
            self.delay -= 1

            return

        if self.best_score < 0:
            self._update(score, weights, optim)

            return

        self.patience -= 1
        if (score + self.tolerance) < self.best_score:
            self._update(score, weights, optim)
            self.reset_counter()

        if self.patience <= 0:
            self.stop = True

    def _update(self, score, weights, optim):
        self.best_score = score
        self.best_weights = deepcopy(weights.state_dict())
        self.best_optim = deepcopy(optim.state_dict())

    def reset_counter(self):
        self.patience = self._patience_default
        self.stop = False

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

