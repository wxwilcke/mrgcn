
#!/usr/bin/env python

import logging
import os
import random

import numpy as np
import torch


logger = logging.getLogger(__name__)

def set_seed(seed=-1):
    if seed >= 0:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        logger.debug("Setting seed to {}".format(seed))
    else:
        logger.debug("Using random seed")
