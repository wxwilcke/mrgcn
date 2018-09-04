
#!/usr/bin/env python

import logging
import os
import random

from keras import backend as K
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

def get_config():
    config = tf.ConfigProto()
    config.CopyFrom(K.get_session()._config)

    return config

def set_config(config):
    sess = tf.Session(graph=tf.get_default_graph(), config=config)
    K.set_session(sess)

def update_config(update):
    config = get_config()
    config.MergeFrom(update)
    set_config(config)

def reload_config(reset_session=True):
    # https://github.com/keras-team/keras/issues/2102
    config = get_config()

    if reset_session:
        K.clear_session()

    set_config(config)
    logger.debug("Reloaded configuration")

def set_tensorflow_device_placement(mode='gpu'):
    config = get_config()
    config.ClearField('allow_soft_placement')

    if mode == 'gpu':
        update = tf.ConfigProto(allow_soft_placement=False)
        logger.debug("Setting device to GPU only")
    else:
        update = tf.ConfigProto(allow_soft_placement=True)
        logger.debug("Setting device to system default")

    config.MergeFrom(update)
    set_config(config)

def set_number_of_threads(n=1):
    # reproducability is not guaranteed if n > 1
    update = tf.ConfigProto(intra_op_parallelism_threads=n,
                            inter_op_parallelism_threads=n)

    update_config(update)
    logger.debug("Set number of threads to {}".format(n))

def set_seed(seed=-1):
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    if seed >= 0:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        logger.debug("Setting seed to {}".format(seed))
    else:
        logger.debug("Using random seed")
