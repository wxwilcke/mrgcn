#!/usr/bin/python3

import argparse
import json
import logging
from os.path import expanduser

from data.utils import is_readable, is_writable


logger = logging.getLogger(__name__)

def set_keras_backend(conf_filename=expanduser('~')+'/.keras/keras.json', backend='tensorflow'):
    assert is_readable(conf_filename)
    assert is_writable(conf_filename)
    assert backend in ['tensorflow', 'theano']

    with open(conf_filename, 'r+') as f:
        jf = json.load(f)

        if 'backend' in jf:
            if jf['backend'] == backend:
                return

            logger.debug("Switching to {} Keras backend".format(backend.title()))
            jf['backend'] = backend

            f.seek(0)
            json.dump(jf, f)
            f.truncate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Keras configuration file (json)", required=True, default=None)
    parser.add_argument("backend", help="Keras backend", choices=['tensorflow',
                                                                  'theano'])
    args = parser.parse_args()
    set_keras_backend(args.config, args.backend)
    logging.shutdown()
