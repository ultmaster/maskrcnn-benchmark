# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
from datetime import datetime


def setup_logger(name, save_dir, distributed_rank, filename=None):
    if filename is None:
        filename = "logs/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
