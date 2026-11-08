import glob
import json
import logging
import os

from src.baselines.ppga.compat import AttrDict

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s][%(process)05d] %(message)s")
ch.setFormatter(formatter)

fh = None

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)


def set_file_handler(logdir):
    global fh
    if fh is not None:
        log.removeHandler(fh)
    filepath = os.path.join(logdir, 'log.txt')
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def save_cfg(dir, cfg):
    def to_dict(cfg):
        if isinstance(cfg, AttrDict):
            cfg = dict(cfg)
    filename = 'cfg.json'
    fp = os.path.join(dir, filename)
    with open(fp, 'w') as f:
        json.dump(cfg, f, default=to_dict, indent=4)


def get_checkpoints(checkpoints_dir):
    checkpoints = glob.glob(os.path.join(checkpoints_dir, 'cp_*'))
    return sorted(checkpoints)
