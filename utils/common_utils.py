
import argparse
import logging
import torch
import numpy as np
import datetime
import time
from functools import wraps
from typing import Any, Callable
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class LRScheduler(object):

    def __init__(self, optimizer, warm_up):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.warm_up = warm_up

    def update(self, iteration, learning_rate, num_group=1000, ratio=1):
        if iteration < self.warm_up:
            learning_rate *= iteration / self.warm_up
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = learning_rate * ratio**i


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Times a function, usually used as decorator"""
    # ref: http://zyxue.github.io/2017/09/21/python-timeit-decorator.html
    @wraps(func)
    def timed_func(*args: Any, **kwargs: Any) -> Any:
        """Returns the timed function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
        print("time spent on %s: %s"%(func.__name__, elapsed_time))
        return result

    return timed_func


def get_offset_from_cat(offset2cat_pathname='data/synsetoffset2category.txt', cat=None):
    """
    given a class, return the offset
    """
    offset2cat = {}
    cat2offset = {}
    with open(offset2cat_pathname, 'r') as f:
        for line in f:
            ls = line.strip().split()
            offset2cat[ls[1]] = ls[0]
            cat2offset[ls[0]] = ls[1]

    assert(cat in cat2offset)
    return cat2offset[cat]

def get_cat_from_offset(offset2cat_pathname='data/synsetoffset2category.txt', offset=None):
    """
    given offset, return class
    """
    
    offset2cat = {}
    cat2offset = {}
    with open(offset2cat_pathname, 'r') as f:
        for line in f:
            ls = line.strip().split()
            offset2cat[ls[1]] = ls[0]
            cat2offset[ls[0]] = ls[1]
    
    assert(offset in offset2cat)
    return offset2cat[offset]

def cat2offset(cat):
    if cat == 'boat':
        cat = 'watercraft'
    if cat == 'dresser':
        cat = 'cabinet'
    if cat == 'sofa':
        cat = 'couch'
    cat_dict = {
        'car': '02958343',
        'chair': '03001627',
        'plane': '02691156',
        'table': '04379243',
        'lamp': '03636649',
        'couch':   '04256520', # sofa
        'watercraft': '04530566', # boat
        'cabinet': '02933112' # dresser
    }
    return cat_dict[cat]

def cat2idxoffset(cat):
    if cat == 'boat':
        cat = 'watercraft'
    if cat == 'dresser':
        cat = 'cabinet'
    if cat == 'sofa':
        cat = 'couch'
    cat_dict = {
        'car': 750,
        'chair': 0,
        'plane': 900,
        'table': 150,
        'lamp': 600,
        'couch':   300, # sofa
        'watercraft': 1050, # boat
        'cabinet': 450 # dresser
    }
    return cat_dict[cat]

class Val:
    def __init__(self, data_path):      
        pcn_train_pathname = os.path.join(data_path,'train.list')
        pcn_val_pathname = os.path.join(data_path,'valid.list')
        pcn_test_pathname = os.path.join(data_path,'test.list')
        pcn_train = open(pcn_train_pathname).readlines()
        pcn_val = open(pcn_val_pathname).readlines()
        pcn_test = open(pcn_test_pathname).readlines()
        # splited
        self.pcn_train = [itm.rstrip().split('/') for itm in pcn_train]
        self.pcn_val = [itm.rstrip().split('/') for itm in pcn_val]
        self.pcn_test = [itm.rstrip().split('/') for itm in pcn_test]
        