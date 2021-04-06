import logging
import os
import datetime
import random

from collections import defaultdict
import numpy as np
import torch
from torch.autograd import Variable



class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def logger_init(args):
    logging.basicConfig(level=logging.DEBUG, format='%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S')
    if args.log_to_file:
        log_filename = os.path.join(args.log_dir, args.log_prefix+datetime.datetime.now().strftime("%m%d%H%M%S"))
        logging.getLogget().addHandler(logging.FileHandler(log_filename))


def plot_config(args):
    out_str = "\noptim:{} lr:{} lamb:{}, d:{}, decay_rate:{}, batch_size:{}, shared_epoch:{}, controller_epoch:{}, derive_sample:{}\n".format(
            args.optim, args.lr, args.lamb, args.n_dim, args.decay_rate, args.n_batch, args.n_shared_epoch, args.n_controller_epoch, args.n_derive_sample)
    print(out_str)
    with open(args.perf_file, 'a') as f:
        f.write(out_str)


def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(random.randint(0, i))
    for ls in lists:
        j = idx[i]
        ls[i], ls[j] = ls[j], ls[i]

def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    for i in range(n_batch):
        start = int(n_sample * i / n_batch)
        end = int(n_sample * (i+1) / n_batch)
        ret = [ls[start:end] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    start = 0
    while(start < n_sample):
        end = min(n_sample, start + batch_size)
        ret = [ls[start:end] for ls in lists]
        start += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]
        
def gen_struct(num):
    struct = []
    for i in range(num):
        if i < 4:
            struct.append(random.randint(0,3))      #t
        else:
            struct.append(random.randint(0,3))      #h
            struct.append(random.randint(0,3))      #t
            struct.append(random.randint(-1,1))  #1
    return struct



