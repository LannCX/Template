import os
import gc
import time
import math
import inspect
import datetime
import numpy as np
from collections import namedtuple


import torch
import torch.nn as nn
import torch.optim as optim
from utils.radam import RAdam, AdamW


def save_checkpoint(checkpoint_file, net, epoch, optim, gs, is_parallel=True):
    checkpoint_dict = {
        'epoch': epoch,
        'global_step': gs,
        'optimizer': optim.state_dict(),
        'state_dict': net.module.state_dict() if is_parallel else net.state_dict()
    }
    torch.save(checkpoint_dict, checkpoint_file)


def load_checkpoint(checkpoint_file, is_parallel=True):
    checkpoint = torch.load(checkpoint_file)
    if is_parallel:
        w_dict = checkpoint['state_dict']
        w_dict = {'module.' + k: v for k, v in w_dict.items()}
    else:
        w_dict = checkpoint['state_dict']
        # w_dict = {k.replace('module.',''):v for k,v in mdl.items()}
    return w_dict, checkpoint


def get_optimizer(cfg, model, policies=None):
    if policies is None:
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                  'initial_lr': cfg.TRAIN.LR}],
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                  'initial_lr': cfg.TRAIN.LR}],
                lr=cfg.TRAIN.LR
            )
        else:
            raise(KeyError, '%s not supported yet...'%cfg.TRAIN.OPTIMIZER)
    else:
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                policies,
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                policies,
                lr=cfg.TRAIN.LR
            )
        elif cfg.TRAIN.OPTIMIZER == 'radam':
            optimizer = RAdam(
                policies,
                lr=cfg.TRAIN.LR
            )
        else:
            raise(KeyError, '%s not supported yet...'%cfg.TRAIN.OPTIMIZER)

    return optimizer


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
