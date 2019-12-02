import os
import time
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from utils.radam import RAdam

def save_checkpoint(checkpoint_file, net, epoch, optim, is_parallel=True):
    checkpoint_dict = {
        'epoch': epoch,
        'optimizer': optim.state_dict(),
        'state_dict': net.module.state_dict() if is_parallel else net.state_dict()
    }
    torch.save(checkpoint_dict, checkpoint_file)

def load_checkpoint(checkpoint_file, is_paraller=True):
    checkpoint = torch.load(checkpoint_file)
    if is_paraller:
        w_dict = checkpoint['state_dict']
        w_dict = {'module.' + k: v for k, v in w_dict.items()}
    else:
        w_dict = checkpoint['state_dict']

    return checkpoint['epoch'], w_dict, checkpoint['optimizer']

def temporal_boundary_to_residual(proposal_bbox, gt_bbox):
    t = proposal_bbox[:, -1] - proposal_bbox[:, 0] + 1
    t = t / 2.0
    fa = (proposal_bbox[:, -1] + proposal_bbox[:, 0]) / 2.0
    res = (gt_bbox - fa.unsqueeze(1)) / t.unsqueeze(1)

    return res

def residual_to_temporal_boundary(proposal_bbox, pre_res):
    t = proposal_bbox[:, -1] - proposal_bbox[:, 0] + 1
    t = t / 2.0
    fa = (proposal_bbox[:, -1] + proposal_bbox[:, 0]) / 2.0
    pre_loc = pre_res * t.unsqueeze(1).float() + fa.unsqueeze(1).float()

    return pre_loc

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'radam':
        optimizer = RAdam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer

# from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    summary = []
    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details

# from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

# TODO: compute accuracy for heatmaps
def calc_accuracy(heatmap, target):
    acc = 0
    batch_size = heatmap.size(0)

    for idx in range(batch_size):
        pred = heatmap[idx, ...]
    gt = target.sum(dim=1)

    return acc

if __name__ == '__main__':
    pass
