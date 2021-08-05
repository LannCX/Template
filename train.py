import os
import pprint
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn

import init_path
from utils.logger import Logger
from scheduler.virat_model import ViratModel as Model
from config.virat_cfg import ViratConfig as Config
from data.data_loader import customer_data_loader

from networks.resnet_3d import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', default='./experiments/demo_config.yaml',
                        help='experiment configure file name',
                        # required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify cfg options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config(args).getcfg()
    exp_suffix = os.path.basename(args.cfg).split('.')[0]
    logger = Logger(os.path.join(cfg.LOG_DIR, '_'.join([cfg.MODEL.NAME, exp_suffix, cfg.TRAIN.OPTIMIZER, str(cfg.TRAIN.LR)])))

    logger.log(pprint.pformat(args))
    logger.log(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # define
    train_loader = customer_data_loader(cfg, 'train')
    val_loader = customer_data_loader(cfg, 'validation')
    net = eval('networks.'+cfg.MODEL.NAME+'.get_net')(cfg, is_train=cfg.IS_TRAIN)
    model = Model(net, cfg=cfg, logger=logger, suffix=exp_suffix, loss_type='KL')

    if cfg.IS_TRAIN:
        model.train(train_loader, val_loader=val_loader)
    else:
        model.test(train_loader, epoch=1, is_val=False)


if __name__ == '__main__':
    main()
