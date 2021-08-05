import os
from cfg import BaseConfig


class MainConfig(BaseConfig):
    def __init__(self, args=None):
        super(MainConfig, self).__init__(args=args)

        self.cfg.DATASET.ANNO_FILE=''
        self.cfg.DATASET.CLS_FILE = ''

        self.cfg.MODEL.USE_POINT = False
        self.cfg.MODEL.ENDPOINT = 'layer1'
        self.cfg.MODEL.POINT_RATIO = 0.3
        self.cfg.MODEL.POINT_NETWORK = 'GCN'

        self.cfg.DATASET.N_SEGMENT = 8

        if args is not None:
            self.update_config(args)
