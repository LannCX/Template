from config.base_cfg import BaseConfig
from yacs.config import CfgNode as CN


class ViratConfig(BaseConfig):
    def __init__(self, args=None):
        super(ViratConfig, self).__init__(args=args)

        self.cfg.IN_TYPE = 'rgb'  # rgb, flow or fuse
        self.cfg.PAR = True  # if use parallel gpu

        if args is not None:
            self.update_config(args)
