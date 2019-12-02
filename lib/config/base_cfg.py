import os
from yacs.config import CfgNode as CN

class BaseConfig(object):
    def __init__(self, args=None):
        _C = CN()
        _C.GPUS = (0, 1)  # choose gpus
        _C.WORKERS = 4
        _C.SAVE_FREQ = 1
        _C.PRINT_FREQ = 20
        _C.USE_VAL = False
        _C.IS_TRAIN = False
        _C.PIN_MEMORY = True
        _C.AUTO_RESUME = True

        # path configs
        _C.DATA_DIR = ''  # root directory of all data(dataset, output, visulization, etc)
        _C.LOG_DIR = ''  # root directory of log files
        _C.OUTPUT_DIR = ''  # root directory of output data

        # Cudnn related params
        _C.CUDNN = CN()
        _C.CUDNN.BENCHMARK = True
        _C.CUDNN.DETERMINISTIC = False
        _C.CUDNN.ENABLED = True

        # common params for NETWORK
        _C.MODEL = CN()
        _C.MODEL.NAME = 'action_hrnet'
        _C.MODEL.INIT_WEIGHTS = True
        _C.MODEL.PRETRAINED = ''
        _C.MODEL.NUM_CLASSES = 18
        _C.MODEL.T_WIN = 64
        _C.MODEL.TARGET_TYPE = 'gaussian'
        _C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
        _C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
        _C.MODEL.SIGMA = 2
        _C.MODEL.EXTRA = CN(new_allowed=True)

        _C.LOSS = CN()
        _C.LOSS.USE_OHKM = False
        _C.LOSS.TOPK = 8
        _C.LOSS.USE_TARGET_WEIGHT = True
        _C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

        # DATASET related params
        _C.DATASET = CN()
        _C.DATASET.ROOT = ''
        _C.DATASET.NAME = ''  # name of dataset
        _C.DATASET.TRAIN_SET = 'train'  # directory name of training set
        _C.DATASET.TEST_SET = 'validation'  # directory name of validation set
        _C.DATASET.DATA_FORMAT = '.mp4'  # data format
        _C.DATASET.MODALITY = 'rgb'
        _C.DATASET.VERSION = ''

        # training data augmentation
        _C.DATASET.FLIP = False
        _C.DATASET.SCALE_FACTOR = 0.25
        _C.DATASET.ROT_FACTOR = 30
        _C.DATASET.COLOR_RGB = False

        # train
        _C.TRAIN = CN()
        _C.TRAIN.LR_FACTOR = 0.1
        _C.TRAIN.LR_STEP = [90, 110]
        _C.TRAIN.LR = 0.001

        _C.TRAIN.OPTIMIZER = 'adam'
        _C.TRAIN.MOMENTUM = 0.9
        _C.TRAIN.WD = 0.0001
        _C.TRAIN.NESTEROV = False
        _C.TRAIN.GAMMA = [0.99, 0.0]

        _C.TRAIN.BEGIN_EPOCH = 0
        _C.TRAIN.END_EPOCH = 140

        _C.TRAIN.RESUME = False
        _C.TRAIN.CHECKPOINT = ''

        _C.TRAIN.BATCH_SIZE = 32
        _C.TRAIN.SHUFFLE = True

        # testing
        _C.TEST = CN()
        _C.TEST.LOSS = False
        _C.TEST.BATCH_SIZE = 32  # size of images for each device
        _C.TEST.POST_PROCESS = False
        _C.TEST.SHIFT_HEATMAP = False

        # debug
        _C.DEBUG = CN()
        _C.DEBUG.DEBUG = False
        _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
        _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
        _C.DEBUG.SAVE_HEATMAPS_GT = False
        _C.DEBUG.SAVE_HEATMAPS_PRED = False

        self.cfg = _C

    def update_config(self, args):
        self.cfg.defrost()
        self.cfg.merge_from_file(args.cfg)

        if args.modelDir:
            self.cfg.OUTPUT_DIR = args.modelDir

        if args.logDir:
            self.cfg.LOG_DIR = args.logDir

        if args.dataDir:
            self.cfg.DATA_DIR = args.dataDir

        self.cfg.DATASET.ROOT = os.path.join(
            self.cfg.DATA_DIR, self.cfg.DATASET.ROOT
        )

        self.cfg.MODEL.PRETRAINED = os.path.join(
            self.cfg.DATA_DIR, self.cfg.MODEL.PRETRAINED
        )

        self.cfg.freeze()

    def getcfg(self):
        return self.cfg
