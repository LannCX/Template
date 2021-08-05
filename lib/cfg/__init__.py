from yacs.config import CfgNode as CN


class BaseConfig(object):
    def __init__(self, args=None):
        _C = CN()
        _C.GPUS = (0,)  # set gpu ids
        _C.WORKERS = 8  # number of workers
        _C.SAVE_FREQ = 1000  # save checkpoint file every #steps
        _C.PRINT_FREQ = 20  # interval steps for printing training information
        _C.USE_VAL = False  # conduct validation during training
        _C.IS_TRAIN = False  # if train this model
        _C.AUTO_RESUME = True  # load latest weight and training configurations automatically
        _C.PRINT_PARAM = False  # print Macs and #Parameters of network
        _C.SNAPSHOT_PREF = 'my_model'  # prefix of log and save directories
        _C.SHOW_BAR = True  # if show the progress bar
        _C.LOG_DIR = 'log'  # root directory of log files
        _C.OUTPUT_DIR = 'checkpoints'  # root directory of output data

        # Cudnn related params
        _C.CUDNN = CN()
        _C.CUDNN.BENCHMARK = True
        _C.CUDNN.DETERMINISTIC = False
        _C.CUDNN.ENABLED = True

        # common params for NETWORK
        _C.MODEL = CN()
        _C.MODEL.NAME = ''
        _C.MODEL.AUX = False  # if use auxiliary head
        _C.MODEL.PRETRAINED = ''  # pre-trained checkpoint file
        _C.MODEL.DROPOUT = 0.5
        _C.MODEL.NUM_CLASSES = 10
        _C.MODEL.HIST_SHOW_NAME = []  # add histograms for specified layers by name, split with commas
        _C.MODEL.EXTRA = CN(new_allowed=True)

        # TODO: loss related arguments, not implemented yet!
        _C.LOSS = CN()
        _C.LOSS.USE_OHKM = False
        _C.LOSS.USE_TARGET_WEIGHT = True

        # DATASET related configurations
        _C.DATASET = CN()
        _C.DATASET.ROOT = ''
        _C.DATASET.NAME = ''  # name of dataset
        _C.DATASET.TRAIN_SET = 'train'
        _C.DATASET.TEST_SET = 'val'
        _C.DATASET.DATA_TYPE = 'video'  # data format, [img, video]
        _C.DATASET.IMG_FORMAT = 'img_{:05d}.jpg'
        _C.DATASET.MODALITY = 'rgb'
        _C.DATASET.VERSION = ''
        _C.DATASET.SCALE_SIZE = (256, 256)  # width * height
        _C.DATASET.CROP_SIZE = (224, 224)
        _C.DATASET.MEAN = [0.485, 0.456, 0.406]
        _C.DATASET.STD = [0.229, 0.224, 0.225]
        _C.DATASET.IS_FLIP = False
        _C.DATASET.SCALE_FACTOR = 0.25
        _C.DATASET.ROT_FACTOR = 30
        _C.DATASET.COLOR_RGB = False
        _C.DATASET.SAMP_MODE = 'global'

        # train
        _C.TRAIN = CN()
        _C.TRAIN.USE_APEX = False
        _C.TRAIN.WARM_UP = False
        _C.TRAIN.WARM_UP_EPOCHS = 5
        _C.TRAIN.PRE_FRFETCH = False  # prefetch data
        _C.TRAIN.LR_FACTOR = 0.1
        _C.TRAIN.LR_STEP = [20, 40, 80]
        _C.TRAIN.LR = 0.001
        _C.TRAIN.PATIENCE = 10
        _C.TRAIN.LR_SCHEDULER = 'multistep'  # support [multistep, cosine, plateau]
        _C.TRAIN.OPTIMIZER = 'sgd'
        _C.TRAIN.MOMENTUM = 0.9
        _C.TRAIN.WD = 0.0001
        _C.TRAIN.NESTEROV = False
        _C.TRAIN.BEGIN_EPOCH = 0
        _C.TRAIN.END_EPOCH = 100
        _C.TRAIN.BATCH_SIZE = 32  # training batch size
        _C.TRAIN.SHUFFLE = True  # shuffle the data during training
        _C.TRAIN.PARTIAL_BN = False  # freeze bn layers
        _C.TRAIN.DROP_LAST = True  # prevent something not % n_GPU during data loading
        _C.TRAIN.ACCUM_N_BS = 1  # accumulate loss every N batches
        _C.TRAIN.DROPOUT = 0.5  # dropout probability

        # testing or validation
        _C.TEST = CN()
        _C.TEST.VAL_FREQ = 1  # test frequency
        _C.TEST.BATCH_SIZE = 1  # testing batch size
        _C.TEST.INIT_VAL = False  # test model before training
        _C.TEST.LOAD_WEIGHT = ''
        _C.TEST.NUM_CROPS = 1

        # debug, visualization
        _C.DEBUG = CN()
        _C.DEBUG.STAT = False
        _C.DEBUG.VIS_FEAT = ''
        _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
        _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False

        self.cfg = _C

    def update_config(self, args):
        self.cfg.defrost()
        self.cfg.merge_from_file(args.cfg)

        if args.modelDir:
            self.cfg.OUTPUT_DIR = args.modelDir

        if args.logDir:
            self.cfg.LOG_DIR = args.logDir

    def getcfg(self):
        self.cfg.freeze()
        return self.cfg
