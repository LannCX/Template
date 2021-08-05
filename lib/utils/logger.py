import os
import time
import logging


class Logger:

    def __init__(self, log_dir, name='', tbX_dir=None, backend='tensorboardX'):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        prefix = 'log_{}'.format(time_str)
        log_file = os.path.join(log_dir, prefix+'.log')
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(log_file), format=head)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        logging.getLogger(name).addHandler(logging.StreamHandler())

        tbX_dir = log_dir if tbX_dir is None else tbX_dir
        if not os.path.exists(tbX_dir):
            os.makedirs(tbX_dir)
        if backend == 'tensorboardX':
            from tensorboardX import SummaryWriter
            print('Using tensorboardX as backend.')
            self.writer = SummaryWriter(log_dir=tbX_dir)
        elif backend == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            print('Using tensorboard in pytorch as backend.')
            self.writer = SummaryWriter(log_dir=tbX_dir)
        else:
            raise(ValueError, 'Backend {} is not supported yet.'.format(backend))

    def log(self, msg, type='info', *args, **kwargs):
        if type=='info':
            self.logger.info(msg, *args, **kwargs)
        elif type=='warn':
            self.logger.warning(msg, *args, **kwargs)
        elif type=='error':
            self.logger.error(msg, *args, **kwargs)
        elif type=='critical':
            self.logger.critical(msg, *args, **kwargs)
        else:
            raise(ValueError, 'Not supported type')
        # self.writer.add_text(msg)

    def add_scalar(self, *args, **kwargs):
        self.writer.add_scalar(*args, **kwargs)

    def add_scalars(self, *args, **kwargs):
        self.writer.add_scalars(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        self.writer.add_graph(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        # images should be normalized, and with [3,H,W] shape
        self.writer.add_image(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        # figure input should be matplotlib.pyplot.figure or a list of matplotlib.pyplot.figure
        self.writer.add_figure(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        self.writer.add_histogram(*args, **kwargs)
