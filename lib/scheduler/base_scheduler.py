'''
Construct general training, validating and testing skeleton.

Reconstructed from: https://github.com/marshimarocj/conv_rnn_trn/
'''
from utils.nn_utils import *


class BaseScheduler(object):

    def __init__(self, net, cfg, suffix='', logger=None):
        self._net = net
        self._cfg = cfg

        self.parallel = True if len(cfg.GPUS) > 1 else False
        self.checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME+'_'+suffix)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.global_steps = 0
        self.device = torch.device('cuda:'+str(cfg.GPUS[0]))

        if logger is not None:
            self.logger = logger
        else:
            raise(ValueError, 'Please define a logger first.')

    @property
    def cfg(self):
        return self._cfg

    @property
    def net(self):
        return self._net

    def init_op(self):
        '''
        Implement initialization operations.
        '''
        raise NotImplementedError("""please customize init_op""")

    def feed_data_and_run_loss_op(self, data):
        """
        return loss value
        """
        raise NotImplementedError("""please customize feed_data_and_run_loss_op_in_val""")

    def predict_and_eval_in_val(self, val_loader, metrics):
        """
        add eval result to metrics dictionary, key is metric name, val is metric value
        """
        raise NotImplementedError("""please customize predict_and_eval_in_val""")

    def predict_in_tst(self, tst_loader):
        """
        write predict result to predict_file
        """
        raise NotImplementedError("""please customize predict_in_tst""")

    def _validation(self, val_loader):
        metrics = {}
        if self.cfg.TEST.LOSS:
            avg_loss = AverageMeter()
            for data in val_loader:
                loss = self.feed_data_and_run_loss_op(data)
                avg_loss.update(loss)
            metrics['loss'] = avg_loss.avg
        self.predict_and_eval_in_val(val_loader, metrics)
        return metrics

    def _train_one_epoch(self, train_loader, optimizer, epoch):
        total_step = len(train_loader)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # update parameters for each batch
        end_time = time.time()
        for step, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time()-end_time)

            # forward
            loss = self.feed_data_and_run_loss_op(data)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            losses.update(loss.item(), data[0].size(0))  # TODO: input data must in the first dim
            self.logger.add_scalar('train_loss', losses.val, self.global_steps)
            self.global_steps += 1

            # measure elapsed time
            batch_time.update(time.time()-end_time)
            end_time = time.time()

            # display training info
            if (step) % self.cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t \
                      Speed {speed:.1f} samples/s\t Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t \
                      Loss {loss.val:.5f} ({loss.avg:.5f}))'.\
                    format(
                        epoch, step, total_step, batch_time=batch_time,
                        speed=data[0].size(0) / batch_time.val,
                        data_time=data_time, loss=losses)
                self.logger.log(msg)

    def train(self, data_loader, **kwarg):
        optimizer = get_optimizer(self.cfg, self.net)
        self.net.train()

        start_epoch = self.cfg.TRAIN.BEGIN_EPOCH
        if self.cfg.AUTO_RESUME:
            try:
                which_epoch = kwarg['which_epoch']
                has_key = True
            except KeyError:
                which_epoch = 'latest'
                has_key = False
            checkpoint_file = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(which_epoch))
            if os.path.exists(checkpoint_file):
                self.logger.log('=> loading checkpoint {}'.format(checkpoint_file))
                last_epoch, w_dict, optm_dict = load_checkpoint(checkpoint_file, is_paraller=self.parallel)
                self.net.load_state_dict(w_dict)
                optimizer.load(optm_dict)
                start_epoch = last_epoch+1
            else:
                if has_key:
                    raise(ValueError, 'checkpoint file of epoch {} not existed!'.format(which_epoch))

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.cfg.TRAIN.LR_STEP,
                                                            self.cfg.TRAIN.LR_FACTOR, last_epoch=start_epoch-1)
        for epoch in range(start_epoch, self.cfg.TRAIN.END_EPOCH):
            start_time = time.time()
            lr_scheduler.step()

            # train for one epoch
            self._train_one_epoch(data_loader, optimizer, epoch=epoch)

            # validate
            if self.cfg.USE_VAL:
                try:
                    metrics = self._validation(kwarg['val_loader'])
                except KeyError:
                    raise(KeyError, 'No validation data loader defined')
                for key in metrics:
                    self.logger.log('{}:{}'.format(key, metrics[key]))

            # save epoch
            if epoch % self.cfg.SAVE_FREQ == 0:
                save_path = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(epoch))
                save_checkpoint(checkpoint_file=save_path,
                                net=self.net,
                                epoch=epoch,
                                optim=optimizer,
                                is_parallel=self.parallel)
                self.logger.log('saving checkpoint to {}'.format(save_path))

            self.logger.log('=> epoch: ({}/{}), leaning rate: {}, cost {:.3f}s'.format(
                epoch, self.cfg.TRAIN.END_EPOCH, lr_scheduler.get_lr(), time.time() - start_time))

    def test(self, test_reader, epoch, is_val):
        checkpoint_file = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(epoch))
        _, state_dict, _, = load_checkpoint(checkpoint_file, self.parallel)
        self.logger.log('=> loading checkpoint {}'.format(checkpoint_file))
        self.net.load_state_dict(state_dict)
        self.net.eval()
        if is_val:
            self.logger.log('Validation')
            metrics = self._validation(test_reader)
            for key in metrics:
                self.logger.log('{}:{}'.format(key, metrics[key]))
        else:
            self.logger.log('Test')
            self.predict_in_tst(test_reader)
