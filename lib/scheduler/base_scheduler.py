"""
General training, validation and testing schema.

Reconstructed from: https://github.com/marshimarocj/conv_rnn_trn/
"""
from utils.nn_utils import *
from utils import AverageMeter
import traceback
from tqdm import tqdm
try:
    from apex import amp
except ImportError:
    pass


class BaseScheduler(object):
    def __init__(self, net, cfg, logger=None):
        self._net = net
        self._cfg = cfg
        self.logger = logger
        self.modality = cfg.DATASET.MODALITY
        self.show_process_bar = cfg.SHOW_BAR
        self._enable_pbn = cfg.TRAIN.PARTIAL_BN
        self.parallel = True if len(cfg.GPUS) > 1 else False
        self.optimizer = get_optimizer(self.cfg, self._net, policies=self.get_optim_policies())
        self.checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, '.'.join([cfg.SNAPSHOT_PREF, cfg.MODEL.NAME]))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.best_metrics = 0  # initialized best metrics
        self.best_loss = 1000  # initialized best loss
        self.is_best = False
        self.global_steps = 0
        self.train_metrics = {}
        self.device = torch.device('cuda:'+str(cfg.GPUS[0])) if torch.cuda.device_count() else torch.device('cpu')

        if self.cfg.TRAIN.USE_APEX:
            self._net, self.optimizer = amp.initialize(self._net, self.optimizer, opt_level='O1')

    @property
    def cfg(self):
        return self._cfg

    def init_op(self):
        '''
        Implement initialization operations.
        '''
        raise NotImplementedError("""please customize init_op""")

    def feed_data_and_run_loss(self, data):
        """
        return loss value
        """
        raise NotImplementedError("""please customize feed_data_and_run_loss""")

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

    def adjust_learning_rate(self, epoch):
        """
        adjust learning rate by customer
        """
        raise NotImplementedError("""please customize adjust_learning_rate""")

    def get_optim_policies(self):
        return None

    def get_lr_schedule(self, start_epoch):
        warm_up_epoch = 0
        if self.cfg.TRAIN.WARM_UP:
            warm_up_epoch = self.cfg.TRAIN.WARM_UP_EPOCHS

        if self.cfg.TRAIN.LR_SCHEDULER == 'cosine':
            lr_lamda = lambda epoch: (epoch-self.cfg.TRAIN.BEGIN_EPOCH)/warm_up_epoch if epoch <= warm_up_epoch \
                else 0.5*(math.cos((epoch-warm_up_epoch)/(self.cfg.TRAIN.END_EPOCH-warm_up_epoch)*math.pi) + 1)
        else:
            # Default: MultiStepLR
            lr_lamda = lambda epoch: (epoch-self.cfg.TRAIN.BEGIN_EPOCH) / warm_up_epoch if epoch < warm_up_epoch \
                else self.cfg.TRAIN.LR_FACTOR ** len([m for m in self.cfg.TRAIN.LR_STEP if m <= epoch])

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                      lr_lambda=lr_lamda,
                                                      last_epoch=start_epoch - 1)
        if self.cfg.TRAIN.LR_SCHEDULER == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                   min_lr=1e-7,
                                                                   patience=self.cfg.TRAIN.PATIENCE,
                                                                   factor=self.cfg.TRAIN.LR_FACTOR,
                                                                   verbose=True)
        return scheduler

    def train_one_epoch(self, train_loader, epoch):
        total_step = len(train_loader)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # update parameters for each batch
        end_time = time.time()
        if self.show_process_bar:
            pbar = tqdm(total=len(train_loader))
        for step, data in enumerate(train_loader):
            if self.show_process_bar:
                pbar.update(1)
            # measure data loading time
            data_time.update(time.time() - end_time)
            step += 1

            # forward
            loss = self.feed_data_and_run_loss_op(data)

            # backward
            loss = loss / self.cfg.TRAIN.ACCUM_N_BS  # loss regularization
            if self.cfg.TRAIN.USE_APEX:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # accumulate gradient
            if step % self.cfg.TRAIN.ACCUM_N_BS==0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # record loss and parameters
            losses.update(loss.item(), data[0].size(0))
            self.logger.add_scalar('train_loss', losses.val, self.global_steps)
            for name, param in self._net.named_parameters():
                for item in self.cfg.MODEL.HIST_SHOW_NAME:
                    if item in name:
                        self.logger.add_histogram(self.cfg.MODEL.NAME+'_'+name,
                                                  param.clone().cpu().data.numpy(),
                                                  self.global_steps)
            self.global_steps += 1

            # save epoch
            if self.global_steps % self.cfg.SAVE_FREQ == 0:
                save_path = os.path.join(self.checkpoint_dir, 'epoch_latest.pth')
                save_checkpoint(checkpoint_file=save_path,
                                net=self._net,
                                epoch=epoch,
                                gs=self.global_steps,
                                optim=self.optimizer,
                                is_parallel=self.parallel)
                self.logger.log('saving checkpoint to {}'.format(save_path))

            # display training info
            if step % self.cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t ' \
                      'Speed {speed:.1f} samples/s\t Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t ' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f}))'.format(
                    epoch, step, total_step, batch_time=batch_time,
                    speed=data[0].size(0)/batch_time.val,
                    data_time=data_time,
                    loss=losses)
                self.logger.log(msg)

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

        if self.show_process_bar:
            pbar.close()
        return losses.avg

    def _validation(self, val_loader):
        metrics = {}
        with torch.no_grad():
            self.predict_and_eval_in_val(val_loader, metrics)
        return metrics

    def train(self, data_loader, **kwarg):
        self._net.train()
        start_epoch = self.cfg.TRAIN.BEGIN_EPOCH
        if self.cfg.AUTO_RESUME:
            try:
                which_epoch = kwarg['which_epoch']
                if which_epoch is None:
                    which_epoch = 'latest'
                    have_key = False
                else:
                    have_key = True
            except KeyError:
                which_epoch = 'latest'
                have_key = False
            checkpoint_file = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(which_epoch))
            if os.path.exists(checkpoint_file):
                self.logger.log('=> loading checkpoint {}'.format(checkpoint_file))
                w_dict, chech_info = load_checkpoint(checkpoint_file, self.parallel)
                self._net.load_state_dict(w_dict)
                self.optimizer.load_state_dict(chech_info['optimizer'])
                start_epoch = chech_info['epoch']+1
                self.global_steps = chech_info['global_step']
            else:
                if have_key:
                    raise(ValueError, 'checkpoint file of epoch {} not existed!'.format(which_epoch))

        # round 0, just for quick check
        if self.cfg.USE_VAL and self.cfg.TEST.INIT_VAL:
            self.logger.log('step (0)')
            self._net.eval()
            try:
                metrics = self._validation(kwarg['val_loader'])
            except KeyError:
                raise (KeyError, 'No validation data loader defined')
            for key in metrics:
                self.logger.log('{}:{}'.format(key, metrics[key]))
            self._net.train()

        lr_scheduler = self.get_lr_schedule(start_epoch)

        for epoch in range(start_epoch, self.cfg.TRAIN.END_EPOCH+1):
            try:
                start_time = time.time()
                # train for one epoch
                self.train_one_epoch(data_loader, epoch=epoch)

                # validate
                if self.cfg.USE_VAL and epoch % self.cfg.TEST.VAL_FREQ == 0:
                    self.logger.log('Validating...')
                    self._net.eval()
                    try:
                        metrics = self._validation(kwarg['val_loader'])
                    except KeyError:
                        raise (KeyError, 'No validation data loader defined')
                    for key in metrics:
                        self.logger.log('{}:{}'.format(key, metrics[key]))
                    self.logger.log('Best metric: {}'.format(self.best_metrics))
                    if self.is_best:
                        save_path = os.path.join(self.checkpoint_dir, 'best_model.pth'.format(epoch))
                        save_checkpoint(save_path,
                                        self._net,
                                        epoch=epoch,
                                        gs=self.global_steps,
                                        optim=self.optimizer,
                                        is_parallel=self.parallel)
                        self.is_best = False
                    self._net.train()

                # update learning rate
                try:
                    self.adjust_learning_rate(epoch)
                except NotImplementedError:
                    if self.cfg.TRAIN.LR_SCHEDULER == 'plateau':
                        # TODO: process conflict with INIT_VAL
                        lr_scheduler.step(metrics['loss'])
                        # try:
                        #     lr_scheduler.step(metrics['loss'])
                        # except :
                        #     self.logger.log('Please set INIT_VAL as true when using plateau lr_scheduler.')
                    else:
                        lr_scheduler.step()

                # Print info: epoch, lr, time, metrics
                lr_list = [str(param_group['lr']) for param_group in self.optimizer.param_groups]
                self.logger.log('=>learning rate: '+' '.join(lr_list))
                self.logger.log('=>epoch: ({}/{}), cost {:.3f}s'.format(
                    epoch, self.cfg.TRAIN.END_EPOCH, time.time() - start_time))
                for k, v in self.train_metrics.items():
                    self.logger.log('=>' + k + ':' + str(v.avg))
            except KeyboardInterrupt:
                # TODO: handle exception error
                # save_path = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format('latest'))
                # save_checkpoint(checkpoint_file=save_path,
                #                 net=self._net,
                #                 epoch=epoch,
                #                 gs=self.global_steps,
                #                 optim=self.optimizer,
                #                 is_parallel=self.parallel)
                # self.logger.log('saving checkpoint to {}'.format(save_path))
                # self.logger.writer.close()
                traceback.print_exc()
        self.logger.writer.close()

    def test(self, test_reader, weight_file):
        state_dict, _, = load_checkpoint(weight_file, self.parallel)
        self.logger.log('=> loading checkpoint {}'.format(weight_file))
        self._net.load_state_dict(state_dict, strict=False)
        self._net.eval()
        self.logger.log('Testing...')
        with torch.no_grad():
            self.predict_in_tst(test_reader)
