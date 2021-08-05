import sys
import time
sys.path.append('lib')
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F

from utils import AverageMeter
from utils.eval import accuracy
from utils.thop import profile, clever_format
from scheduler.base_scheduler import BaseScheduler


class MainScheduler(BaseScheduler):
    def __init__(self, net, cfg, logger=None):
        super(MainScheduler, self).__init__(net, cfg, logger=logger)
        self.cls_criterion = nn.CrossEntropyLoss().to(self.device)
        self.init_op()
        self.train_metrics['train_acc'] = AverageMeter()

    def init_op(self):
        # Parameters and GFLOPs of the network
        if self.cfg.PRINT_PARAM:
            dump_input = torch.rand(1, self.cfg.DATASET.N_SEGMENT, 3,
                                    self.cfg.DATASET.CROP_SIZE[0], self.cfg.DATASET.CROP_SIZE[0])

            macs, params = profile(self._net, inputs=(dump_input,))
            macs, params = clever_format([macs, params], "%.3f")
            self.logger.log('Macs:' + macs + ', Params:' + params)

        if self.parallel:
            self._net = torch.nn.DataParallel(self._net, device_ids=self.cfg.GPUS).to(self.device)

        self.logger.log('Get %d/%d points form %s.' % (self._net.need_points,
                                                       self._net.LAYER_RES[self.cfg.MODEL.ENDPOINT],
                                                       self.cfg.MODEL.ENDPOINT))

    def feed_data_and_run_loss(self, data):
        input_tensor, labels = data[0], data[1]
        input_tensor = input_tensor.to(self.device)
        labels = labels.to(self.device)

        out_logits, _ = self._net(input_tensor)
        loss = self.cls_criterion(out_logits, labels)
        acc = accuracy(out_logits.data, labels, topk=(1,))[0]
        self.train_metrics['train_acc'].update(acc.item(), input_tensor.size(0))

        return loss

    def predict_and_eval_in_val(self, val_loader, metrics):
        top1 = AverageMeter()
        top5 = AverageMeter()
        avg_loss = AverageMeter()
        if self.show_process_bar:
            nbar = tqdm(total=len(val_loader))
        count = 0
        for data in val_loader:
            if self.show_process_bar:
                nbar.update(1)
            count += 1
            input_tensor, labels = data[0], data[1]
            input_tensor = input_tensor.to(self.device)
            labels = labels.to(self.device)

            out_logits, heatmap = self._net(input_tensor)
            loss = self.cls_criterion(out_logits, labels)
            prec1, prec5 = accuracy(out_logits.data, labels, topk=(1, 5))
            top1.update(prec1.item(), input_tensor.size(0))
            top5.update(prec5.item(), input_tensor.size(0))
            avg_loss.update(loss.item())
            if count == 1:
                for n in range(self.cfg.DATASET.N_SEGMENT):
                    self.logger.add_image('heatmap_t%d' % n, heatmap[0][n].repeat(3,1,1)*255, self.global_steps)

        if self.show_process_bar:
            nbar.close()
        metrics['loss'] = avg_loss.avg
        metrics['top1'] = top1.avg
        metrics['top5'] = top5.avg
        self.logger.add_scalar('val_loss', metrics['loss'], self.global_steps)
        self.logger.add_scalar('top1_acc', metrics['top1'], self.global_steps)
        self.logger.add_scalar('top5_acc', metrics['top5'], self.global_steps)
        if metrics['top1'] > self.best_metrics:
            self.best_metrics = metrics['top1']
            self.is_best = True

    def predict_in_tst(self, tst_loader):
        top1 = AverageMeter()
        top5 = AverageMeter()
        nbar = tqdm(total=len(tst_loader))
        start_time = time.time()
        for data in tst_loader:
            nbar.update(1)
            input_tensor, labels = data[0], data[1]
            b, n_clip, _, _, _ = input_tensor.shape
            input_tensor = input_tensor.reshape((-1,)+input_tensor.shape[2:]).to(self.device)
            labels = labels.to(self.device)
            output = self._net(input_tensor)
            pred = F.softmax(output, dim=1).reshape(b, n_clip, -1).mean(dim=1)
            prec1, prec5 = accuracy(pred.data, labels, topk=(1, 5))
            top1.update(prec1.item(), input_tensor.size(0))
            top5.update(prec5.item(), input_tensor.size(0))
        nbar.close()
        cnt_time = time.time()-start_time
        self.logger.log('=>Evaluation finished, \n '
                        '=>Average {:.3f} sec/video \n'
                        '=>Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(float(cnt_time)/len(tst_loader.dataset), top1.avg, top5.avg))

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self._net.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) \
                    or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose2d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                lr5_weight.append(ps[0])
                if len(ps) == 2:
                    lr10_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'initial_lr': self.cfg.TRAIN.LR, 'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'initial_lr': self.cfg.TRAIN.LR, 'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1, 'initial_lr': self.cfg.TRAIN.LR,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0, 'initial_lr': self.cfg.TRAIN.LR,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0, 'initial_lr': self.cfg.TRAIN.LR,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1, 'initial_lr': self.cfg.TRAIN.LR,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1, 'initial_lr': self.cfg.TRAIN.LR,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0, 'initial_lr': self.cfg.TRAIN.LR,
             'name': "lr10_bias"},
        ]

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if self.cfg.TRAIN.LR_SCHEDULER == 'multistep':
            decay = self.cfg.TRAIN.LR_FACTOR ** (sum(epoch >= np.array(self.cfg.TRAIN.LR_STEP)))
            lr = self.cfg.TRAIN.LR * decay
        elif self.cfg.TRAIN.LR_SCHEDULER == 'cosine':
            import math
            lr = 0.5 * (1 + math.cos(math.pi * epoch / self.cfg.TRAIN.END_EPOCH))
        else:
            raise NotImplementedError
        w_decay = self.cfg.TRAIN.WD
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = w_decay * param_group['decay_mult']

