import sys
sys.path.append('lib')
import torch
from tqdm import tqdm
# from torchstat import stat

from utils.loss import ActionMSELoss
from scheduler.base_scheduler import BaseScheduler
from utils.nn_utils import get_model_summary
from utils.vis import save_batch_video

class ViratScheduler(BaseModel):

    def __init__(self, net, cfg, suffix='', logger=None, loss_type='MSE'):
        super(ViratScheduler, self).__init__(net, cfg, suffix=suffix, logger=logger)
        self.init_op()
        self.criterion = ActionMSELoss(type=loss_type).to(self.device)

    def init_op(self):
        # warm up?
        # dump_input = torch.rand(1, 3, self.cfg.MODEL.T_WIN, self.cfg.MODEL.IMAGE_SIZE[0],
        #                         self.cfg.MODEL.IMAGE_SIZE[1])
        # dump_input = torch.rand(1, 3, self.cfg.MODEL.T_WIN, 112, 112)
        # self.logger.writer.add_graph(self.net, dump_input)
        # self.logger.log(get_model_summary(self.net, dump_input))

        if self.parallel:
            self._net = torch.nn.DataParallel(self._net, device_ids=self.cfg.GPUS).to(self.device)
        else:
            self._net = self._net.to(self.device)

    def feed_data_and_run_loss_op(self, data):
        input_tensor, heatmap = data[0], data[1]
        input_tensor = input_tensor.to(self.device)
        heatmap = heatmap.to(self.device)
        output = self.net(input_tensor)
        loss = self.criterion(output, heatmap)
        return loss

    def predict_and_eval_in_val(self, val_loader, metrics):
        for data in val_loader:
            input_tensor, heatmap = data[0], data[1]
            input_tensor = input_tensor.to(self.device)
            heatmap = heatmap.to(self.device)
            output = self.net(input_tensor)

    def predict_in_tst(self, tst_loader):
        for data in tqdm(tst_loader):
            input_tensor, scale, seg, vid_names, chunk_ids = data
            input_tensor = input_tensor.to(self.device)
            heatmap = self.net(input_tensor)
            save_batch_video(input_tensor, heatmap, 'train', vid_names, chunk_ids)
