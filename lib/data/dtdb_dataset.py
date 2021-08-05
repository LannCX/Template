import os
import sys
import json
sys.path.append('../')
from data.base_dataset import BaseDataset
from utils.cv_utils import fast_read_sel_frames_from_vid_file

class DTDBDataset(BaseDataset):
    __bad_list = ['Dominant_rigid_g5_c38.mp4', 'Chaotic_motion_g3_c42.mp4',
                  'Underconstrained_flicker_g1_c112.mp4','Stochastic_motion_g2_c190.mp4',
                  'Underconstrained_blinking_g1_c145.mp4', 'Rotary_motion_g4_c44.mp4',
                  'Rotary_motion_g5_c143.mp4']

    def __init__(self, cfg, split='train'):
        super(DTDBDataset, self).__init__(cfg, split)
        trn_root = '/data_ssd/chenxu/workspace/DTDB/BY_%s/TRAIN' % self.version
        tst_root = '/data_ssd/chenxu/workspace/DTDB/BY_%s/TEST' % self.version
        data_root = trn_root if split=='train' else tst_root
        # data_root = tst_root if split == 'train' else trn_root
        class_file = cfg.DATASET.CLS_FILE % self.version
        self.name = 'DTDB-'+self.version
        self.class_2_id = json.load(open(class_file))
        self.id_2_class = {v: k for k, v in self.class_2_id.items()}

        for cls_name in os.listdir(data_root):
            for vid_name in os.listdir(os.path.join(data_root, cls_name)):
                vid_id = vid_name
                label = cls_name
                vid_path = os.path.join(data_root, cls_name, vid_name)
                if os.path.exists(vid_path) and len(os.listdir(vid_path))>0:
                    self.anno.append({'id': vid_id, 'path': vid_path, 'label': label, 'nf': len(os.listdir(vid_path))})

        print('Creating %s dataset completed, %d samples.' % (self.split, len(self.anno)))


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg', default='../../experiments/dtdb.yaml',
                        help='experiment configure file name',
                        # required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify cfg options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

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
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    from tqdm import tqdm
    from cfg.coconv_cfg import CoConvConfig as Config
    from data.data_loader import customer_data_loader
    args = parse_args()
    cfg = Config(args).getcfg()
    d_loader = customer_data_loader(cfg, cfg.DATASET.TEST_SET)
    nbar = tqdm(total=len(d_loader))
    for item in d_loader:
        nbar.update(1)
    nbar.close()
