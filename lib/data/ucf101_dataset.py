import os
import sys
sys.path.append('../')
from data.base_dataset import BaseDataset


class UCF101Dataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(UCF101Dataset, self).__init__(cfg, split)
        data_root = cfg.DATASET.ROOT
        anno_file = cfg.DATASET.ANNO_FILE % (self.split, self.version)
        class_file = cfg.DATASET.CLS_FILE
        self.name = 'ucf101-'+self.version

        # Load annotations
        with open(class_file, 'r') as f:
            cls_data = f.readlines()
        self.id_2_class = {int(x.split(' ')[0])-1: x.split(' ')[1][:-1] for x in cls_data}
        self.class_2_id = {v: k for k, v in self.id_2_class.items()}

        with open(anno_file, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                vid_id = line.split('/')[-1].split(' ')[0]
                label = line.split('/')[0]
                vid_path = os.path.join(data_root, line.split(' ')[0])
                if os.path.exists(vid_path):
                    self.anno.append({'id': vid_id, 'path': vid_path, 'label': label})
        print('Creating %s dataset completed, %d samples.' % (self.split, len(self.anno)))


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg', default='../../experiments/ucf101_coconv.yaml',
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
