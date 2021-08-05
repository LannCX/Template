import os
import sys
import json
import pickle
import pandas as pd
from tqdm import tqdm

sys.path.append('../')
from utils.cv_utils import fast_read_sel_frames_from_vid_file
from data.base_dataset import BaseDataset


class KineticsDataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(KineticsDataset, self).__init__(cfg, split)
        data_root = cfg.DATASET.ROOT
        anno_file = cfg.DATASET.ANNO_FILE
        class_file = cfg.DATASET.CLS_FILE
        self.name = 'kinetics-'+self.version
        self.class_2_id = json.load(open(class_file % self.version))
        self.id_2_class = {v: k for k, v in self.class_2_id.items()}

        df = pd.read_csv(anno_file % (self.version, split))

        print('Loading annotations...')
        save_path = '%s-%s_data.pkl' % (self.name, split)
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.anno = pickle.load(f)
        else:
            nbar = tqdm(total=df.shape[0])
            for i in range(df.shape[0]):
                nbar.update(1)
                vid_id = df['youtube_id'][i]
                label = df['label'][i].replace(' ', '_')
                is_cc = df['is_cc'][i]
                start = str(int(df['time_start'][i])).zfill(6)
                end = str(int(df['time_end'][i])).zfill(6)
                if split == 'train':
                    vid_path = os.path.join(data_root, split, label, '_'.join([vid_id, start, end]))
                else:
                    vid_path = os.path.join(data_root, split, label, vid_id)
                if os.path.exists(vid_path):
                    nf = len(os.listdir(vid_path))
                    if nf==0:
                        print(vid_path)
                        continue
                    self.anno.append({'id': vid_id, 'path': vid_path, 'label': label.replace('_', ' '),
                                      'nf': nf, 'is_cc': is_cc})

            nbar.close()
            with open(save_path, 'wb') as f:
                pickle.dump(self.anno, f)

        print('Creating %s dataset completed, %d samples.' % (split, len(self.anno)))


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='Train keypoints network')

        parser.add_argument('--cfg', default='../../experiments/kinetics_coconv.yaml',
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
    from cfg.coconv_cfg import CoConvConfig as Config
    args = parse_args()
    cfg = Config(args).getcfg()
    trn = KineticsDataset(cfg, 'train')
    tst = KineticsDataset(cfg, 'val')
