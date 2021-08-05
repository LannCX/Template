import os
import sys
import json

sys.path.append('../')
from data.base_dataset import BaseDataset


class HMDB51Dataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(HMDB51Dataset, self).__init__(cfg, split)
        data_root = cfg.DATASET.ROOT
        anno_file = cfg.DATASET.ANNO_FILE
        class_file = cfg.DATASET.CLS_FILE
        self.name = 'hmdb51-'+self.version
        self.class_2_id = json.load(open(class_file))
        self.id_2_class = {v: k for k, v in self.class_2_id.items()}

        split_id = '1' if split=='train' else '2'
        anno_files = [x for x in os.listdir(anno_file) if self.version in x]

        for split_file in anno_files:
            label = split_file.replace('_test_'+self.version+'.txt', '')
            split_f_path = os.path.join(anno_file, split_file)
            with open(split_f_path, 'r') as f:
                for l in f:
                    if l.split(' ')[-2] == split_id:
                        vid_id = l.split(' ')[0]
                        vid_path = os.path.join(data_root, label, vid_id)
                        if os.path.exists(vid_path):
                            self.anno.append({'id': vid_id, 'path': vid_path, 'label': label})

        print('Creating %s dataset completed, %d samples.' % (self.split, len(self.anno)))
