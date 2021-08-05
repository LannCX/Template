import os
import matplotlib.pyplot as plt
from numpy.random import randint
from torch.utils.data import Dataset

from decord import VideoReader
from data.video_transforms import *
import torchvision.transforms as transf


class BaseDataset(Dataset):
    def __init__(self, cfg, split):
        self.split = split if split == 'train' else 'test'
        self.cfg = cfg
        self.eval = not cfg.IS_TRAIN
        self.test_crops = cfg.TEST.NUM_CROPS
        self.n_segment = cfg.DATASET.N_SEGMENT
        self.version = cfg.DATASET.VERSION
        self.scale_size = cfg.DATASET.SCALE_SIZE
        self.crop_size = cfg.DATASET.CROP_SIZE
        self.data_type = cfg.DATASET.DATA_TYPE
        self.image_tmpl = cfg.DATASET.IMG_FORMAT
        self.samp_mode = cfg.DATASET.SAMP_MODE  # dense, global

        self.anno = []
        self.id_2_class = {}
        self.class_2_id = {}
        self.t_step = 8

        # Transformation
        self.roll = True if 'Inception' in cfg.MODEL.NAME else False
        self.div = not self.roll
        self.is_flip = False if 'something' in cfg.DATASET.NAME else True
        normalize = IdentityTransform() if 'InceptionV1' in cfg.MODEL.NAME else GroupNormalize(cfg.DATASET.MEAN, cfg.DATASET.STD)
        flipping = GroupRandomHorizontalFlip() if self.is_flip and self.split=='train' else IdentityTransform()

        if self.eval:
            if self.test_crops==1:
                scale_crop = transf.Compose([GroupScale(self.scale_size), GroupCenterCrop(self.crop_size)])
            elif self.test_crops==3:
                scale_crop = GroupFCNSample_0(256)
            else:
                raise (KeyError, 'Not supported inference mode {}'.format(self.samp_mode))
        else:
            scale_crop = GroupMultiScaleCrop(self.crop_size, [1, .875, .75, .66], fix_crop=True, more_fix_crop=True) \
                if self.split == 'train' else transf.Compose([GroupScale(self.scale_size), GroupCenterCrop(self.crop_size)])
        self.transform = transf.Compose([
            scale_crop,
            flipping,
            Stack(self.roll),
            ToTorchFormatTensor(div=self.div),
            normalize
        ])

    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')

    def _sample_indices(self, num_frames, need_f):
        if self.samp_mode=='dense':
            if num_frames <= need_f:
                indices = list(range(1, num_frames + 1))
                for i in range(need_f - num_frames):
                    indices.append(num_frames)
            else:
                start = random.randint(1, num_frames - need_f + 1)
                indices = list(range(start, start + need_f))
            return indices
        elif self.samp_mode=='global':
            average_duration = num_frames // need_f
            if average_duration > 0:
                offsets = np.multiply(list(range(need_f)), average_duration) + randint(average_duration, size=need_f)
            elif num_frames > need_f:
                offsets = np.sort(randint(num_frames, size=need_f))
            else:
                # offsets = np.zeros((need_f,))
                offsets = np.array(list(range(num_frames)) + [num_frames - 1] * (need_f - num_frames))
            return offsets + 1
        else:
            raise(KeyError, 'Not supported sampling mode %s'%self.samp_mode)

    def _get_val_indices(self, num_frames, need_f):
        if self.samp_mode=='dense':
            if num_frames <= need_f:
                indices = list(range(1, num_frames + 1))
                for i in range(need_f - num_frames):
                    indices.append(num_frames)
            else:
                start = (num_frames - need_f) // 2
                indices = list(range(start, start + need_f))
            return indices
        elif self.samp_mode=='global':
            if num_frames > need_f:
                tick = num_frames / float(need_f)
                # offsets = np.array([int(tick * x) for x in range(need_f)])
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(need_f)])
            else:
                # offsets = np.zeros((need_f,))
                offsets = np.array(list(range(num_frames)) + [num_frames - 1] * (need_f - num_frames))
            return offsets + 1
        else:
            raise(KeyError, 'Not supported sampling mode %s'%self.samp_mode)

    def _get_test_indices(self, num_frames, need_f):
        if self.samp_mode=='dense':
            indices = []
            if num_frames <= need_f:
                indices = list(range(1, num_frames + 1))
                for i in range(need_f - num_frames):
                    indices.append(num_frames)
            else:
                for start in range(1, num_frames - need_f + 1, self.t_step):
                    indices.extend(list(range(start, start + need_f)))
            return indices
        elif self.samp_mode=='global':
            if num_frames > need_f:
                tick = num_frames / float(need_f)
                # offsets = np.array([int(tick * x) for x in range(need_f)])
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(need_f)])
            else:
                # offsets = np.zeros((need_f,))
                offsets = np.array(list(range(num_frames)) + [num_frames - 1] * (need_f - num_frames))
            return offsets + 1
        else:
            raise(KeyError, 'Not supported sampling mode %s'%self.samp_mode)

    def get_indices(self, nf, need_f):
        indices = self._sample_indices(nf, need_f) if self.split=='train' else \
            self._get_test_indices(nf, need_f) if self.eval else \
                self._get_val_indices(nf, need_f)
        return indices

    def __getitem__(self, index):
        vid_info = self.anno[index]
        if self.data_type=='video':
            vr = VideoReader(vid_info['path'])
            nf = len(vr)
            indices = [x-1 for x in self.get_indices(nf, self.n_segment)]
            img_group = vr.get_batch(indices).asnumpy()
            img_group = [Image.fromarray(img) for img in img_group]
        elif self.data_type=='img':
            nf = vid_info['nf']
            indices = self.get_indices(nf, self.n_segment)
            img_group = [self._load_image(vid_info['path'], int(ind)) for ind in indices]
        else:
            raise KeyError('Not supported data type: {}.'.format(self.data_type))
        img_tensor = self.transform(img_group)  # [T,C,H,W]

        if self.eval:
            vid_data = img_tensor.reshape((-1, self.n_segment)+img_tensor.shape[-3:])
            img_tensor = torch.stack(vid_data, dim=0)  # N clips x C x H x W

        if 'InceptionV1' in self.cfg.MODEL.NAME:
            img_tensor = img_tensor/255. * 2 - 1.  # Inception normalization

        return img_tensor, self.class_2_id[vid_info['label']]

    def __len__(self):
        return len(self.anno)
