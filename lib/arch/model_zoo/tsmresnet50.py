import re
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import sys
sys.path.append('../')
from torch.nn.init import normal_, constant_

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ShiftModule(nn.Module):
    """1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """
    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(
            input_channels, input_channels,
            kernel_size=3, padding=1, groups=input_channels,
            bias=False)
        # weight_size: (2*self.fold, 1, 3)
        if mode == 'shift':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = False
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        # shift by conv
        # import pdb; pdb.set_trace()
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_segment=8):
        super(Bottleneck, self).__init__()
        self.shift = ShiftModule(inplanes, n_segment=n_segment)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.shift(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TSMResNet50(nn.Module):
    VALID_ENDPOINTS = (
        'conv1',
        'bn1',
        'relu',
        'maxpool',
        'layer1',
        'layer2',
        'layer3',
        'layer4',
        'avg_pool',
        'fc',
    )

    def __init__(self, pretrained=False, num_classes=1000, final_endpoint='fc', n_segment=8):
        super(TSMResNet50, self).__init__()
        self.inplanes = 64
        block = Bottleneck
        layers = [3, 4, 6, 3]
        self._final_endpoint = final_endpoint

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], n_segment=n_segment)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, n_segment=n_segment)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, n_segment=n_segment)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, n_segment=n_segment)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if pretrained:
            self.load_org_weights(model_zoo.load_url(model_urls['resnet50']))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    std = 0.001
                    normal_(m.weight, 0, std)
                    constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, n_segment=8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_segment=n_segment))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_segment=n_segment))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape((-1,)+x.shape[-3:])
        for end_point in self.VALID_ENDPOINTS:
            x = getattr(self, end_point)(x)
            if end_point == self._final_endpoint:
                break
        return x

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.drop(x)
        # x = self.fc(x)
        #
        # x = x.reshape((-1, self.t_size) + x.shape[-1:]).mean(dim=1)
        # return x

    def load_org_weights(self, pre_dict):
        tic = time.time()
        model_dict = self.state_dict()
        if 'state_dict' in pre_dict.keys():
            pre_dict = pre_dict['state_dict']
        for name in model_dict.keys():
            if 'num_batches_tracked' in name:
                continue
            is_null = True
            try:
                if model_dict[name].shape == pre_dict[name].shape:
                    model_dict[name] = pre_dict[name]
                    is_null = False
                else:
                    print('size mismatch for %s, expect (%s), but got (%s).'
                          % (name, ','.join([str(x) for x in model_dict[name].shape]),
                             ','.join([str(x) for x in pre_dict[name].shape])))
                continue
            except KeyError:
                pass
            if is_null:
                print('Do not load %s' % name)

        self.load_state_dict(model_dict)

        print('Load pre-trained weightin %.4f sec.' % (time.time()-tic))


if __name__ == '__main__':
    # pred_dict = torch.load('/data_ssd/chenxu/Code/temporal-shift-module/checkpoint/TSM_something_RGB_st_resnet50_avg_segment8_e50/ckpt.best.pth.tar')

    from utils.thop import profile, clever_format
    d_in = torch.rand(1, 8, 3, 224, 224)
    m = TSMResNet50(pretrained=True, num_classes=174, final_endpoint='layer4')

    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
