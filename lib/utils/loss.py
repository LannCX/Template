import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss


class SSLLoss(nn.Module):
    def __init__(self, device):
        super(SSLLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss().to(device)
        self.device = device

    def forward(self, inp, target):
        b, m, n = inp.shape
        row_inp = inp.reshape(-1, n)
        row_target = target.reshape(-1)
        loss_r = self.ce(row_inp, row_target)

        col_inp = inp.permute(0, 2, 1).reshape(-1, n)
        ones = torch.sparse.torch.eye(n)
        one_hot = ones.to(self.device).index_select(0, row_target)
        col_target = one_hot.argmax(dim=1).reshape(-1)
        loss_c = self.ce(col_inp, col_target)

        return loss_r, loss_c
