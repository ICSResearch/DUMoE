import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=0.001):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


class MixedLoss(nn.Module):
    def __init__(self):
        super(MixedLoss, self).__init__()
        self.charloss = CharbonnierLoss()

    def forward(self, x, y, aux_loss):
        loss = torch.mean(self.charloss(x, y) + aux_loss, dim=0)
        return loss
