from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import torchbearer
from torchbearer import metrics, Metric
from torchbearer.metrics import default_for_key, running_mean, mean
import torch.nn.functional as F



@metrics.default_for_key("triplet_acc")
@running_mean
@mean
@metrics.lambda_metric("triplet_acc", on_epoch=False)
def triplet_acc(y_pred: torch.Tensor, y_true: torch.Tensor):
    p=2.
    eps=1e-6
    
    anchor, positive, negative = y_pred
    positive_distance = F.pairwise_distance(anchor, positive, p, eps)
    negative_distance = F.pairwise_distance(anchor, negative, p, eps)

    return (positive_distance < negative_distance).view(-1).float()


class RelativeTripletLoss(_Loss):
    def __init__(self, c=100, p=2., margin=1, eps=1e-6, swap=False, size_average=None,
                 reduce=None, reduction="mean", triplet_loss="triplet_margin"):
        super().__init__(size_average, reduce, reduction)
        self.p = p
        self.eps = eps
        self.swap = swap
        self.margin = margin
        self.c  = c

        if triplet_loss == "triplet_margin":
            self.triplet_loss = nn.TripletMarginLoss(p=self.p, reduction="none", margin=self.margin, swap=self.swap)
        elif triplet_loss == "bpr_triplet":
            self.triplet_loss = BayesianPersonalizedRankingTripletLoss(p=self.p, reduction="none")
        else:
            raise NotImplementedError

    def forward(self, anchor, positive, negative, relative_pos, total_ocr, prob):
        loss = self.triplet_loss(anchor, positive, negative)
        
        loss = (self.c/total_ocr.float())*loss

        #loss = (loss*(1-prob)) / (1 + torch.log(relative_pos.float()))

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()