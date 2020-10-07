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


class BayesianPersonalizedRankingTripletLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = 1 - \sigma (d(a_i, p_i) - d(a_i, n_i))

    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Args:
        margin (float, optional): Default: `1`.
        p (int, optional): The norm degree for pairwise distance. Default: `2`.
        swap (float, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Shape:
        - Input: :math:`(N, D)` where `D` is the vector dimension.
        - Output: scalar. If `reduce` is False, then `(N)`.

    >>> triplet_loss = BayesianPersonalizedRankingTripletLoss(p=2)
    >>> input1 = torch.randn(100, 128, requires_grad=True)
    >>> input2 = torch.randn(100, 128, requires_grad=True)
    >>> input3 = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()

    .. _BPR: Bayesian Personalized Ranking from Implicit Feedback:
        https://arxiv.org/abs/1205.2618
    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, p=2., eps=1e-6, swap=False, size_average=None,
                 reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, self.p, self.eps)
        negative_distance = F.pairwise_distance(anchor, negative, self.p, self.eps)

        if self.swap:
            positive_negative_distance = F.pairwise_distance(positive, negative, self.p, self.eps)
            negative_distance = torch.min(negative_distance, positive_negative_distance)

        loss = 1 - F.sigmoid(positive_distance - negative_distance)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
            
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