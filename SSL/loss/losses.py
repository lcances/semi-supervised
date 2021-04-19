from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss

class Activation(Enum):
    SIGMOID=0
    SOFTMAX=1


class ValidLoss(Enum):
    CROSS_ENTROPY=0
    BINARY_CROSS_ENTROPY=1


class DCTSupWithLogitsLoss(_WeightedLoss):
    def __init__(weight: Optional[Tensor] = None,size_average=None, reduce=None, reduction: str = 'mean',
                 sub_loss: ValidLoss = ValidLoss.CROSS_ENTROPY):
        super().__init__(weight, size_average, reduce, reduction)

        if sub_loss == ValidLoss.CROSS_ENTROPY:
            self.sub_loss = nn.CrossEntropyLoss(weight, size_average, reduce, reduction)
        
        else sub_loss == ValidLoss.BINARY_CROSS_ENTROPY:
            self.sub_loss = nn.BCEWithLogitsLoss(size_average, reduce, reduction)

    def forward(self, input1: Tuple, input2: Tuple) -> Tensor:
        logits_1, y_1 = input1
        logits_2, y_2 = input2
        loss1 = self.sub_loss(logits_1, y_1)
        loss2 = self.sub_loss(logits_2, y_2)
        return (loss1 + loss2)


class DCTCotWithLogitsLoss(_WeightedLoss):
    def __init__(weight: Optional[Tensor] = None,size_average=None, reduce=None, reduction: str = 'mean',
                 activation: Activation = Activation.SOFTMAX):
        super().__init__(weight, size_average, reduce, reduction)

        if activation == Activation.SOFTMAX:
            self.S = nn.Softmax(dim=1)
            self.LS = nn.LogSoftmax(dim=1)
        else:
            self.S = nn.Sigmoid()
            self.LS = nn.LogSigmoid()

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        U_batch_size = input1.size()[0]
        eps=1e-8

        a1 = 0.5 * (self.S(input1) + self.S(input2))
        a1 = torch.clamp(a1, min=eps)
        
        loss1 = a1 * torch.log(a1)
        loss1 = -torch.sum(loss1)

        loss2 = self.S(input1) * self.LS(input1)
        loss2 = -torch.sum(loss2)

        loss3 = self.S(input2) * self.LS(input2)
        loss3 = -torch.sum(loss3)

        return (loss1 - 0.5 * (loss2 + loss3)) / U_batch_size


class DCTDiffWithLogitsLoss(_WeightedLoss):
    def __init__(weight: Optional[Tensor] = None,size_average=None, reduce=None, reduction: str = 'mean',
                 activation: Activation = Activation.SOFTMAX):
        super().__init__(weight, size_average, reduce, reduction)

        if activation == Activation.SOFTMAX:
            self.S = nn.Softmax(dim=1)
            self.LS = nn.LogSoftmax(dim=1)
        else:
            self.S = nn.Sigmoid()
            self.LS = nn.LogSigmoid()
        pass

    def forward(self, input_s1: Tuple, input_s2: Tuple, input_u1: Tuple, input_u2: Tuple) -> Tensor:
        logits_s1, adv_logits_s1 = input_s1
        logits_s2, adv_logits_s2 = input_s2
        logits_u1, adv_logits_u1 = input_u1
        logits_u2, adv_logits_u2 = input_u2

        s_batch_size = logit_s1.size()[0]
        u_batch_size = logit_u1.size()[0]
        total_batch_size = s_batch_size + u_batch_size

        a = self.S(logits_s2) * self.LS(adv_logits_s1)
        a = torch.sum(a)

        b = self.S(logits_s1) * self.LS(adv_logits_s2)
        b = torch.sum(b)

        c = self.S(logits_u2) * self.LS(adv_logits_u1)
        c = torch.sum(c)

        d = self.S(logits_u1) * self.LS(adv_logits_u2)
        d = torch.sum(d)

        return -(a + b + c + d) / total_batch_size



def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss()
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2)
    return (loss1 + loss2)


def p_loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss()
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2)
    return loss1, loss2, (loss1 + loss2)


def loss_cot(U_p1, U_p2, activation: Activation = Activation.SOFTMAX):
    # the Jensen-Shannon divergence between p1(x) and p2(x)
    if activation == Activation.SOFTMAX:
        S = nn.Softmax(dim=1)
        LS = nn.LogSoftmax(dim=1)

    elif activation == Activation.SIGMOID:
        S = nn.Sigmoid()
        LS = nn.LogSigmoid()

    else:
        raise f'This activation ({activation}) is not available'

    U_batch_size = U_p1.size()[0]
    eps=1e-8

    a1 = 0.5 * (S(U_p1) + S(U_p2))
    a1 = torch.clamp(a1, min=eps)
    
    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)

    loss2 = S(U_p1) * LS(U_p1)
    loss2 = -torch.sum(loss2)

    loss3 = S(U_p2) * LS(U_p2)
    loss3 = -torch.sum(loss3)

    return (loss1 - 0.5 * (loss2 + loss3)) / U_batch_size


def JensenShanon(logits_1, logits_2):
    return loss_cot(logits_1, logits_2)


def js_from_softmax(p1, p2):
    U_batch_size = p1.size()[0]
    eps=1e-8

    a1 = 0.5 * (p1 + p2)
    a1 = torch.clamp(a1, min=eps)

    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)

    loss2 = p1 * torch.log(p1)
    loss2 = -torch.sum(loss2)

    loss3 = p2 * torch.log(p2)
    loss3 = -torch.sum(loss3)

    return (loss1 - 0.5 * (loss2 + loss3)) / U_batch_size


def loss_diff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2,
              logit_U1, logit_U2, perturbed_logit_U1, perturbed_logit_U2):
    S = nn.Softmax(dim=1)
    LS = nn.LogSoftmax(dim=1)

    S_batch_size = logit_S1.size()[0]
    U_batch_size = logit_U1.size()[0]
    total_batch_size = S_batch_size + U_batch_size

    a = S(logit_S2) * LS(perturbed_logit_S1)
    a = torch.sum(a)

    b = S(logit_S1) * LS(perturbed_logit_S2)
    b = torch.sum(b)

    c = S(logit_U2) * LS(perturbed_logit_U1)
    c = torch.sum(c)

    d = S(logit_U1) * LS(perturbed_logit_U2)
    d = torch.sum(d)

    return -(a + b + c + d) / total_batch_size


def loss_diff_fusion(logits_s, adv_logits_s, logits_u, adv_logits_u):
    S_batch_size = logits_s.size()[0]
    U_batch_size = logits_u.size()[0]
    total_batch_size = S_batch_size + U_batch_size

    a = logits_s * torch.log(adv_logits_s)
    a = torch.sum(a)

    b = logits_u * torch.log(adv_logits_u)
    b = torch.sum(b)

    return -(a + b) / total_batch_size


def loss_diff_fusion_partial(fusion_s, adv_logits_s1, adv_logits_s2,
                             fusion_u, adv_logits_u1, adv_logits_u2):
    LS = nn.LogSoftmax(dim=1)

    S_batch_size = fusion_s.size()[0]
    U_batch_size = fusion_u.size()[0]
    total_batch_size = S_batch_size + U_batch_size

    a = fusion_s * ( LS(adv_logits_s1) + LS(adv_logits_s2) )
    a = torch.sum(a)

    b = fusion_u * ( LS(adv_logits_u1) + LS(adv_logits_u2) )
    b = torch.sum(b)

    return -(a + b) / total_batch_size


def loss_diff_fusion_simple(fusion_s, fusion_adv_s, fusion_u, fusion_adv_u):
    S_batch_size = fusion_s.size()[0]
    U_batch_size = fusion_u.size()[0]
    total_batch_size = S_batch_size + U_batch_size

    a = fusion_s * torch.log(fusion_adv_s)
    a = torch.sum(a)

    b = fusion_u * torch.log(fusion_u)
    b = torch.sum(b)

    return -(a + b) / total_batch_size


def loss_diff_simple(logits_s, adv_logits_ts, logits_u, adv_logits_tu):
    S = nn.Softmax(dim=1)
    LS = nn.LogSoftmax(dim=1)

    S_batch_size = logits_s.size()[0]
    U_batch_size = logits_u.size()[0]
    total_batch_size = S_batch_size + U_batch_size

    a = S(logits_s) * LS(adv_logits_ts)
    a = torch.sum(a)

    b = S(logits_u) * LS(adv_logits_tu)
    b = torch.sum(b)

    return -(a + b) / total_batch_size