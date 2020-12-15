import torch
import torch.nn as nn

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


def loss_cot(U_p1, U_p2):
    # the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Softmax(dim=1)
    LS = nn.LogSoftmax(dim=1)
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