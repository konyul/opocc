import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.amp import autocast
from torch.autograd import Variable
@autocast('cuda',torch.float32)
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

@autocast('cuda',torch.float32)
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss
@autocast('cuda',torch.float32)
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

@autocast('cuda',torch.float32)
def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

@autocast('cuda',torch.float32)
def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 2:
        if ignore is not None:
            valid = (labels != ignore)
            probas = probas[valid]
            labels = labels[valid]
        return probas, labels

    elif probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        #3D segmentation
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

@autocast('cuda',torch.float32)
def elbo_loss(logits, targets, uncertainty, batch_size=65536, kl_weight=0.1):
    total_loss = 0
    num_samples = logits.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # 올림 나눗셈
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        # 현재 배치에 대한 데이터 추출
        batch_logits = logits[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]
        batch_uncertainty = uncertainty[start_idx:end_idx]
        
        # 배치에 대한 loss 계산
        nll = F.cross_entropy(batch_logits, batch_targets, reduction='none')
        kl_div = -torch.log(batch_uncertainty.squeeze() + 1e-8)
        batch_loss = torch.mean(nll * batch_uncertainty.squeeze() + kl_weight * kl_div)
        
        # 전체 loss에 누적
        total_loss += batch_loss * (end_idx - start_idx) / num_samples
    
    return total_loss

@autocast('cuda',torch.float32)
def geo_scal_loss(pred, ssc_target, semantic=True):

    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 0, :, :, :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255 # 255 noise
    nonempty_target = ssc_target != 0 # 0 empty
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        (-torch.ones_like(precision)*torch.log(precision) - (1-torch.ones_like(precision))*torch.log(1-precision)) + \
        (-torch.ones_like(recall)*torch.log(recall) - (1-torch.ones_like(recall))*torch.log(1-recall)) + \
        (-torch.ones_like(spec)*torch.log(spec) - (1-torch.ones_like(spec))*torch.log(1-spec))
    )

@autocast('cuda',torch.float32)
def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = -torch.ones_like(precision)*torch.log(precision) - (1-torch.ones_like(precision))*torch.log(1-precision)
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = -torch.ones_like(recall)*torch.log(recall) - (1-torch.ones_like(recall))*torch.log(1-recall)
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = -torch.ones_like(specificity)*torch.log(specificity) - (1-torch.ones_like(specificity))*torch.log(1-specificity)
                loss_class += loss_specificity
            loss += loss_class
    return loss / count