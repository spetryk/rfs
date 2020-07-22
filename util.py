import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, 
                                              size_average=size_average, 
                                              reduce=reduce, 
                                              reduction=reduction,
                                              pos_weight=pos_weight)
    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_lang_loss(features, lang, lang_length, lang_mask, lang_model):
    hyp_batch_size = features.shape[0]
    seq_len = lang.shape[1]

    hypo = lang_model(features, lang, lang_length) # hypo shape: [batch_size, 32?, 5727]

    # Predict all tokens besides start of sentence (which is already given)
    hypo_nofinal = hypo[:, :-1].contiguous() # shape [batch_size, 31, 5727]
    lang_nostart = lang[:, 1:].contiguous()  # shape [batch_size, 31]
    mask_nostart = lang_mask[:, 1:].contiguous() # shape [batch_size, 31]

    hypo_nofinal_2d = hypo_nofinal.view(hyp_batch_size * (seq_len - 1), -1)
    lang_nostart_2d = lang_nostart.long().view(hyp_batch_size * (seq_len - 1))
    hypo_loss = F.cross_entropy(hypo_nofinal_2d, lang_nostart_2d, reduction="none")
    hypo_loss = hypo_loss.view(hyp_batch_size, (seq_len - 1))
    # Mask out sequences based on length
    hypo_loss.masked_fill_(mask_nostart, 0.0)
    # Sum over timesteps / divide by length
    hypo_loss_per_sentence = torch.div(
        hypo_loss.sum(dim=1), (lang_length - 1).float()
    )
    hypo_loss = hypo_loss_per_sentence.mean()

    return hypo_loss
