from __future__ import print_function

import torch
import time

#from .util import AverageMeter, accuracy

import sys
sys.path.append('..')
from util import AverageMeter, accuracy, compute_lang_loss

def validate(val_loader, model, criterion, opt, lang_model=None):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    lang_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):

            if opt.lsl:
                (input, target, _, (lang, lang_length, lang_mask)) = data
                # Trim padding to max length in batch
                max_lang_length = lang_length.max()
                lang = lang[:, :max_lang_length]
                lang_mask = lang_mask[:, :max_lang_length]
                assert lang_model is not None, 'Must provide lang_model with LSL'
            else:
                (input, target, _) = data

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                if opt.lsl:
                    lang = lang.cuda()
                    lang_length = lang_length.cuda()
                    lang_mask   = lang_mask.cuda()

            # compute output
            if opt.use_logit:
                output = model(input)
                features = output
            else:
                features, output = model(input, is_feat=True)
                features = features[-1]
                output = model(input)
            loss = criterion(output, target)

            if opt.lsl:
                 lang_loss = compute_lang_loss(features, lang, lang_length, lang_mask, lang_model)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            if opt.lsl:
                lang_losses.update(lang_loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      '{lang}'.format(
                          idx, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5,
                          lang='Lang Loss {lang_losses.val:.4f} {lang_losses.avg:.4f}'.format(lang_losses=lang_losses)
                          if opt.lsl else None))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg, lang_losses.avg if opt.lsl else None
