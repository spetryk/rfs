from __future__ import print_function

import os
import argparse
import socket
import time
import sys

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.cub import CUB2011, MetaCUB2011
from dataset.transform_cfg import transforms_options, transforms_list, unnormalize_cub

from util import adjust_learning_rate, accuracy, AverageMeter, compute_lang_loss
from eval.meta_eval import meta_test
from eval.cls_eval import validate

import wandb

# lsl
from lsl.birds.data import lang_utils
from lsl.birds.models.language import TextProposal


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam dataset')

    # optimizer
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', 'CUB'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    #parser.add_argument('--model_path', type=str, default='', help='path to save model')
    #parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    #parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

    # *** LSL params
    parser.add_argument("--lsl", action="store_true",
                        help='Perform LSL during training on embedding. Only CUB dataset supported for now')
    parser.add_argument("--rnn_type", choices=["gru", "lstm"], default="gru",
                        help='Gated Recurrent Unit, or RNN for language embedding')
    parser.add_argument("--rnn_dropout", default=0.0, type=float)
    parser.add_argument("--rnn_num_layers", default=1, type=int)
    parser.add_argument(
        "--language_filter", default="all", choices=["all", "color", "nocolor"]
    )
    parser.add_argument(
        "--lang_supervision", default="instance", choices=["instance", "class"]
    )
    parser.add_argument("--glove_init", action="store_true")
    parser.add_argument("--freeze_emb", action="store_true",
                        help='Freeze LM word embedding layer')
    parser.add_argument("--scramble_lang", action="store_true")
    parser.add_argument("--sample_class_lang", action="store_true",
                        help='Sample language randomly from class, rather than getting lang assoc. w/ img')
    parser.add_argument("--scramble_all", action="store_true")
    parser.add_argument("--shuffle_lang", action="store_true")
    parser.add_argument("--scramble_lang_class", action="store_true")
    parser.add_argument("--n_caption", choices=list(range(1, 11)), type=int, default=1)
    parser.add_argument("--max_class", type=int, default=None)
    parser.add_argument("--max_img_per_class", type=int, default=None)
    parser.add_argument("--max_lang_per_class", type=int, default=None)
    parser.add_argument("--lang_lambda", type=float, default=5)
    parser.add_argument("--lang_emb_size", type=int, default=300)
    parser.add_argument("--lang_hidden_size", type=int, default=200)
    parser.add_argument("--rnn_lr_scale", default=1.0, type=float)
    parser.add_argument("--lang_dir", default='./lsl/birds/reed-birds', type=str)
    parser.add_argument('--use_logit', action='store_true',
                        help='use logit layer for input to lang model. otherwise, uses pre-classification feature vector')

    # wandb logging
    parser.add_argument('--dryrun',      action='store_true',
                        help='Use flag to prevent logging to wandb')
    parser.add_argument('--name', type=str, default=None,
                        help='name for wandb run')

    opt = parser.parse_args()

    if opt.dryrun:
        os.environ['WANDB_MODE'] = 'dryrun'
    else:
        os.environ['WANDB_MODE'] = 'run'

    if opt.name is not None:
        os.environ['WANDB_RUN_ID'] = opt.name


    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if opt.dataset == 'CUB':
        opt.dataset = 'CUB_200_2011'

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.data_root:
        opt.data_root = os.path.join('./data', opt.dataset)
    else:
        opt.data_root = os.path.join(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.n_gpu = torch.cuda.device_count()

    return opt


def main():

    opt = parse_option()

    if opt.name is not None:
        wandb.init(name=opt.name)
    else:
        wandb.init()
    wandb.config.update(opt)

    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='train_phase_val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']

        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    elif opt.dataset == 'CUB_200_2011':
        train_trans, test_trans = transforms_options['C']

        vocab = lang_utils.load_vocab(opt.lang_dir) if opt.lsl else None
        devocab = {v:k for k,v in vocab.items()} if opt.lsl else None

        train_loader = DataLoader(CUB2011(args=opt, partition=train_partition, transform=train_trans,
                                          vocab=vocab),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CUB2011(args=opt, partition='val', transform=test_trans, vocab=vocab),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaCUB2011(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans, vocab=vocab),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCUB2011(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans, vocab=vocab),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            raise NotImplementedError(opt.dataset) # no trainval supported yet
            n_cls = 150
        else:
            n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    print('Amount training data: {}'.format(len(train_loader.dataset)))
    print('Amount val data:      {}'.format(len(val_loader.dataset)))

    # model
    model = create_model(opt.model, n_cls, opt.dataset)

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()


    # lsl
    lang_model = None
    if opt.lsl:
        if opt.glove_init:
            vecs = lang_utils.glove_init(vocab, emb_size=opt.lang_emb_size)
        embedding_model = nn.Embedding(
            len(vocab), opt.lang_emb_size, _weight=vecs if opt.glove_init else None
        )
        if opt.freeze_emb:
            embedding_model.weight.requires_grad = False

        lang_input_size = n_cls if opt.use_logit else 640 # 640 for resnet12
        lang_model = TextProposal(
            embedding_model,
            input_size=lang_input_size,
            hidden_size=opt.lang_hidden_size,
            project_input=lang_input_size != opt.lang_hidden_size,
            rnn=opt.rnn_type,
            num_layers=opt.rnn_num_layers,
            dropout=opt.rnn_dropout,
            vocab=vocab,
            **lang_utils.get_special_indices(vocab)
        )


    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        if opt.lsl:
            embedding_model = embedding_model.cuda()
            lang_model = lang_model.cuda()

    # tensorboard
    #logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    # routine: supervised pre-training
    best_val_acc = 0
    for epoch in range(1, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, train_lang_loss = train(
            epoch, train_loader, model, criterion, optimizer, opt, lang_model,
            devocab=devocab if opt.lsl else None
        )
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        print("==> validating...")
        test_acc, test_acc_top5, test_loss, test_lang_loss = validate(
            val_loader, model, criterion, opt, lang_model
        )

        # wandb
        log_metrics = {
            'train_acc':     train_acc,
            'train_loss':    train_loss,
            'val_acc':      test_acc,
            'val_acc_top5': test_acc_top5,
            'val_loss':     test_loss
        }
        if opt.lsl:
            log_metrics['train_lang_loss'] = train_lang_loss
            log_metrics['val_lang_loss']  = test_lang_loss
        wandb.log(log_metrics, step=epoch)

        # # regular saving
        # if epoch % opt.save_freq == 0 and not opt.dryrun:
        #     print('==> Saving...')
        #     state = {
        #         'epoch': epoch,
        #         'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
        #     }
        #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     torch.save(state, save_file)


        if test_acc > best_val_acc:
            wandb.run.summary['best_val_acc'] = test_acc
            wandb.run.summary['best_val_acc_epoch'] = epoch

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
    }
    save_file = os.path.join(wandb.run.dir, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)

    # evaluate on test set
    print("==> testing...")
    start = time.time()
    (test_acc, test_std), (test_acc5, test_std5) = meta_test(model, meta_testloader)
    test_time = time.time() - start
    print('Using logit layer for embedding')
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))
    print('test_acc top 5: {:.4f}, test_std top 5: {:.4f}, time: {:.1f}'.format(test_acc5, test_std5, test_time))

    start = time.time()
    (test_acc_feat, test_std_feat), (test_acc5_feat, test_std5_feat)  = meta_test(model, meta_testloader,
                                                                              use_logit=False)
    test_time = time.time() - start
    print('Using layer before logits for embedding')
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(
        test_acc_feat, test_std_feat, test_time))
    print('test_acc_feat top 5: {:.4f}, test_std top 5: {:.4f}, time: {:.1f}'.format(
        test_acc5_feat, test_std5_feat, test_time))

    wandb.run.summary['test_acc'] = test_acc
    wandb.run.summary['test_std'] = test_std
    wandb.run.summary['test_acc5'] = test_acc5
    wandb.run.summary['test_std5'] = test_std5
    wandb.run.summary['test_acc_feat'] = test_acc_feat
    wandb.run.summary['test_std_feat'] = test_std_feat
    wandb.run.summary['test_acc5_feat'] = test_acc5_feat
    wandb.run.summary['test_std5_feat'] = test_std5_feat


def train(epoch, train_loader, model, criterion, optimizer, opt, lang_model, devocab=None):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    lang_losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):

        if opt.lsl:
            (input, target, _, (lang, lang_length, lang_mask)) = data
            # Trim padding to max length in batch
            max_lang_length = lang_length.max()
            lang = lang[:, :max_lang_length]
            lang_mask = lang_mask[:, :max_lang_length]

            # images = [(unnormalize_cub(im).cpu().numpy() * 255).astype(int).transpose(1,2,0) for im in input]
            # decoded = []
            # for cap, length in zip(lang, lang_length):
            #     decoded.append(' '.join([devocab[int(i)] for i in cap[:length]]))
            # wandb.log(
            #     {'samples': [wandb.Image(images[i], caption=decoded[i])
            #                  for i in range(min(10, len(decoded)))]
            #     }, step=epoch
            # )
            # print('saved')

        else:
            (input, target, _) = data

        # lang shape:        [batch_size, 32]
        # lang_length shape: [batch_size]
        # lang_mask shape:   [batch_size, 32]

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input  = input.cuda()
            target = target.cuda()
            if opt.lsl:
                lang = lang.cuda()
                lang_length = lang_length.cuda()
                lang_mask   = lang_mask.cuda()

        # ===================forward=====================
        if opt.use_logit:
            output = model(input)
            features = output
        else:
            features, output = model(input, is_feat=True)
            features = features[-1]
        loss = criterion(output, target)

        # add language loss
        if opt.lsl:
            lang_loss = compute_lang_loss(features, lang, lang_length, lang_mask, lang_model)
            lang_loss = opt.lang_lambda * lang_loss
            loss = lang_loss + loss

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        if opt.lsl:
            lang_losses.update(lang_loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  '{lang}'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5,
                      lang='Lang Loss {lang_losses.val:.4f} {lang_losses.avg:.4f}'.format(lang_losses=lang_losses)
                          if opt.lsl else None))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, lang_losses.avg if opt.lsl else None


if __name__ == '__main__':
    main()
