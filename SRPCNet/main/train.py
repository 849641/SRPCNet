#!/usr/bin/python3
#coding=utf-8

import os
import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import dataset
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.get_model import get_model
from Evaluate import Eval


# IoU Loss
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def train(Dataset, parser):
    args = parser.parse_args()
    _MODEL_ = args.model
    _DATASET_ = args.dataset
    _LR_ = args.lr
    _DECAY_ = args.decay
    _MOMEN_ = args.momen
    _BATCHSIZE_ = args.batchsize
    _EPOCH_ = args.epoch
    _LOSS_ = args.loss
    _SAVEPATH_ = args.savepath

    print(args)

    if _MODEL_ == "SRPC":
        cfg = Dataset.Config(datapath=_DATASET_, savepath=_SAVEPATH_, mode='train', batch=_BATCHSIZE_, lr=_LR_, momen=_MOMEN_, decay=_DECAY_, epoch=_EPOCH_)
    else:
        cfg = None
        print("_MODEL_ IS NOT FOUND.")

    data = Dataset.Data(cfg, _MODEL_)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=0)

    ## network
    net = get_model(cfg, _MODEL_)
    net.train(True)
    net.cuda()

    ## parameter
    base, head = [], []

    for name, param in net.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            pass
        elif 'encoder' in name:
            base.append(param)
        elif 'network' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    sw = SummaryWriter(cfg.savepath)

    global_step = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda(), mask.cuda()

            out1, out2, out3, out4, out5 = net(image)

            loss1 = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)
            loss2 = F.binary_cross_entropy_with_logits(out2, mask) + iou_loss(out2, mask)
            loss3 = F.binary_cross_entropy_with_logits(out3, mask) + iou_loss(out3, mask)
            loss4 = F.binary_cross_entropy_with_logits(out4, mask) + iou_loss(out4, mask)
            loss5 = F.binary_cross_entropy_with_logits(out5, mask) + iou_loss(out5, mask)

            loss = loss1 + loss2 + loss3 + loss4 + loss5

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            global_step += 1

            sw.add_scalars('loss', {'loss1': loss1.item(), 'loss2': loss2.item(), 'loss3': loss3.item(), 'loss4': loss4.item(), 'loss5': loss5.item(), 'loss': loss.item()}, global_step=global_step)

        print('%s | epoch:%d/%d | lr=%.6f | loss1=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f| loss5=%.6f | loss=%.6f' % (datetime.datetime.now(), epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'], loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss.item()))

        if (epoch + 1) % 200 == 0:
            torch.save(net.state_dict(), cfg.savepath + '/' + _MODEL_ + str(epoch + 1))
            # Evaluate
            eval_epoch = epoch + 1
            Eval(eval_epoch, dataset)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='SRPC')
    parser.add_argument("--dataset", default='../SSP2000/Train')
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--momen", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--loss", default='CPR')
    parser.add_argument("--savepath", default='../checkpoint/')
    train(dataset, parser)
