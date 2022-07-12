import _init_paths
from numbers import Number
import os
import sys
import datetime
import pprint


from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import evaluate_cls_loc
from test_cam import val_loc_one_epoch

import json

import torch
from torch.utils.tensorboard import SummaryWriter

from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
import numpy as np
from re import compile

CUBV2=False

def creat_model(cfg, args):
    print('==> Preparing networks for baseline...')
    # use gpu
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    model = create_deit_model(
            cfg.MODEL.ARCH,
            pretrained=True,
            num_classes=cfg.DATA.NUM_CLASSES,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = {}

        for k, v in checkpoint['state_dict'].items():
            k_ = '.'.join(k.split('.')[1:])

            pretrained_dict.update({k_: v})

        model.load_state_dict(pretrained_dict)
        print('load pretrained ts-cam model.')
    optimizer = create_optimizer(args, model)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)
    print('Preparing networks done!')
    return device, model, optimizer, cls_criterion


def main():
    args = update_config()
    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join(cfg.BASIC.SAVE_ROOT,'ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_CAM-THR{}_BS{}_{}'.format(
        cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE, cfg.BASIC.TIME))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, test_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, optimizer, cls_criterion = creat_model(cfg, args)

    best_gtknown = 0
    best_top1_loc = 0
    update_train_step = 0
    update_val_step = 0
    opt_thred = -1
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS+1):
        adjust_learning_rate_normal(optimizer, epoch, cfg)
        update_train_step, loss_train, cls_top1_train, cls_top5_train = \
            train_one_epoch(train_loader, model, device, cls_criterion,
                            optimizer, epoch, writer, cfg, update_train_step)
        if CUBV2:
            eval_results = val_loc_one_epoch(val_loader, model, device, )
        else:
            eval_results = val_loc_one_epoch(test_loader, model, device, )
        eval_results['epoch'] = epoch
        with open(os.path.join(cfg.BASIC.SAVE_DIR, 'val.txt'), 'a') as val_file:
            val_file.write(json.dumps(eval_results))
            val_file.write('\n')        

        loc_gt_known = eval_results['GT-Known_top-1']
        thred = eval_results['det_optThred_thr_50.00_top-1']
        # if loc_top1_val > best_top1_loc:
        #     best_top1_loc = loc_top1_val
        #     torch.save({
        #         "epoch": epoch,
        #         'state_dict': model.state_dict(),
        #         'best_map': best_gtknown
        #     }, os.path.join(ckpt_dir, 'model_best_top1_loc.pth'))
        if loc_gt_known > best_gt