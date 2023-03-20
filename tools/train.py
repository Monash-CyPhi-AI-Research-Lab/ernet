# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-7-24
# ------------------------------------------------------------------------------
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import argparse
import os

import numpy as np
import random

import pprint
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import _init_paths
from configs import cfg
from configs import update_config

from libs.datasets.collate import collect
from libs.utils import misc
from libs.utils.utils import create_logger
from libs.utils.utils import get_model
from libs.utils.utils import get_dataset
from libs.utils.utils import get_trainer
from libs.utils.utils import load_checkpoint
from libs.utils.utils import get_lr_scheduler

from madgrad import MADGRAD
from lion import Lion

def parse_args():
    parser = argparse.ArgumentParser(description='HOI Transformer Task')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/fcos_detector.yaml',
        required=True,
        type=str)    
    # default distributed training
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='if use distribute train')

    parser.add_argument(
        '--dist-url',
        dest='dist_url',
        default='tcp://10.5.38.36:23456',
        type=str,
        help='url used to set up distributed training')
    parser.add_argument(
        '--world-size',
        dest='world_size',
        default=1,
        type=int,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--rank',
        default=0,
        type=int,
        help='node rank for distributed training, machine level')
    
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
          
    return args

def get_ip(ip_addr):
    ip_list = ip_addr.split('-')[2:6]
    for i in range(4):
        if ip_list[i][0] == '[':
            ip_list[i] = ip_list[i][1:].split(',')[0]
    return f'tcp://{ip_list[0]}.{ip_list[1]}.{ip_list[2]}.{ip_list[3]}:23456'

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main_per_worker():
    args = parse_args()

    update_config(cfg, args)
    ngpus_per_node = torch.cuda.device_count()

    print(cfg.OUTPUT_ROOT)
    if 'SLURM_PROCID' in os.environ.keys():
        proc_rank = int(os.environ['SLURM_PROCID'])
        local_rank = proc_rank % ngpus_per_node
        args.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        proc_rank = 0
        local_rank = 0
        args.world_size = 1

    args.distributed = (args.world_size > 1 or args.distributed)
    
    #create logger
    if proc_rank == 0:
        logger, output_dir = create_logger(cfg, proc_rank)

    # distribution
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            print('Not using distributed mode')
            args.distributed = False
            return

        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=cfg.DIST_BACKEND, init_method=args.dist_url,
                                            world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)

        # torch seed
        seed = cfg.SEED + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = torch.device(cfg.DEVICE)
        model, criterion, postprocessors = get_model(cfg, device)  
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=True
        )
        
    else:
        assert proc_rank == 0, ('proc_rank != 0, it will influence '
                                'the evaluation procedure')
        # torch seed
        seed = cfg.SEED
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if cfg.DEVICE == 'cuda':
            torch.cuda.set_device(local_rank)
        device = torch.device(cfg.DEVICE)
        model, criterion, postprocessors = get_model(cfg, device)  
        model = torch.nn.DataParallel(model).to(device)
    
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params:', n_parameters)

     # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            # "lr": cfg.TRAIN.LR_BACKBONE,
        },
    ] 

    if cfg.TRAIN.OPTIMIZER == 'lion':
        optimizer = Lion(param_dicts, lr=cfg.TRAIN.LR,
                        weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    elif cfg.TRAIN.OPTIMIZER == 'madgrad':
        optimizer = MADGRAD(param_dicts, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, 
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY, eps=1e-6)
        
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                        weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    print('Using the {} optimizer'.format(optimizer.__class__.__name__))
    
    lr_scheduler = get_lr_scheduler(cfg, optimizer, cfg.TRAIN.MAX_EPOCH)

    last_iter=-1

    step=0
    if cfg.TRAIN.RESUME:
        model, optimizer, lr_scheduler, last_iter = load_checkpoint(cfg, model,
            optimizer, lr_scheduler, device)
        step = np.where(np.greater_equal(last_iter,np.array(cfg.TRAIN.PL_STAGE))==True)[0][-1]

    train_dataset, eval_dataset = get_dataset(cfg, step)
    train_sampler = None if not args.distributed else torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
                  train_dataset,
                  batch_size=cfg.DATASET.IMG_NUM_PER_GPU,
                  shuffle=(train_sampler is None),
                  drop_last=True,
                  collate_fn=collect,
                  num_workers=cfg.WORKERS,
                  pin_memory=True,
                  sampler=train_sampler
                  )

    eval_loader = torch.utils.data.DataLoader(
                  eval_dataset,
                  batch_size=cfg.DATASET.IMG_NUM_PER_GPU,
                  shuffle=False,
                  drop_last=False,
                  collate_fn=collect,
                  num_workers=cfg.WORKERS
                  )

    Trainer = get_trainer(
        cfg,
        model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        postprocessors=postprocessors,
        log_dir='output',
        performance_indicator='mAP',
        last_iter=last_iter,
        rank=args.rank,
        device=device,
        max_norm=cfg.TRAIN.CLIP_MAX_NORM
    )

    print('Start training...')

    while True:            
        Trainer.train(train_loader, eval_loader, step)

if __name__ == '__main__':
    main_per_worker()
