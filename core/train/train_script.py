import json
import os
import sys

from loguru import logger
from core.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from core.train.trainers import LTRTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from .base_functions import *
from core.models.custrack import build_custrack
from core.train.actors import CUSTrackActor
import importlib

from ..utils.focal_loss import FocalLoss

def init_logger(path,local_rank):
    with open(path,'w',encoding='utf-8'):pass
    logger.remove(0)
    if local_rank == -1 or local_rank == 0:
        logger.configure(
            handlers=[
                dict(sink=sys.stderr, level="INFO"),
                dict(sink=path,
                     enqueue=True,
                     serialize=True,
                     diagnose=True,
                     backtrace=True,
                     rotation="1 hours",
                     level="INFO")
            ],
            extra={"common_to_all": "default"},
        )

    else:
        logger.configure(
            handlers=[
                dict(sink=sys.stderr, level="WARNING"),
                dict(sink=path,
                     enqueue=True,
                     serialize=True,
                     diagnose=True,
                     backtrace=True,
                     rotation="1 minutes",
                     level="WARNING")
            ],
            extra={"common_to_all": "default"},
        )


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("core.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    # update settings based on cfg
    update_settings(settings, cfg)
    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s/%s" % (settings.script_name, settings.config_name))
    os.makedirs(settings.log_file,exist_ok=True)
    settings.log_file = os.path.join(settings.log_file, "training.log")
    init_logger(settings.log_file,settings.local_rank)

    if settings.local_rank in [-1, 0]:
        for key in cfg.keys():
            config_value = json.dumps(cfg[key], indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
            logger.info("{} configuration: \n{}".format(key, config_value))
            logger.info('\n')

    # Build dataloaders
    loaders = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "custrack":
        net = build_custrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == 'custrack':
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0,
                       'kl':cfg.TRAIN.KL_WEIGHT,'u':cfg.TRAIN.U_BRANCH_WEIGHT,'t':cfg.TRAIN.T_BRANCH_WEIGHT,'k':cfg.TRAIN.K_BRANCH_WEIGHT,
                       'fusion': cfg.TRAIN.FUSION_WEIGHT
                       }
        actor = CUSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, loaders, optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
