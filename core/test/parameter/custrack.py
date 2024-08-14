# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/7
# @Time        : 15:32
# @Description :
from core.test.utils import TrackerParams
import os
from core.test.evaluation.environment import env_settings
from core.config.custrack.config import cfg, update_config_from_file


def parameters(yaml_name: str,epoch=None,checkpoint=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'configs/custrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if epoch is not None:
        cfg.TEST.EPOCH = epoch
    if cfg.TEST.CUSTOM_CKPT_DIR:
        params.checkpoint = os.path.join(save_dir, f"checkpoints/train/custrack/%s/CUSTrack_ep%04d.pth.tar" %
                                         (cfg.TEST.CKPT_DIR, cfg.TEST.EPOCH))
    else:
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/custrack/%s/CUSTrack_ep%04d.pth.tar" %
                                         (yaml_name, cfg.TEST.EPOCH))
    if checkpoint is not None:
        params.checkpoint = checkpoint
    assert os.path.exists(params.checkpoint), f"checkpoints {params.checkpoint} not exits!"
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
