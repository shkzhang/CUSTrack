# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/7
# @Time        : 15:41
# @Description :
import math

from core.models.custrack import build_custrack
from core.test.tracker.basetracker import BaseTracker
import torch

from core.test.tracker.vis_utils import gen_visualization
from core.test.utils.hann import hann2d
from core.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from core.test.tracker.data_utils import Preprocessor
from core.utils.box_ops import clip_box
from core.utils.ce_utils import generate_mask_cond


class CUSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(CUSTrack, self).__init__(params)
        network = build_custrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        self.reset_history = True
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
                if not self.use_visdom:
                    self.save_dir = "debug"
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.init_state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
        self.reset_history = True

    def limit_state(self, update_state):
        self.state = update_state
        return

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=x_dict.tensors,
                ce_template_mask=self.box_mask_z,
                ce_keep_rate=None,
                reset_history=self.reset_history,
                branch=self.cfg.MODEL.CAUSAL.BRANCH
            )
            if self.reset_history:
                self.reset_history = False

        # add hann windows
        out_dict = out_dict
        pred_score_map = out_dict['score_map'][-1]
        response = self.output_window * pred_score_map
        pred_boxes = self.network.k_head.cal_bbox(response, out_dict['size_map'][-1], out_dict['offset_map'][-1])
        # pred_boxes = out_dict['pred_boxes'][-1]
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        update_state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        self.limit_state(update_state)
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_causal')
                self.visdom.register(out_dict['fusion']['score_map'].view(self.feat_sz, self.feat_sz), 'heatmap', 1,
                                     'score_map_fusion')
                self.visdom.register(out_dict['kp']['score_map'].view(self.feat_sz, self.feat_sz), 'heatmap', 1,
                                     'score_map_kp')
                self.visdom.register(out_dict['tp']['score_map'].view(self.feat_sz, self.feat_sz), 'heatmap', 1,
                                     'score_map_tp')
                self.visdom.register(out_dict['up']['score_map'].view(self.feat_sz, self.feat_sz), 'heatmap', 1,
                                     'score_map_up')
                self.visdom.register(out_dict['nde']['score_map'].view(self.feat_sz, self.feat_sz), 'heatmap', 1,
                                     'score_map_nde')

                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')
                kp_outs = out_dict['kp']
                if 'removed_indexes_s' in kp_outs and kp_outs['removed_indexes_s']:
                    if isinstance(kp_outs['removed_indexes_s'], list):
                        kp_outs['removed_indexes_s'] = kp_outs['removed_indexes_s'][-1]
                    removed_indexes_s = kp_outs['removed_indexes_s']
                    if removed_indexes_s is not None and removed_indexes_s[0] is not None:
                        removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in
                                             removed_indexes_s]
                        masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                        self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1,
                                             'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return CUSTrack
