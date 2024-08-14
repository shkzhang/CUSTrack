# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/8
# @Time        : 15:05
# @Description :
from . import BaseActor
from core.utils.misc import NestedTensor
from core.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from core.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap, generate_corner_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class CUSTrackActor(BaseActor):
    """ Actor for training CUSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass

        out_dicts = self.forward_pass(data)

        # compute losses

        return self.compute_losses(out_dicts, data)

    def forward_pass(self, data):
        assert len(data['template_images']) == 1
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)
        search_images = data['search_images']
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dicts = self.net(template=template_list,
                             search=search_images,
                             ce_template_mask=box_mask_z,
                             ce_keep_rate=ce_keep_rate,
                             return_last_attn=False, reset_history=data['reset_history'],
                             branch=self.cfg.MODEL.CAUSAL.BRANCH)

        return out_dicts

    def compute_losses(self, pred_outs, gt_dict, return_status=True):
        if self.cfg.MODEL.CAUSAL.BRANCH == 'UTP':
            return self.compute_losses_utp(pred_outs, gt_dict, return_status)
        elif self.cfg.MODEL.CAUSAL.BRANCH == 'TP':
            return self.compute_losses_tp(pred_outs, gt_dict, return_status)
        else:
            raise NotImplementedError

    def compute_losses_tp(self, pred_outs, gt_dict, return_status=True):
        if return_status:
            status = {}
            fusion_loss, fusion_status = self.compute_branch_losses(pred_outs['fusion'], gt_dict)
            for key, value in fusion_status.items():
                status[f'{key}'] = value
            # compute NDE KL
            if 'score_tl' in pred_outs:
                tl_te = pred_outs['fusion']['score_tl'].view(-1,
                                                             pred_outs['fusion']['score_tl'].shape[3] *
                                                             pred_outs['fusion']['score_tl'].shape[4])
                br_te = pred_outs['fusion']['score_br'].view(-1,
                                                             pred_outs['fusion']['score_br'].shape[3] *
                                                             pred_outs['fusion']['score_br'].shape[4])
                te = torch.cat([tl_te, br_te], dim=0)
                tl_nde = pred_outs['nde']['score_tl']
                br_nde = pred_outs['nde']['score_br']
                nde = torch.cat([tl_nde, br_nde], dim=0)
                nde = nde.view(-1, nde.shape[3] * nde.shape[4])
            elif 'score_map' in pred_outs:
                te = pred_outs['fusion']['score_map'].view(-1,
                                                           pred_outs['fusion']['score_map'].shape[3] *
                                                           pred_outs['fusion']['score_map'].shape[4])
                nde = pred_outs['nde']['score_map'].view(-1, pred_outs['nde']['score_map'].shape[3] *
                                                         pred_outs['nde']['score_map'].shape[4])
            else:
                raise Exception("Error key score key.")
            p_te = torch.nn.functional.softmax(te, -1).clone().detach()
            p_nde = torch.nn.functional.softmax(nde, -1)
            kl_loss = - p_te * p_nde.log()
            kl_loss = kl_loss.sum(1).mean()
            loss = (fusion_loss +
                    self.loss_weight['kl'] * kl_loss)

            causal_loss, causal_status = self.compute_branch_losses(pred_outs, gt_dict)
            for key, value in causal_status.items():
                status[f'causal/{key}'] = value
            status['Loss/kl'] = kl_loss.item()
            status['causal/Loss/total'] = causal_loss.item()
            status['Loss/total'] = loss.item()
            return loss, status

    def compute_losses_utp(self, pred_outs, gt_dict, return_status=True):
        if return_status:
            status = {}
            fusion_loss, fusion_status = self.compute_branch_losses(pred_outs['fusion'], gt_dict)
            tp_loss, tp_status = self.compute_branch_losses(pred_outs['tp'], gt_dict)
            up_loss, up_status = self.compute_branch_losses(pred_outs['up'], gt_dict)
            kp_loss, kp_status = self.compute_branch_losses(pred_outs['kp'], gt_dict)

            for key, value in fusion_status.items():
                if key == 'IoU': continue
                status[f'Fusion/{key}'] = value
            for key, value in tp_status.items():
                if key == 'IoU': continue
                status[f'TP/{key}'] = value
            for key, value in up_status.items():
                if key == 'IoU': continue
                status[f'UP/{key}'] = value
            for key, value in kp_status.items():
                if key == 'IoU': continue
                status[f'KP/{key}'] = value
            # compute NDE KL
            if 'score_tl' in pred_outs:
                tl_te = pred_outs['fusion']['score_tl'].view(-1,
                                                             pred_outs['fusion']['score_tl'].shape[3] *
                                                             pred_outs['fusion']['score_tl'].shape[4])
                br_te = pred_outs['fusion']['score_br'].view(-1,
                                                             pred_outs['fusion']['score_br'].shape[3] *
                                                             pred_outs['fusion']['score_br'].shape[4])
                te = torch.cat([tl_te, br_te], dim=0)
                tl_nde = pred_outs['nde']['score_tl']
                br_nde = pred_outs['nde']['score_br']
                nde = torch.cat([tl_nde, br_nde], dim=0)
                nde = nde.view(-1, nde.shape[3] * nde.shape[4])
            elif 'score_map' in pred_outs:
                te = pred_outs['fusion']['score_map'].view(-1,
                                                           pred_outs['fusion']['score_map'].shape[3] *
                                                           pred_outs['fusion']['score_map'].shape[4])
                nde = pred_outs['nde']['score_map'].view(-1, pred_outs['nde']['score_map'].shape[3] *
                                                         pred_outs['nde']['score_map'].shape[4])
            else:
                raise Exception("Error key score key.")
            p_te = torch.nn.functional.softmax(te, -1).clone().detach()
            p_nde = torch.nn.functional.softmax(nde, -1)
            kl_loss = - p_te * p_nde.log()
            kl_loss = kl_loss.sum(1).mean()
            loss = (self.loss_weight['fusion']*fusion_loss + self.loss_weight['k']*kp_loss +
                    self.loss_weight['t'] * tp_loss +
                    self.loss_weight['u'] * up_loss +
                    self.loss_weight['kl'] * kl_loss)

            causal_loss, causal_status = self.compute_branch_losses(pred_outs, gt_dict)
            for key, value in causal_status.items():
                if key == 'IoU': continue
                status[f'Causal/{key}'] = value

            status['Loss/kl'] = kl_loss.item()
            status['Loss/total'] = loss.item()
            status['Loss/tp'] = tp_loss.item()
            status['Loss/up'] = up_loss.item()
            status['Loss/kp'] = kp_loss.item()
            status['Loss/fusion'] = fusion_loss.item()

            status['IoU/up'] = up_status['IoU']
            status['IoU/fusion'] = fusion_status['IoU']
            status['IoU/causal'] = causal_status['IoU']
            status['IoU/kp'] = kp_status['IoU']

            return loss, status

    def compute_branch_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = gt_dict['search_anno'].view(-1, 4)
        # Get boxes
        loss = torch.tensor(0.0).to(gt_bbox.device)
        status = {}
        if 'pred_boxes' in pred_dict:
            pred_boxes = pred_dict['pred_boxes'].view(-1, 4)
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).view(-1, 4).clamp(min=0.0,
                                                                       max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            loss += self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
            status["Loss/giou"] = giou_loss.item()
            status["Loss/l1"] = l1_loss.item()
            mean_iou = iou.detach().mean()
            status["IoU"] = mean_iou.item()

        # compute location loss
        if 'score_tl' in pred_dict:
            # gt gaussian map
            gt_gaussian_maps_tl, gt_gaussian_maps_br = generate_corner_heatmap(gt_bbox, self.cfg.DATA.SEARCH.SIZE,
                                                                               self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps_tl = gt_gaussian_maps_tl.unsqueeze(1)
            gt_gaussian_maps_br = gt_gaussian_maps_br.unsqueeze(1)
            score_map = pred_dict['score_tl']
            score_map = score_map.view(-1, 1, pred_dict['score_tl'].shape[3], pred_dict['score_tl'].shape[4])
            location_loss = self.objective['focal'](score_map, gt_gaussian_maps_tl)

            score_map = pred_dict['score_br']
            score_map = score_map.view(-1, 1, pred_dict['score_br'].shape[3], pred_dict['score_br'].shape[4])
            location_loss += self.objective['focal'](score_map, gt_gaussian_maps_br)
        elif 'score_map' in pred_dict:
            gt_gaussian_maps = generate_heatmap(gt_bbox, self.cfg.DATA.SEARCH.SIZE,
                                                self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps = gt_gaussian_maps.unsqueeze(1)
            score_map = pred_dict['score_map']
            score_map = score_map.view(-1, 1, pred_dict['score_map'].shape[3], pred_dict['score_map'].shape[4])
            location_loss = self.objective['focal'](score_map, gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=gt_bbox.device)
            # weighted sum
        loss += self.loss_weight['focal'] * location_loss
        status["Loss/location"] = location_loss.item()
        if return_status:
            status['Loss/total'] = loss.item()
            return loss, status
        else:
            return loss
