# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/8
# @Time        : 14:57
# @Description :
"""
Basic CUSTrack model.
"""
from loguru import logger
import math
import os
import time
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from core.models.custrack.transformer import transformer_encoder
from core.models.layers.attn import CrossAttention
from core.models.layers.grad import grad_mul_const
from core.models.layers.head import build_box_head, MLP
from core.models.custrack.vit import vit_base_patch16_224
from core.models.custrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from core.models.layers.patch_embed import PatchEmbed
from core.utils.box_ops import box_xyxy_to_cxcywh


def _sigmoid(x):
    y = torch.clamp(torch.sigmoid_(x), min=1e-4, max=1 - 1e-4)
    return y

class CUSTrack(nn.Module):
    """ This is the base class for CUSTrack """

    def __init__(self, transformer, t_model, k_head, u_head, aux_loss=False, head_type="CORNER",
                 max_history_length=5,
                 feature_token_length=1,assumption='Learnable',graph='full'):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.t_model = t_model
        self.k_head = k_head
        self.u_head = u_head

        self.feature_token_length = feature_token_length
        self.max_history_length = max_history_length
        self.assumption = assumption
        self.graph = graph
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(k_head.feat_sz)
            self.feat_len_s = int(k_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.feat_token_mlp = MLP(self.backbone.num_features, 128, self.feature_token_length, 2)
        # History prompt
        self.history_prompt = []
        if self.head_type == "CORNER":
            self.cu_tl = nn.Parameter(torch.zeros(1, 1, self.u_head.feat_sz, self.u_head.feat_sz))
            self.ck_tl = nn.Parameter(torch.zeros(1, 1, self.k_head.feat_sz, self.k_head.feat_sz))
            self.cu_br = nn.Parameter(torch.zeros(1, 1, self.u_head.feat_sz, self.u_head.feat_sz))
            self.ck_br = nn.Parameter(torch.zeros(1, 1, self.k_head.feat_sz, self.k_head.feat_sz))
        elif self.head_type == 'CENTER':
            self.cu_score = nn.Parameter(torch.zeros(1, 1, self.u_head.feat_sz, self.u_head.feat_sz))
            self.cu_size = nn.Parameter(torch.zeros(1, 1, self.u_head.feat_sz, self.u_head.feat_sz))
            self.cu_offset = nn.Parameter(torch.zeros(1, 1, self.u_head.feat_sz, self.u_head.feat_sz))
            self.ck_score = nn.Parameter(torch.zeros(1, 1, self.k_head.feat_sz, self.k_head.feat_sz))
            self.ck_size = nn.Parameter(torch.zeros(1, 1, self.k_head.feat_sz, self.k_head.feat_sz))
            self.ck_offset = nn.Parameter(torch.zeros(1, 1, self.k_head.feat_sz, self.k_head.feat_sz))
        self.constant_input_x_feat = nn.Parameter(
            torch.ones(1, self.backbone.num_features, self.k_head.feat_sz * self.k_head.feat_sz))
        self.constant_input_t_feat = nn.Parameter(
            torch.ones(1, self.max_history_length, self.k_head.feat_sz * self.k_head.feat_sz))
        self.prompt_padding = nn.Parameter(torch.zeros(1, 1,self.k_head.feat_sz * self.k_head.feat_sz))



    def feat_transform(self, feat_u=None, feat_t=None):  # (B, WH, C) (B, N1, WH)
        if feat_u is None:  # T
            feat_x = self.constant_input_x_feat
        else:
            feat_x = feat_u[:, -self.feat_len_s:]
        if feat_t is None:  # U
            feat_t = self.constant_input_t_feat
        return torch.mean(feat_t, dim=1,keepdim=True).transpose(1, 2) * feat_x

    def forward(self, branch='UTP', **kwargs
                ):
        if branch == 'UTP':
            return self.forward_utp(**kwargs)
        elif branch == 'TP':
            return self.forward_tp(**kwargs)
        else:
            raise NotImplementedError

    def forward_utp(self, template: torch.Tensor,
                    search: torch.Tensor,
                    ce_template_mask=None,
                    ce_keep_rate=None,
                    return_last_attn=False,
                    reset_history=True
                    ):
        if reset_history:
            self.reset_history()
        if len(search.shape) == 4:
            search = search.unsqueeze(0)
        kp_outs = {}
        tp_outs = {}
        up_outs = {}
        causal_outs = {}
        fusion_outs = {}
        nde_outs = {}
        for search_img in search:
            # Ecode U
            x_u, aux_dict = self.backbone(z=template, x=search_img,
                                          ce_template_mask=ce_template_mask,
                                          ce_keep_rate=ce_keep_rate,
                                          return_last_attn=return_last_attn)
            # ******** 1. UP Model ********
            feat_last = x_u[:, -self.feat_len_s:]
            if isinstance(feat_last, list):
                feat_last = feat_last[-1]
            out_kp = self.forward_head(self.u_head, feat_last, None)
            out_kp.update(aux_dict)
            out_kp['backbone_feat'] = x_u
            self.fusion_out(kp_outs, out_kp)

            # ******** 2. TP Model ********
            padding_length = (self.max_history_length - len(self.history_prompt))
            if padding_length > 0:  # padding
                padding_prompt = (self.prompt_padding *
                                  torch.ones(template.shape[0], padding_length, self.prompt_padding.shape[-1]).to(
                                      self.prompt_padding.device))
                if len(self.history_prompt) >= 1:
                    _history_prompt = torch.cat(self.history_prompt, dim=1).clone().detach()
                    _history_prompt = torch.cat([padding_prompt, _history_prompt], dim=1)
                else:
                    _history_prompt = padding_prompt
            else:
                _history_prompt = torch.cat(self.history_prompt, dim=1).clone().detach()

            x_t, pred, aux_dict = self.t_model(_history_prompt)
            effect = pred.view(-1,1,self.k_head.feat_sz,self.k_head.feat_sz)
            out_tp = {'score_map':_sigmoid(effect) ,'effect_map':effect}
            out_tp.update(aux_dict)
            out_tp['backbone_feat'] = x_t
            self.fusion_out(tp_outs, out_tp)

            # ******** 3. KP Model ********
            x_kp = self.feat_transform(x_u, x_t)
            feat_last = x_kp
            if isinstance(feat_last, list):
                feat_last = feat_last[-1]
            out_up = self.forward_head(self.k_head, feat_last, None)
            out_up.update(aux_dict)
            out_up['backbone_feat'] = feat_last
            self.fusion_out(up_outs, out_up)

            # ******** 4. causal fusion ********
            causal_out, fusion_dict, nde_dict = self._causal_fusion(out_kp, out_up, out_tp)

            self.fusion_out(causal_outs, causal_out)
            self.fusion_out(fusion_outs, fusion_dict)
            self.fusion_out(nde_outs, nde_dict)
            self.update_history(x_kp, causal_out)

        causal_outs['kp'] = kp_outs
        causal_outs['tp'] = tp_outs
        causal_outs['up'] = up_outs
        causal_outs['nde'] = nde_outs
        causal_outs['fusion'] = fusion_outs
        return causal_outs

    def fusion_out(self, fusion_dict, frame_dict):
        for key, value in frame_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.unsqueeze(0)
                if key not in fusion_dict:
                    fusion_dict[key] = value
                else:
                    fusion_dict[key] = torch.cat([fusion_dict[key], value], dim=0)
            else:
                if key not in fusion_dict:
                    fusion_dict[key] = []
                fusion_dict[key].append(frame_dict[key])

    def _causal_fusion(self, kp_out, up_out, tp_out):
        causal_dict = {}
        if self.head_type == "CORNER":
            # Full causal
            fusion_dict = {
                'score_tl': _sigmoid(kp_out['score_tl'] + up_out['score_tl'] + tp_out['score_tl']),
                'score_br': _sigmoid(kp_out['score_br'] + up_out['score_br'] + tp_out['score_br'])
            }
            fusion_pred_boxes = self.k_head.cal_bbox(fusion_dict['score_tl'], fusion_dict['score_br'])
            outputs_coord = box_xyxy_to_cxcywh(fusion_pred_boxes)
            outputs_coord_new = outputs_coord.view(-1, 1, 4)
            fusion_dict['pred_boxes'] = outputs_coord_new

            # Full causal - TP causal (TE-NDE)
            nde_fusion_dict = {
                'score_tl': _sigmoid(self.ck_tl * torch.ones_like(tp_out['score_tl']).to(self.ck_tl.device)
                                     + self.cu_tl * torch.ones_like(tp_out['score_tl']).to(self.cu_tl.device)
                                     + tp_out['score_tl']),
                'score_br': _sigmoid(self.ck_br * torch.ones_like(tp_out['score_br']).to(self.ck_br.device)
                                     + self.cu_br * torch.ones_like(tp_out['score_br']).to(self.cu_br.device)
                                     + tp_out['score_br']),
            }
            causal_dict['score_tl'] = _sigmoid(fusion_dict['score_tl'] - nde_fusion_dict['score_tl'])
            causal_dict['score_br'] = _sigmoid(fusion_dict['score_br'] - nde_fusion_dict['score_br'])
            fusion_pred_boxes = self.k_head.cal_bbox(causal_dict['score_tl'], causal_dict['score_br'])
            outputs_coord = box_xyxy_to_cxcywh(fusion_pred_boxes)
            outputs_coord_new = outputs_coord.view(-1, 1, 4)
            causal_dict['pred_boxes'] = outputs_coord_new
        elif self.head_type == "CENTER":
            # Full causal (TE)
            have_ip = 1
            if self.graph == 'Simple_IP' or self.graph == 'Simple_All':
                have_ip = 0

            te_score_map = _sigmoid(kp_out['effect_map'] + up_out['effect_map']*have_ip + tp_out['effect_map'])

            fusion_dict = {
                'score_map': te_score_map,
                'size_map': kp_out['size_map'],
                'offset_map': kp_out['offset_map'],
            }
            fusion_pred_boxes = self.k_head.cal_bbox(te_score_map, fusion_dict['size_map'],
                                                     fusion_dict['offset_map'])
            outputs_coord_new = fusion_pred_boxes.view(-1, 1, 4)
            fusion_dict['pred_boxes'] = outputs_coord_new

            # NDE

            if self.assumption == 'Learnable':
                nde_fusion_dict = {
                    'score_map': ((self.ck_score * torch.ones_like(kp_out['effect_map']).to(self.ck_score.device))
                                  + (have_ip*self.cu_score * torch.ones_like(up_out['effect_map']).to(self.cu_score.device))
                                  + tp_out['effect_map'].detach().clone())
                }
            elif self.assumption == 'Random':
                nde_fusion_dict = {
                    'score_map': ((torch.rand_like(kp_out['effect_map']).to(self.ck_score.device))
                                  + (have_ip*torch.rand_like(up_out['effect_map']).to(self.cu_score.device))
                                  + tp_out['effect_map'].detach().clone())
                }
            elif self.assumption == 'Uniform':
                nde_fusion_dict = {
                    'score_map': ((torch.ones_like(kp_out['effect_map']).to(self.ck_score.device))
                                  + (have_ip*torch.ones_like(up_out['effect_map']).to(self.cu_score.device))
                                  + tp_out['effect_map'].detach().clone())
                }
            else:
                raise NotImplementedError

            # Full causal - TP causal (TIE = TE - NDE)
            with torch.no_grad():
                if self.graph == 'Simple_Sub' or self.graph == 'Simple_All':
                    causal_dict['score_map'] = _sigmoid(kp_out['effect_map'] + up_out['effect_map'] *have_ip).detach().clone()
                else:
                    causal_dict['score_map'] =(te_score_map - nde_fusion_dict['score_map']).detach().clone()
                causal_dict['size_map'] = fusion_dict['size_map'].detach().clone()
                causal_dict['offset_map'] = fusion_dict['offset_map'].detach().clone()
                fusion_pred_boxes = self.k_head.cal_bbox(causal_dict['score_map'], causal_dict['size_map'],
                                                         causal_dict['offset_map'])
                outputs_coord_new = fusion_pred_boxes.view(-1, 1, 4)
                causal_dict['pred_boxes'] = outputs_coord_new
        else:
            raise NotImplementedError
        return causal_dict, fusion_dict, nde_fusion_dict


    def forward_head(self, box_head, enc_opt, gt_score_map=None, get_bbox=True):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C,Nq)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, prob_vec_tl, prob_vec_br = box_head(opt_feat, return_dist=True, softmax=False)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_tl': prob_vec_tl,
                   'score_br': prob_vec_br
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            if get_bbox:
                score_map_ctr, bbox, size_map, offset_map,score_map_effect = box_head(opt_feat, gt_score_map)
                outputs_coord = bbox
                outputs_coord_new = outputs_coord.view(bs, Nq, 4)
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'effect_map':score_map_effect,
                       'size_map': size_map,
                       'offset_map': offset_map}
            else:
                score_map_ctr,score_map_effect = box_head(opt_feat, gt_score_map)
                out = {'score_map': score_map_ctr,'effect_map':score_map_effect}
            return out
        else:
            raise NotImplementedError

    def reset_history(self):
        self.history_prompt = []

    def update_history(self, feature, result):
        if self.head_type == "CORNER":
            score_map = (result['score_tl'].clone()).detach() + (result['score_br'].clone()).detach()
        elif self.head_type == "CENTER":
            score_map = (result['score_map'].clone()).detach()
        else:
            raise NotImplementedError
        prompt_token_score = score_map.view(-1,1,self.feat_sz_s*self.feat_sz_s)
        if len(self.history_prompt) >= 2:
            self.history_prompt.pop(0)
        self.history_prompt.append(prompt_token_score)


def build_custrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('CUSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    k_head = build_box_head(cfg, hidden_dim)
    u_head = build_box_head(cfg, hidden_dim)
    t_model = transformer_encoder(in_chans=k_head.feat_sz*k_head.feat_sz,
                                  input_num_tokens=cfg.MODEL.MAX_HISTORY_LENGTH - 1,
                                  depth=3,
                                  num_heads=8,
                                  )
    model = CUSTrack(
        backbone,
        t_model,
        k_head,
        u_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        max_history_length=cfg.MODEL.MAX_HISTORY_LENGTH,
        feature_token_length=cfg.MODEL.FEATURE_TOKEN_LENGTH,
        assumption=cfg.MODEL.CAUSAL.ASSUMPTION,
        graph=cfg.MODEL.CAUSAL.GRAPH,
    )

    if 'CUSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        logger.info('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
