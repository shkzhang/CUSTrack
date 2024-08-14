# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/17
# @Time        : 14:45
# @Description :
from functools import partial

import torch
from torch import nn

from core.models.custrack.backbone import BaseBackbone
from core.models.custrack.vit import Block


class TransformerEncode(BaseBackbone):
    def __init__(self, in_chans=256, input_num_tokens=3, depth=8, num_heads=8, drop_rate=0.,norm_layer=None, act_layer=None):
        super().__init__()
        self.num_features = in_chans  # num_features for consistency with other models
        self.input_num_tokens = input_num_tokens
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_chans,
            nhead=num_heads,
            dropout=drop_rate,
            dim_feedforward=4 * in_chans,
            batch_first = True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_chans,
            nhead=num_heads,
            dropout=drop_rate,
            dim_feedforward=4 * in_chans,
            batch_first=True

        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=in_chans)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=in_chans)

        self.norm = norm_layer(self.num_features)

    def encode_src(self, src):  # (B, L, C)
        src_start = src.permute(1, 0, 2)  # (L,B, C)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        ) # (B, L, 1)
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)  #(B, L, C) (L,B, C)
        src = src_start + pos_encoder
        src = self.encoder(src.permute(1, 0, 2)) + src_start.permute(1, 0, 2)
        return src

    def decode_trg(self, trg, memory):
        trg_start = trg.permute(1, 0, 2)
        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start
        trg_mask = self.gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg.permute(1, 0, 2), memory=memory, tgt_mask=trg_mask) + trg_start.permute(1, 0, 2)
        return out

    def gen_trg_mask(self, length, device):
        mask = torch.tril(torch.ones(length, length, device=device)) == 1

        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward_features(self, x, **kwargs):
        src, trg = x[:,:self.input_num_tokens,:],x[:,self.input_num_tokens:,:]
        src = self.encode_src(src)
        pred = self.decode_trg(trg=trg, memory=src)
        aux_dict = {"attn": None}
        x_feat = self.encode_src(x)
        return self.norm(x_feat),pred, aux_dict
    def forward(self, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x_feat,pred, aux_dict = self.forward_features(x)

        return x_feat,pred, aux_dict


def transformer_encoder(in_chans=768, input_num_tokens=3, depth=8, num_heads=8, mlp_ratio=4., qkv_bias=True):
    return TransformerEncode(in_chans = in_chans, input_num_tokens=input_num_tokens, depth=depth, num_heads=num_heads)


if __name__ == '__main__':
    x = torch.zeros(8, 4, 256)
    net = TransformerEncode()
    encode = net(x)
    print(encode[0].shape)
