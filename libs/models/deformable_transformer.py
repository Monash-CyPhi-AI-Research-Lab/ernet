# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from mish_cuda import *
from libs.utils.misc import inverse_sigmoid
from libs.models.ops.modules.ms_deform_attn import MSDeformAttn

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DetAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(self, num_queries, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1., tokens_norm=False):
        super().__init__()
        self.num_queries = num_queries
        self.norm1 = norm_layer(dim)

        self.attn = DetAttn(num_queries,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        self.tokens_norm = tokens_norm

    def forward(self, x):
        x_norm1 = self.norm1(x)
        x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, self.num_queries:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, :self.num_queries]), x[:, self.num_queries:]], dim=1)
        x_res = x
        det_token = x[:, :self.num_queries]
        det_token = self.gamma2 * self.mlp(det_token)
        x = torch.cat([det_token, x[:, self.num_queries:]], dim=1)
        x = x_res + self.drop_path(x)
        return x

class DetAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, num_queries, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, :self.num_queries]).unsqueeze(1).reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_det = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_det = self.proj(x_det)
        x_det = self.proj_drop(x_det)

        return x_det

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_x, pos_y), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_x, pos_y, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class InteractionTransformer(nn.Module):

    def __init__(self, 
                d_model=256, 
                nhead=8, 
                num_encoder_layers=6,
                num_decoder_layers=6, 
                num_rel_decoder_layers=6,
                dim_feedforward=2048, 
                dropout=0.1,
                activation="relu", 
                num_feature_levels=5, dec_n_points=5,  enc_n_points=5,
                two_stage=False, two_stage_num_proposals=300, 
                two_stage_rel_num_proposals=100,
                Nd=100, Nr=100, 
                return_intermediate_dec=False,
                eta=1e-5
                ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.two_stage_rel_num_proposals = two_stage_rel_num_proposals
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # encoder of the backbone to refine feature sequence
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, 
                                                          enc_n_points,eta=eta)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        
        # interaction branch
        rel_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, 
                                                          dec_n_points,seq_len=Nr,eta=eta)
        # instance branch
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, 
                                                          dec_n_points,seq_len=Nd,eta=eta)
        
        # branch aggregation: instance-aware attention
        interaction_layer = InteractionLayer(d_model, d_model, dropout,eta=eta)

        self.decoder = InteractionTransformerDecoder(
            d_model,
            activation,
            decoder_layer,
            rel_decoder_layer,
            num_decoder_layers,
            interaction_layer,
            return_intermediate_dec)

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.rel_enc_output = nn.Linear(d_model, d_model)

            self.enc_output_norm = nn.LayerNorm(d_model)
            self.rel_enc_output_norm = nn.LayerNorm(d_model)
        else:
            self.hs_reference_points = nn.Linear(d_model, 2)
            self.rel_hs_reference_points = nn.Linear(d_model, 2)

        # [DET] and [REL] Token Generation
        self.det_token = nn.Parameter(torch.zeros(1, two_stage_num_proposals, d_model))
        self.rel_det_token = nn.Parameter(torch.zeros(1, two_stage_rel_num_proposals, d_model))
        self.det_pos_embed = nn.Parameter(torch.zeros(1, two_stage_num_proposals, d_model))
        self.rel_det_pos_embed = nn.Parameter(torch.zeros(1, two_stage_rel_num_proposals, d_model))        

        self.det_attn_blocks = nn.Sequential(*nn.ModuleList([
            DetAttentionBlock(two_stage_num_proposals,
                dim=d_model, num_heads=8, tokens_norm=True)
            for _ in range(2)]))

        self.rel_attn_blocks = nn.Sequential(*nn.ModuleList([
            DetAttentionBlock(two_stage_rel_num_proposals,
                dim=d_model, num_heads=8, tokens_norm=True)
            for _ in range(2)]))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.hs_reference_points.weight.data, gain=1.0)
            xavier_uniform_(self.rel_hs_reference_points.weight.data, gain=1.0)
            constant_(self.hs_reference_points.bias.data, 0.)
            constant_(self.rel_hs_reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 0.05
        proposals = []
        _cur = 0

        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * base_scale * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1) 
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory_base = memory
        output_memory_base = output_memory_base.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory_base = output_memory_base.masked_fill(~output_proposals_valid, float(0))

        # ReZero Memory
        output_memory = self.enc_output_norm(self.enc_output(output_memory_base))

        # # ReZero Memory
        rel_output_memory = self.rel_enc_output_norm(self.rel_enc_output(output_memory_base))

        return output_memory, rel_output_memory, output_proposals

    @torch.jit.export
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @torch.jit.ignore
    def forward(self, srcs, masks, query_embed, rel_query_embed, pos_embeds):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # refine the feature sequence using encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # [DET] and [REL] Token Generation
        bs, _, c = memory.shape
        det_token = self.det_token.expand(bs, -1, -1) + self.det_pos_embed.expand(bs, -1, -1)
        rel_token = self.rel_det_token.expand(bs, -1, -1) + self.rel_det_pos_embed.expand(bs, -1, -1)
        tokens = torch.cat([det_token, memory], dim=1)
        tgt = self.det_attn_blocks(tokens)[:, :self.two_stage_num_proposals]
        tokens = torch.cat([rel_token, memory], dim=1)
        rel_tgt = self.rel_attn_blocks(tokens)[:, :self.two_stage_rel_num_proposals]

        if self.two_stage:

            # Hack implementation for instance and interaction prediction heads 
            # of Efficient DETR
            enc_outputs_class, _ = self.decoder.class_embed[self.decoder.num_layers](tgt,self.training)
            rel_enc_outputs_class, _ = self.decoder.rel_class_embed[self.decoder.num_layers](rel_tgt,self.training)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](tgt,self.training) 
            rel_enc_outputs_coord_unact = self.decoder.rel_bbox_embed[self.decoder.num_layers](rel_tgt,self.training) 
            enc_rel_id_embed = self.decoder.rel_id_embed[self.decoder.num_layers](tgt,self.training)
            enc_src_embed = self.decoder.rel_src_embed[self.decoder.num_layers](rel_tgt,self.training)
            enc_dst_embed = self.decoder.rel_dst_embed[self.decoder.num_layers](rel_tgt,self.training)

            hs_reference_points = enc_outputs_coord_unact.sigmoid()
            hs_init_reference_out = hs_reference_points
            rel_hs_reference_points = rel_enc_outputs_coord_unact.sigmoid()
            rel_hs_init_reference_out = rel_hs_reference_points

        else:
            # hs
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            hs_reference_points = self.hs_reference_points(query_embed).sigmoid()
            hs_init_reference_out = hs_reference_points

            # rel_hs
            rel_query_embed, rel_tgt = torch.split(rel_query_embed, c, dim=1)
            rel_query_embed = rel_query_embed.unsqueeze(0).expand(bs, -1, -1)
            rel_tgt = rel_tgt.unsqueeze(0).expand(bs, -1, -1)
            rel_hs_reference_points = self.rel_hs_reference_points(rel_query_embed).sigmoid()
            rel_hs_init_reference_out = rel_hs_reference_points

        # decoder
        hs, rel_hs, hs_inter_references, rel_hs_inter_references = self.decoder(tgt, rel_tgt, hs_reference_points, rel_hs_reference_points,
                                                                      memory, spatial_shapes, level_start_index, 
                                                                      valid_ratios, mask_flatten)
        hs_inter_references_out = hs_inter_references
        rel_hs_inter_references_out = rel_hs_inter_references

        if self.two_stage:
            return (hs, hs_init_reference_out, hs_inter_references_out,
                  rel_hs, rel_hs_init_reference_out, rel_hs_inter_references_out,
                  enc_outputs_class, enc_outputs_coord_unact, 
                  rel_enc_outputs_class, rel_enc_outputs_coord_unact,
                  enc_rel_id_embed, enc_src_embed, enc_dst_embed)
        
        return (hs, hs_init_reference_out, hs_inter_references_out, rel_hs,
              rel_hs_init_reference_out, rel_hs_inter_references_out, 
              None, None, None, None, None, None, None)

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 eta=1e-5):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    @torch.jit.ignore
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod 
    @torch.jit.export
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @torch.jit.ignore
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 seq_len=100, eta=1e-5):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos): 
        return tensor if pos is None else tensor + pos 

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @torch.jit.ignore
    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1) 
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt) 

        return tgt

class InteractionLayer(nn.Module):
    def __init__(self, d_model, d_feature, dropout=0.1, eta=1e-5):
        super().__init__()
        self.d_feature = d_feature

        self.det_tfm = nn.Linear(d_model, d_feature)
        self.rel_tfm = nn.Linear(d_model, d_feature)
        self.det_value_tfm = nn.Linear(d_model, d_feature)

        self.rel_norm = nn.LayerNorm(d_model)

        if dropout is not None:
            self.dropout = dropout
            self.det_dropout = nn.Dropout(dropout)
            self.rel_add_dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, det_in, rel_in):
        det_in = det_in.transpose(0,1)
        rel_in = rel_in.transpose(0,1)
        det_attn_in = self.det_tfm(det_in)
        rel_attn_in = self.rel_tfm(rel_in)
        det_value = self.det_value_tfm(det_in)
        scores = torch.matmul(det_attn_in.transpose(0, 1),
            rel_attn_in.permute(1, 2, 0)) / math.sqrt(self.d_feature)
        det_weight = F.softmax(scores.transpose(1, 2), dim = -1)
        if self.dropout is not None:
          det_weight = self.det_dropout(det_weight)
        rel_add = torch.matmul(det_weight, det_value.transpose(0, 1))
        rel_out = rel_in.transpose(0, 1) + self.rel_add_dropout(rel_add)
        rel_out = self.rel_norm(rel_out)
        return det_in.transpose(0,1), rel_out

class InteractionTransformerDecoder(nn.Module):

    def __init__(self,
                 d_model,
                 activation,
                 decoder_layer,
                 rel_decoder_layer,
                 num_layers,
                 interaction_layer=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.rel_layers = _get_clones(rel_decoder_layer, num_layers)
        self.num_layers = num_layers
        if interaction_layer is not None:
            self.rel_interaction_layers = _get_clones(interaction_layer, num_layers)
        else:
            self.rel_interaction_layers = None
        self.return_intermediate = return_intermediate

        self.hs_ref_point = MLP(2 * d_model, d_model, d_model, 2, activation)
        self.hs_query_scale = MLP(d_model, d_model, d_model, 2, activation)
        self.rel_hs_ref_point_sbj = MLP(d_model, d_model, d_model, 2, activation)
        self.rel_hs_query_scale_sbj = MLP(d_model, d_model, d_model, 2, activation)
        self.rel_hs_ref_point_obj = MLP(d_model, d_model, d_model, 2, activation)
        self.rel_hs_query_scale_obj = MLP(d_model, d_model, d_model, 2, activation)

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.class_embed = None
        self.rel_class_embed = None
        self.bbox_embed = None
        self.rel_bbox_embed = None
        self.rel_id_embed = None
        self.rel_src_embed = None
        self.rel_dst_embed = None

    @torch.jit.ignore
    def forward(self, tgt, rel_tgt, 
                hs_reference_points, rel_hs_reference_points,
                src, src_spatial_shapes, src_level_start_index, 
                src_valid_ratios, src_padding_mask=None):

        output = tgt
        rel_output = rel_tgt

        intermediate = []
        rel_intermediate = []
        intermediate_reference_points = []
        rel_intermediate_reference_points = []

        for lid in range(self.num_layers):
            # hs
            if hs_reference_points.shape[-1] == 4:
                hs_reference_points_input = hs_reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert hs_reference_points.shape[-1] == 2
                hs_reference_points_input = hs_reference_points[:, :, None] * src_valid_ratios[:, None]
            
            query_sine_embed = gen_sineembed_for_position(hs_reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
            raw_query_pos = self.hs_ref_point(query_sine_embed) # bs, nq, 256
            pos_scale = self.hs_query_scale(output) if lid != 0 else 1
            query_pos = pos_scale * raw_query_pos

            output = self.layers[lid](output, query_pos, hs_reference_points_input, 
                        src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            
            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output,self.training)
                if hs_reference_points.shape[-1] == 4:
                    new_hs_reference_points = tmp + inverse_sigmoid(hs_reference_points)
                    new_hs_reference_points = new_hs_reference_points.sigmoid()
                else:
                    assert hs_reference_points.shape[-1] == 2
                    new_hs_reference_points = tmp
                    new_hs_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(hs_reference_points)
                    new_hs_reference_points = new_hs_reference_points.sigmoid()
                hs_reference_points = new_hs_reference_points.detach()          

            # rel hs
            if rel_hs_reference_points.shape[-1] == 4:
                rel_hs_reference_points_input = rel_hs_reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert rel_hs_reference_points.shape[-1] == 2
                rel_hs_reference_points_input = rel_hs_reference_points[:, :, None] * src_valid_ratios[:, None]
            
            query_sine_embed_sbj = gen_sineembed_for_position(rel_hs_reference_points_input[:, :, 0, :2]) # bs, nq, 256*2 
            raw_query_pos = self.rel_hs_ref_point_sbj(query_sine_embed_sbj) # bs, nq, 256
            pos_scale = self.rel_hs_query_scale_sbj(rel_output) if lid != 0 else 1
            rel_query_pos_sbj = pos_scale * raw_query_pos

            query_sine_embed_obj = gen_sineembed_for_position(rel_hs_reference_points_input[:, :, 0, 2:]) # bs, nq, 256*2 
            raw_query_pos = self.rel_hs_ref_point_obj(query_sine_embed_obj) # bs, nq, 256
            pos_scale = self.rel_hs_query_scale_obj(rel_output) if lid != 0 else 1
            rel_query_pos_obj = pos_scale * raw_query_pos

            rel_output_sbj = self.rel_layers[lid](rel_output, rel_query_pos_sbj, rel_hs_reference_points_input[...,:2], 
                        src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            
            rel_output_obj = self.rel_layers[lid](rel_output, rel_query_pos_obj, rel_hs_reference_points_input[...,2:], 
                        src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            rel_output = rel_output_sbj + rel_output_obj

            # hack implementation for iterative bounding box refinement
            if self.rel_bbox_embed is not None:
                tmp = self.rel_bbox_embed[lid](rel_output,self.training)
                if rel_hs_reference_points.shape[-1] == 4:
                    new_rel_hs_reference_points = tmp + inverse_sigmoid(rel_hs_reference_points)
                    new_rel_hs_reference_points = new_rel_hs_reference_points.sigmoid()
                else:
                    assert rel_hs_reference_points.shape[-1] == 2
                    new_rel_hs_reference_points = tmp
                    new_rel_hs_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(rel_hs_reference_points)
                    new_rel_hs_reference_points = new_rel_hs_reference_points.sigmoid()
                rel_hs_reference_points = new_rel_hs_reference_points.detach()

            # instance-aware attention module
            if self.rel_interaction_layers is not None:
                output, rel_output = self.rel_interaction_layers[lid](
                    output, rel_output
                )

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(hs_reference_points)
                rel_intermediate.append(rel_output)
                rel_intermediate_reference_points.append(rel_hs_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(rel_intermediate), torch.stack(intermediate_reference_points), torch.stack(rel_intermediate_reference_points)

        return output, rel_output, hs_reference_points, rel_hs_reference_points

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.actf = _get_activation_fn(activation)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.actf(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == 'mish':
        return MishCuda()
    if activation == 'tanhExp':
        return tanhExp()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_deformable_transformer(cfg):
    if cfg.TRANSFORMER.BRANCH_AGGREGATION is False:
        return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)
    else:
        return InteractionTransformer(
            d_model=cfg.TRANSFORMER.HIDDEN_DIM,
            dropout=cfg.TRANSFORMER.DROPOUT,
            nhead=cfg.TRANSFORMER.NHEADS,
            dim_feedforward=cfg.TRANSFORMER.DIM_FEEDFORWARD,
            num_encoder_layers=cfg.TRANSFORMER.ENC_LAYERS,
            num_decoder_layers=cfg.TRANSFORMER.DEC_LAYERS,
            num_rel_decoder_layers=cfg.TRANSFORMER.DEC_LAYERS,
            activation="mish",
            return_intermediate_dec=True,
            num_feature_levels=cfg.TRANSFORMER.NUM_FEATURE_LEVELS,
            dec_n_points=cfg.TRANSFORMER.DEC_N_POINTS,
            enc_n_points=cfg.TRANSFORMER.ENC_N_POINTS,
            two_stage=cfg.TRANSFORMER.TWO_STAGE,
            two_stage_num_proposals=cfg.TRANSFORMER.NUM_QUERIES,
            two_stage_rel_num_proposals=cfg.TRANSFORMER.REL_NUM_QUERIES,
            Nd = cfg.TRANSFORMER.NUM_QUERIES,
            Nr = cfg.TRANSFORMER.REL_NUM_QUERIES
        )

