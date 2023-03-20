import mmcv
import numpy as np
import os
import sys
import time
import math
import copy

import torch
from torch import nn
import torch.nn.functional as F
from mish_cuda import *

from scipy.spatial.distance import cdist
from libs.models.backbone import build_backbone
from libs.models.matcher import build_matcher
from libs.models.deformable_transformer_coco import build_deformable_transformer
from libs.utils import box_ops
from libs.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                             accuracy, get_world_size, interpolate,
                             is_dist_avail_and_initialized, inverse_sigmoid)

# Visualization
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class BCNorm(nn.Module):
    def __init__(self, num_channels, num_groups, eps):
        super(BCNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_groups, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_groups, 1))
        self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, inp):
        out = self.bn(inp)
        out = out.view(1, inp.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, self.eps, True)
        out = out.view(inp.size(0), self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.view_as(inp)
        return out

class ASNet(nn.Module):
    """ This is the HOI Transformer module that performs HOI detection """
    def __init__(self, 
                 backbone, 
                 transformer, 
                 num_classes=dict(
                     obj_labels=91,
                 ), 
                 num_queries=100,
                 num_feature_levels=3,
                 id_emb_dim=8,
                 aux_loss=False,
                 with_box_refine=False, 
                 two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: dict of number of sub clses, obj clses and relation clses, 
                         omitting the special no-object category
                         keys: ["obj_labels", "rel_labels"]
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.backbone = backbone
        self.num_feature_levels = num_feature_levels
        self.transformer = transformer
        hidden_dim = transformer.d_model  

        # instance branch
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes['obj_labels']+1, 3, 0.1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3, 0.0)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    # BCNorm(hidden_dim,32,1e-5),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    # BCNorm(hidden_dim,32,1e-5),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([ 
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    # BCNorm(hidden_dim,32,1e-5),
                    nn.GroupNorm(32, hidden_dim),
                )])

        # aux loss of each decoder layer
        self.aux_loss = aux_loss

        # bounding box refinement
        self.with_box_refine = with_box_refine

        # bool two-stage 
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.layers[-1].bias.data = torch.ones(num_classes['obj_labels'] + 1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers

        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed

        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])  
            self.transformer.decoder.bbox_embed = None

        
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # backbone
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        (hs, hs_init_reference, hs_inter_references, 
        enc_outputs_class, enc_outputs_coord_unact) = self.transformer(srcs, masks, query_embeds, pos)

        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            # hs
            if lvl == 0:
                reference = hs_init_reference
            else:
                reference = hs_inter_references[lvl - 1]
            
            reference = inverse_sigmoid(reference)
            outputs_class, outputs_class_var = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # Visualize outputs
        # with torch.no_grad():
        #     b=0
        #     img_h = samples.tensors[b].size(1)
        #     img_w = samples.tensors[b].size(2)
        #     img_xywh = torch.Tensor([img_w, img_h, img_w, img_h])
        #     plt.imshow(samples.tensors[b].permute(1,2,0).cpu().numpy())

        #     for i in range(outputs_coord[-1].size(1)):
        #         box = outputs_coord[-1][b][i].cpu()*img_xywh
        #         # rect = patches.Rectangle((box[0]-box[2]/2,box[1]-box[3]/2), box[2], box[3], linewidth=1, edgecolor='y', facecolor='none')
        #         # plt.gca().add_patch(rect)
        #         plt.plot(box[0],box[1], 'y',markersize=5, marker = 'o')
        #     plt.show()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'pred_logits_var': outputs_class_var}
      
        output = {
            'pred_det': out,
            'pred_rel': None
        }      

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
            output['enc_outputs'] =  {
                                      'pred_det': out,
                                      'pred_rel': None
                                  }   

        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return output  

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        aux_output = []
        for idx in range(len(outputs_class)):
            out = {'pred_logits': outputs_class[idx], 'pred_boxes': outputs_coord[idx]}
            aux_output.append({
                'pred_det': out,
                'pred_rel': None
            })
        return aux_output


class SetCriterion(nn.Module):
    """ This class computes the loss for HOI Transformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  
                 matcher, 
                 losses,
                 weight_dict,
                 eos_coef,
                 rel_eos_coef=0.1,
                 num_classes=dict(
                     obj_labels=90,
                 ),
                 neg_act_id=0):
        """ Create the criterion.
        Parameters:
            num_classes: dict of number of sub clses, obj clses and relation clses, 
                         omitting the special no-object category
                         keys: ["obj_labels", "rel_labels"]
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes['obj_labels']
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def loss_labels(self, outputs_dict, targets, indices_dict, num_boxes_dict, log=True,
                            alpha=0.25, gamma=2, loss_reduce='sum'):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_det' in outputs_dict
        outputs = outputs_dict['pred_det']
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        indices = indices_dict['det']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # Cross-Entropy
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # losses = {'loss_ce': loss_ce}   
        
        # Focal Loss
        num_boxes = num_boxes_dict['det']
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)
            loss = alpha_t * loss

        loss_ce = loss.mean(1).sum() / num_boxes * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}   

        # Asymmetric Loss
        # num_boxes = num_boxes_dict['det']
        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # target_classes_onehot = target_classes_onehot[...,:-1]

        # gamma_neg = 4
        # gamma_pos = 0
        # clip = 0.05
        # disable_torch_grad_focal_loss = True
        # eps = 1e-8

        # # Calculating Probabilities
        # src_logits = src_logits[...,:-1]
        # x_sigmoid = torch.sigmoid(src_logits)
        # xs_pos = x_sigmoid
        # xs_neg = 1 - x_sigmoid

        # # Asymmetric Clipping
        # if clip is not None and clip > 0:
        #     xs_neg = (xs_neg + clip).clamp(max=1)

        # # Basic CE calculation
        # los_pos = target_classes_onehot * torch.log(xs_pos.clamp(min=eps))
        # los_neg = (1 - target_classes_onehot) * torch.log(xs_neg.clamp(min=eps))
        # loss = los_pos + los_neg

        # # Asymmetric Focusing
        # if gamma_neg > 0 or gamma_pos > 0:
        #     if disable_torch_grad_focal_loss:
        #         torch.set_grad_enabled(False)
        #     pt0 = xs_pos * target_classes_onehot
        #     pt1 = xs_neg * (1 - target_classes_onehot)  # pt = p if t > 0 else 1-p
        #     pt = pt0 + pt1
        #     one_sided_gamma = gamma_pos * target_classes_onehot + gamma_neg * (1 - target_classes_onehot)
        #     one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        #     if disable_torch_grad_focal_loss:
        #         torch.set_grad_enabled(True)
        #     loss *= one_sided_w
        # loss_ce = -loss.sum() / num_boxes * src_logits.shape[1]
        # losses = {'loss_ce': loss_ce} 

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs_dict, targets, indices_dict, num_boxes_dict):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        assert 'pred_det' in outputs_dict
        outputs = outputs_dict['pred_det']
        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs_dict, targets, indices_dict, num_boxes_dict):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the CIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_det' in outputs_dict
        outputs = outputs_dict['pred_det']
        assert 'pred_boxes' in outputs

        indices = indices_dict['det']
        num_boxes = num_boxes_dict['det']
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        loss_ciou = box_ops.SCA_loss(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        losses['loss_ciou'] = loss_ciou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_neg_permutation_idx(self, neg_indices):
        # permute neg rel predictions following indices
        batch_idx = torch.cat([torch.full_like(neg_ind, i) for i, neg_ind in enumerate(neg_indices)])
        neg_idx = torch.cat([neg_ind for neg_ind in neg_indices])
        return batch_idx, neg_idx

    def get_loss(self, loss, outputs_dict, targets, indices_dict, num_boxes_dict, **kwargs):
        loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'boxes': self.loss_boxes
            }
        if loss not in loss_map:
            return {}
        return loss_map[loss](outputs_dict, targets, indices_dict, num_boxes_dict, **kwargs)


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        indices_dict = self.matcher(outputs, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
            device=next(iter(outputs['pred_det'].values())).device)
        
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        num_boxes_dict = {
            'det': num_boxes,
        }

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets,
                                        indices_dict, num_boxes_dict))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs.keys():
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_dict = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dict, num_boxes_dict, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:    
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            # for bt in bin_targets:
            #     bt['labels'] = torch.zeros_like(bt['labels'])
            indices_dict = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices_dict, num_boxes_dict, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.actf = MishCuda()
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.resweight = nn.Parameter(torch.zeros(len(self.layers)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.output_dim = output_dim
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.layers[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        for i in range(len(self.layers)-2):
            torch.nn.init.xavier_normal_(self.layers[i+1].weight, gain=torch.sqrt(torch.tensor(2.)))

    def forward(self, x):
        if self.dropout_rate > 0.0:
            n = 10
            x_out = torch.empty(size=([n]+list(x.size()[:-1])+[self.output_dim]))

            for ensemble in range(n):
                for i, layer in enumerate(self.layers):
                    # Non-ReZero
                    # x = self.dropout(self.actf(layer(x))) if i < self.num_layers - 1 else layer(x)
                    
                    # ReZero - Uncertainty
                    if i == 0:
                        x_inter = self.dropout(self.actf(layer(x)))
                    elif i < self.num_layers - 1: 
                        x_inter = x_inter + self.dropout(self.actf(layer(x_inter))) * self.resweight[i] 
                    else:
                        x_inter = layer(x_inter)
                        x_out[ensemble] = x_inter
            
            # Visualize plots
            # fig=plt.figure(figsize=(10,7))
            # ax=fig.add_subplot(111)
            # plt.ylim([0,1])
            # plt.xlim([0, self.output_dim+1])
            # ax.set_title('Predictive Uncertainty Estimation of Interactions')
            # ax.set_xlabel('10 Interaction Classes')
            # ax.set_ylabel('Probabilities')
            # ax.boxplot(x_out.sigmoid().detach().clone().numpy()[:,0,0,:])
            # plt.show()

            x_out = x_out.to(x.device)
            x = torch.mean(x_out,dim=0)
            x_var = torch.var(x_out.sigmoid(),dim=0)
            return x, x_var

        else:
            for i, layer in enumerate(self.layers):
                # Non-ReZero
                # x = self.dropout(self.actf(layer(x))) if i < self.num_layers - 1 else layer(x)
                
                # ReZero - Uncertainty
                if i == 0:
                    x = self.dropout(self.actf(layer(x)))
                elif i < self.num_layers - 1: 
                    x = x + self.dropout(self.actf(layer(x))) * self.resweight[i] 
                else:
                    x = layer(x)
            return x

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self,
                 rel_array_path,
                 use_emb=False):
        super().__init__()
        # use semantic embedding in the matching or not
        self.use_emb = use_emb
        # rel array to remove non-exist hoi categories in training
        self.rel_array_path = rel_array_path
        
    def get_matching_scores(self, s_cetr, o_cetr, s_scores, o_scores, rel_vec,
                            s_emb, o_emb, src_emb, dst_emb): 
        rel_s_centr = rel_vec[..., :2].unsqueeze(-1).repeat(1, 1, s_cetr.shape[0])
        rel_o_centr = rel_vec[..., 2:].unsqueeze(-1).repeat(1, 1, o_cetr.shape[0])
        s_cetr = s_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)
        s_scores = s_scores.repeat(rel_vec.shape[0], 1)
        o_cetr = o_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)
        o_scores = o_scores.repeat(rel_vec.shape[0], 1)
        dist_s_x = abs(rel_s_centr[..., 0, :] - s_cetr[..., 0])
        dist_s_y = abs(rel_s_centr[..., 1, :] - s_cetr[..., 1])
        dist_o_x = abs(rel_o_centr[..., 0, :] - o_cetr[..., 0])
        dist_o_y = abs(rel_o_centr[..., 1, :] - o_cetr[..., 1])
        dist_s = (1.0 / (dist_s_x + 1.0)) * (1.0 / (dist_s_y + 1.0))
        dist_o = (1.0 / (dist_o_x + 1.0)) * (1.0 / (dist_o_y + 1.0))
        # involving emb into the matching strategy
        if self.use_emb is True:
            s_emb_np = s_emb.data.cpu().numpy()
            o_emb_np = o_emb.data.cpu().numpy()
            src_emb_np = src_emb.data.cpu().numpy()
            dst_emb_np = dst_emb.data.cpu().numpy()
            dist_s_emb = torch.from_numpy(cdist(src_emb_np, s_emb_np, metric='euclidean')).to(rel_vec.device)
            dist_o_emb = torch.from_numpy(cdist(dst_emb_np, o_emb_np, metric='euclidean')).to(rel_vec.device)
            dist_s_emb = 1. / (dist_s_emb + 1.0)
            dist_o_emb = 1. / (dist_o_emb + 1.0)
            dist_s *= dist_s_emb
            dist_o *= dist_o_emb
        dist_s = dist_s * s_scores
        dist_o = dist_o * o_scores
        return dist_s, dist_o

    @torch.no_grad()
    def forward(self, outputs_dict, file_name, target_sizes,
                rel_topk=20, sub_cls=1):
        """ Perform the matching of postprocess to generate final predicted HOI triplets
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        outputs = outputs_dict['pred_det']
        # '(bs, num_queries,) bs=1
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        id_emb = outputs['id_emb'].flatten(0, 1)
        rel_outputs = outputs_dict['pred_rel']
        rel_out_logits, rel_out_var, rel_out_bbox = rel_outputs['pred_logits'], \
            rel_outputs['pred_logits_var'], rel_outputs['pred_boxes']
        src_emb, dst_emb = rel_outputs['src_emb'].flatten(0, 1), \
            rel_outputs['dst_emb'].flatten(0, 1)
        assert len(out_logits) == len(target_sizes) == len(rel_out_logits) \
                == len(rel_out_bbox)
        assert target_sizes.shape[1] == 2
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        # parse instance detection results
        out_bbox = out_bbox * scale_fct[:, None, :]
        out_bbox_flat = out_bbox.flatten(0, 1)
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        labels_flat = labels.flatten(0, 1) # '(bs * num_queries, )
        scores_flat = scores.flatten(0, 1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox_flat)
        s_idx = torch.where(labels_flat==sub_cls)[0]
        o_idx = torch.arange(0, len(labels_flat)).long()
        # no detected human or object instances
        if len(s_idx) == 0 or len(o_idx) == 0:
            pred_out = {
                'file_name': file_name,
                'hoi_prediction': [],
                'predictions': []
            }
            return pred_out
        s_cetr = box_ops.box_xyxy_to_cxcywh(boxes[s_idx])[..., :2]
        o_cetr = box_ops.box_xyxy_to_cxcywh(boxes[o_idx])[..., :2]
        s_boxes, s_clses, s_scores = boxes[s_idx], labels_flat[s_idx], scores_flat[s_idx]
        o_boxes, o_clses, o_scores = boxes[o_idx], labels_flat[o_idx], scores_flat[o_idx]
        s_emb, o_emb = id_emb[s_idx], id_emb[o_idx]

        # parse interaction detection results
        rel_prob = rel_out_logits.sigmoid()
        topk = rel_prob.shape[-1]
        rel_scores = rel_prob.flatten(0, 1)
        hoi_vars = rel_out_var.flatten(0,1)
        hoi_labels  = torch.arange(0, topk).repeat(rel_scores.shape[0], 1).to(
            rel_prob.device) + 1
        rel_vec = rel_out_bbox * scale_fct[:, None, :]
        rel_vec_flat = rel_vec.flatten(0, 1)

        # matching distance in post-processing
        dist_s, dist_o = self.get_matching_scores(s_cetr, o_cetr, s_scores,
            o_scores, rel_vec_flat, s_emb, o_emb, src_emb, dst_emb)
        rel_s_scores, rel_s_ids = torch.max(dist_s, dim=-1)
        rel_o_scores, rel_o_ids = torch.max(dist_o, dim=-1)
        hoi_scores = rel_scores * s_scores[rel_s_ids].unsqueeze(-1) * \
            o_scores[rel_o_ids].unsqueeze(-1)

        # exclude non-exist hoi categories of training
        rel_array = torch.from_numpy(np.load(self.rel_array_path)).to(hoi_scores.device)
        valid_hoi_mask = rel_array[o_clses[rel_o_ids], 1:]
        hoi_scores = (valid_hoi_mask * hoi_scores).reshape(-1, 1)
        hoi_vars = (valid_hoi_mask * hoi_vars).reshape(-1, 1)
        hoi_labels = hoi_labels.reshape(-1, 1)
        rel_s_ids = rel_s_ids.unsqueeze(-1).repeat(1, topk).reshape(-1, 1)
        rel_o_ids = rel_o_ids.unsqueeze(-1).repeat(1, topk).reshape(-1, 1)
        hoi_triplet = (torch.cat((rel_s_ids.float(), rel_o_ids.float(), hoi_labels.float(),
            hoi_scores.float(), hoi_vars.float()), 1)).cpu().numpy()
        hoi_triplet = hoi_triplet[hoi_triplet[..., -2]>0.0]

        # remove repeated triplets
        hoi_triplet = hoi_triplet[np.argsort(-hoi_triplet[:,-2])]
        _, hoi_id = np.unique(hoi_triplet[:, [0, 1, 2]], axis=0, return_index=True)
        rel_triplet = hoi_triplet[hoi_id]
        # rel_triplet = rel_triplet[np.argsort(-rel_triplet[:,-2])]
        rel_triplet = rel_triplet[np.argsort(-rel_triplet[:,-1])]

        # save topk hoi triplets
        rel_topk = min(rel_topk, len(rel_triplet))
        rel_triplet = rel_triplet[:rel_topk]
        hoi_labels, hoi_scores = rel_triplet[..., 2], rel_triplet[..., 3]
        rel_s_ids, rel_o_ids = np.array(rel_triplet[..., 0], dtype=np.int64), np.array(rel_triplet[..., 1], dtype=np.int64)
        sub_boxes, obj_boxes = s_boxes.cpu().numpy()[rel_s_ids], o_boxes.cpu().numpy()[rel_o_ids]
        sub_clses, obj_clses = s_clses.cpu().numpy()[rel_s_ids], o_clses.cpu().numpy()[rel_o_ids]
        sub_scores, obj_scores = s_scores.cpu().numpy()[rel_s_ids], o_scores.cpu().numpy()[rel_o_ids]
        self.end_time = time.time()
        
        # wtite to files
        pred_out = {}
        pred_out['file_name'] = file_name
        pred_out['hoi_prediction'] = []
        num_rel = len(hoi_labels)
        for i in range(num_rel):
            sid = i
            oid = i + num_rel
            hoi_dict = {
                'subject_id': sid,
                'object_id': oid,
                'category_id': hoi_labels[i],
                'score': hoi_scores[i]
            }
            pred_out['hoi_prediction'].append(hoi_dict)
        pred_out['predictions'] = []
        for i in range(num_rel):
            det_dict = {
                'bbox': sub_boxes[i],
		        'category_id': sub_clses[i],
                'score': sub_scores[i]
            }
            pred_out['predictions'].append(det_dict)
        for i in range(num_rel):
            det_dict = {
                'bbox': obj_boxes[i],
		        'category_id': obj_clses[i],
                'score': obj_scores[i]
            }
            pred_out['predictions'].append(det_dict)
        return pred_out 


def build_model(cfg, device):
    backbone = build_backbone(cfg)
    transformer = build_deformable_transformer(cfg)
    num_classes=dict(
        obj_labels=cfg.DATASET.OBJ_NUM_CLASSES,
    )
    model = ASNet(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=cfg.TRANSFORMER.NUM_QUERIES,
        num_feature_levels=cfg.TRANSFORMER.NUM_FEATURE_LEVELS,
        aux_loss=cfg.LOSS.AUX_LOSS,
        with_box_refine=cfg.TRANSFORMER.WITH_BOX_REFINE,
        two_stage=cfg.TRANSFORMER.TWO_STAGE 
    )
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.LOSS.DET_CLS_COEF[0], 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF[0]}
    weight_dict['loss_ciou'] = cfg.LOSS.CIOU_LOSS_COEF[0]
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.TRANSFORMER.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(matcher=matcher, losses=losses, weight_dict=weight_dict,
                             eos_coef=cfg.LOSS.EOS_COEF, num_classes=num_classes)
    criterion.to(device)
    postprocessors = PostProcess(cfg.TEST.REL_ARRAY_PATH, cfg.TEST.USE_EMB)
    return model, criterion, postprocessors