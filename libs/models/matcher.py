import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from libs.utils.box_ops import box_cxcywh_to_xyxy, complete_box_iou, SCA_loss, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_ciou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_ciou: This is the relative weight of the ciou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_ciou = cost_ciou
        assert cost_class != 0 or cost_bbox != 0 or cost_ciou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs_dict, targets):
        """ Performs the matching

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        outputs = outputs_dict['pred_det']
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the ciou cost betwen boxes
        cost_ciou = -complete_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_ciou * cost_ciou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        
        if outputs_dict['pred_rel'] is None:
            indices_dict = {
                'det': indices,
                'rel': None
            }
            return indices_dict

        # for rel
        rel_outputs = outputs_dict['pred_rel']
        bs, rel_num_queries = rel_outputs["pred_logits"].shape[:2]
        rel_out_prob = rel_outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        rel_out_bbox = rel_outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        rel_tgt_ids = torch.cat([v["rel_labels"] for v in targets])
        rel_tgt_bbox = torch.cat([v["rel_vecs"] for v in targets])

        # interaction category semantic distance
        rel_cost_list = []

        for idx, r_tgt_id in enumerate(rel_tgt_ids):
            tgt_rel_id = torch.where(r_tgt_id == 1)[0]    

            # Asymmetric Loss
            gamma_pos = 0
            gamma_neg = 4
            m = 0.05
            p_m = rel_out_prob - m 
            p_m[torch.where(p_m<0)] = 0
            pos_cost_class = ((1 - rel_out_prob) ** gamma_pos) * (-(rel_out_prob + 1e-8).log())
            neg_cost_class = (p_m ** gamma_neg) * (-(1 - p_m + 1e-8).log())
            rel_cost_list.append((pos_cost_class[:, tgt_rel_id] - neg_cost_class[:, tgt_rel_id]).sum(dim=-1))

        rel_cost_class = torch.stack(rel_cost_list, dim=-1)

        # interaction vector location distance
        rel_cost_bbox = torch.cdist(rel_out_bbox, rel_tgt_bbox, p=1)

        # Final cost matrix
        rel_C = self.cost_bbox * rel_cost_bbox + self.cost_class * rel_cost_class
        rel_C = rel_C.view(bs, rel_num_queries, -1).cpu()

        rel_sizes = [len(v["rel_vecs"]) for v in targets]
        rel_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(rel_C.split(rel_sizes, -1))]
        rel_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in rel_indices]

        indices_dict = {
            'det': indices,
            'rel': rel_indices,
        }

        return indices_dict


def build_matcher(cfg):
    return HungarianMatcher(cost_class=cfg.MATCHER.COST_CLASS,
        cost_bbox=cfg.MATCHER.COST_BBOX, cost_ciou=cfg.MATCHER.COST_CIOU)

