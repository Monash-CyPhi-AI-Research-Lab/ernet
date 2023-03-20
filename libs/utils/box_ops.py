import torch
import math
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    w = w.clamp(0)
    h = h.clamp(0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def bbox_overlaps(boxes1, boxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of boxes1 and boxes2, otherwise the ious between each aligned pair of
    boxes1 and boxes2.

    Args:
        boxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        boxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> boxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> boxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(boxes1, boxes2)
        tensor([[0.5238, 0.0500, 0.0041],
                [0.0323, 0.0452, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']

    rows = boxes1.size(0)
    cols = boxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return boxes1.new(rows, 1) if is_aligned else boxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [rows, 2]
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (
            boxes1[:, 3] - boxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (
                boxes2[:, 3] - boxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (
            boxes1[:, 3] - boxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (
                boxes2[:, 3] - boxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if (boxes1[:, 2:] < boxes1[:, :2]).any():
        import pdb; pdb.set_trace()
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area 

def complete_box_iou(boxes1, boxes2): 
    """
    Complete IoU from https://arxiv.org/abs/2005.03572

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """

    # degenerate boxes gives inf / nan results
    # so do an early check
    if (boxes1[:, 2:] < boxes1[:, :2]).any():
        import pdb; pdb.set_trace()
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    boxes1_cxcywh = box_xyxy_to_cxcywh(boxes1)
    boxes2_cxcywh = box_xyxy_to_cxcywh(boxes2)

    # Compute S - Overlap Area
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0) 
    area = wh[:, :, 0] * wh[:, :, 1]

    # Compute D - Distance
    c_l = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    c_r = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    c_t = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    c_b = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    inter_diag = (boxes1_cxcywh[:,None,0] - boxes2_cxcywh[:,0])**2 + (boxes1_cxcywh[:,None,1]  - boxes2_cxcywh[:,1] )**2
    c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2
    D = inter_diag / c_diag

    # Compute V - Aspect Ratio
    ar_diff = torch.atan(boxes1_cxcywh[:,None,2] / boxes1_cxcywh[:,None,3]) - torch.atan(boxes2_cxcywh[:,2] / boxes2_cxcywh[:,3])
    V = (4 / (math.pi ** 2)) * torch.pow((-ar_diff), 2)

    # Compute CIoU Loss
    with torch.no_grad():
        S = (iou>0.5).float()
        alpha= S*V/(1-iou+V)
    cious = iou - D - alpha * V
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    return cious

def SCA_loss(boxes1, boxes2): 
    # degenerate boxes gives inf / nan results
    # so do an early check
    if (boxes1[:, 2:] < boxes1[:, :2]).any():
        import pdb; pdb.set_trace()

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # Calculate intersection
    xI1 = torch.max(boxes1[:, None,0], boxes2[:,0])
    xI2 = torch.min(boxes1[:, None,2], boxes2[:,2])
    yI1 = torch.max(boxes1[:, None,1], boxes2[:,1])
    yI2 = torch.min(boxes1[:, None,3], boxes2[:,3])
    w_min = xI2 - xI1
    h_min = yI2 - yI1

    # Smallest enclosing box
    xC1 = torch.min(boxes1[:, None,0], boxes2[:,0])
    xC2 = torch.max(boxes1[:, None,2], boxes2[:,2])
    yC1 = torch.min(boxes1[:, None,1], boxes2[:,1])
    yC2 = torch.max(boxes1[:, None,3], boxes2[:,3])
    w_max = xC2 - xC1 
    h_max = yC2 - yC1

    # Side Overlap (SO) Loss
    SO_loss = 2 - (h_min/h_max + w_min/w_max)

    # Two corner distance points
    D_lt = torch.pow(boxes1[:, None,0]-boxes2[:,0],2) + torch.pow(boxes1[:, None,1]-boxes2[:,1],2)
    D_rb = torch.pow(boxes1[:, None,2]-boxes2[:,2],2) + torch.pow(boxes1[:, None,3]-boxes2[:,3],2)

    # Euclidean distance of smallest enclosing box
    D_diag = torch.pow(w_max-w_min,2) + torch.pow(h_max-h_min,2)

    # Corner Distance (CD) Loss
    CD_loss = D_lt/D_diag + D_rb/D_diag

    return SO_loss + 0.2*CD_loss

def distance_vec(vec1, vec2):
    # euclidean distance 
    vec1xy = torch.sqrt((vec1[:,0] - vec2[:,0])**2 + (vec1[:,1] - vec2[:,1])**2 + 1e-8)
    vec2xy = torch.sqrt((vec1[:,2] - vec2[:,2])**2 + (vec1[:,3] - vec2[:,3])**2 + 1e-8)
    d = torch.clamp(vec1xy+vec2xy,min=0)

    len_vec1 = torch.sqrt((vec1[:,0] - vec1[:,2])**2 + (vec1[:,1] - vec1[:,3])**2 + 1e-8)
    len_vec2 = torch.sqrt((vec2[:,0] - vec2[:,2])**2 + (vec2[:,1] - vec2[:,3])**2 + 1e-8)
    l = torch.clamp(torch.abs(len_vec1-len_vec2),min=0)
    return d + l

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
