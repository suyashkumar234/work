"""Util functions
Extended from original PANet code
TODO: move part of dataset configurations to data_utils
"""
import random
import torch
import numpy as np
import operator

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CLASS_LABELS = {
    'SABS': {
        'pa_all': set( [1,2,3,6]  ),
        0: set([1,6]  ), # upper_abdomen: spleen + liver as training, kidneis are testing
        1: set( [2,3] ), # lower_abdomen
    },
    'C0': {
        'pa_all': set(range(1, 4)),
        0: set([2,3]),
        1: set([1,3]),
        2: set([1,2]),
    },
    'CHAOST2': {
        'pa_all': set(range(1, 5)),
        0: set([1, 4]), # upper_abdomen, leaving kidneies as testing classes
        1: set([2, 3]), # lower_abdomen
    },
}

# class TverskyLoss(nn.Module):
#     r"""Criterion that computes Tversky Coeficient loss.

#     According to [1], we compute the Tversky Coefficient as follows:

#     .. math::

#         \text{S}(P, G, \alpha; \beta) =
#           \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}

#     where:
#        - :math:`P` and :math:`G` are the predicted and ground truth binary
#          labels.
#        - :math:`\alpha` and :math:`\beta` control the magnitude of the
#          penalties for FPs and FNs, respectively.

#     Notes:
#        - :math:`\alpha = \beta = 0.5` => dice coeff
#        - :math:`\alpha = \beta = 1` => tanimoto coeff
#        - :math:`\alpha + \beta = 1` => F beta coeff

#     Shape:
#         - Input: :math:`(N, C, H, W)` where C = number of classes.
#         - Target: :math:`(N, H, W)` where each value is
#           :math:`0 ≤ targets[i] ≤ C−1`.

#     Examples:
#         >>> N = 5  # num_classes
#         >>> loss = tgm.losses.TverskyLoss(alpha=0.5, beta=0.5)
#         >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         >>> output = loss(input, target)
#         >>> output.backward()

#     References:
#         [1]: https://arxiv.org/abs/1706.05721
#     """

#     def __init__(self, alpha: float, beta: float, gamma: float) -> None:
#         super(TverskyLoss, self).__init__()
#         self.alpha: float = alpha
#         self.beta: float = beta
#         self.gamma: float = gamma
#         self.eps: float = 1e-6

#     def forward(
#             self,
#             input: torch.Tensor,
#             target: torch.Tensor) -> torch.Tensor:
#         if not torch.is_tensor(input):
#             raise TypeError("Input type is not a torch.Tensor. Got {}"
#                             .format(type(input)))
#         if not len(input.shape) == 4:
#             raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
#                              .format(input.shape))
#         if not input.shape[-2:] == target.shape[-2:]:
#             raise ValueError("input and target shapes must be the same. Got: {}"
#                              .format(input.shape, input.shape))
#         if not input.device == target.device:
#             raise ValueError(
#                 "input and target must be in the same device. Got: {}" .format(
#                     input.device, target.device))
#         # compute softmax over the classes axis
#         # input_soft = F.softmax(input, dim=1)

#         # create the labels one hot tensor
#         # target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

#         # compute the actual dice score
#         dims = (1, 2, 3)
#         intersection = (input * target).sum() #torch.sum(input * target, dims)
#         fps = (input * (1. - target)).sum()
#         fns = ((1. - input) * target).sum() 

#         numerator = intersection
#         denominator = intersection + self.alpha * fps + self.beta * fns
#         tversky_loss = numerator / (denominator + self.eps)
#         return torch.mean(torch.pow(1. - tversky_loss, self.gamma))

def get_tversky_loss(inp, target, alpha = 0.3, beta = 0.7, gamma = 1.0):
    if not torch.is_tensor(inp):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(inp)))
    if not len(inp.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(inp.shape))
    if not inp.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {}"
                         .format(inp.shape, inp.shape))
    if not inp.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}" .format(
                inp.device, target.device))
    # compute softmax over the classes axis
    # input_soft = F.softmax(input, dim=1)

    # create the labels one hot tensor
    # target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = (inp * target).sum() #torch.sum(input * target, dims)
    fps = (inp * (1. - target)).sum()
    fns = ((1. - inp) * target).sum() 

    numerator = intersection
    denominator = intersection + alpha * fps + beta * fns
    tversky_loss = numerator / (denominator + 1e-6)
    return torch.mean(torch.pow(1. - tversky_loss, gamma))

def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 0
    return fg_bbox, bg_bbox

def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()

def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)

def compose_wt_simple(is_wce, data_name):
    """
    Weights for cross-entropy loss
    """
    if is_wce:
        if data_name in ['SABS', 'SABS_Superpix', 'C0', 'C0_Superpix', 'CHAOST2', 'CHAOST2_Superpix']:
            return torch.FloatTensor([0.05, 1.0]).cuda()
        else:
            raise NotImplementedError
    else:
        return torch.FloatTensor([1.0, 1.0]).cuda()


class CircularList(list):
    """
    Helper for spliting training and validation scans
    Originally: https://stackoverflow.com/questions/8951020/pythonic-circular-list/8951224
    """
    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self[x] for x in self._rangeify(x)]

        index = operator.index(x)
        try:
            return super().__getitem__(index % len(self))
        except ZeroDivisionError:
            raise IndexError('list index out of range')

    def _rangeify(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)

