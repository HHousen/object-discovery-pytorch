import torch
from torch.nn.functional import one_hot

from scipy.optimize import linear_sum_assignment


def binarise_masks(pred_masks, num_classes=None):
    assert len(pred_masks.shape) == 5
    assert pred_masks.shape[2] == 1
    num_classes = num_classes or pred_masks.shape[1]
    return (
        one_hot(pred_masks.argmax(axis=1), num_classes=num_classes)
        .transpose(-1, -2)
        .transpose(-2, -3)
        .transpose(-3, -4)
        .to(bool)
    )


def convert_masks(pred_masks, true_masks, correct_bg=True):
    if correct_bg:
        pred_masks = torch.cat(
            [1.0 - pred_masks.sum(axis=1, keepdims=True), pred_masks], dim=1
        )
    num_classes = max(pred_masks.shape[1], true_masks.shape[1])
    pred_masks = binarise_masks(pred_masks, num_classes=num_classes)
    true_masks = binarise_masks(true_masks, num_classes=num_classes)
    # if torch.any(torch.isnan(pred_masks)) or torch.any(torch.isinf(pred_masks)): import ipdb; ipdb.set_trace()
    # if torch.any(torch.isnan(true_masks)) or torch.any(torch.isinf(true_masks)): import ipdb; ipdb.set_trace()
    return pred_masks, true_masks


def iou_matching(pred_masks, true_masks, threshold=1e-2):
    """The order of true_masks is preserved up to potentially missing background dim (in shic """
    assert pred_masks.shape[0] == true_masks.shape[0], "Batchsize mismatch"
    # true_masks = true_masks.to(torch.bool)
    assert (
        pred_masks.dtype == true_masks.dtype
    ), f"Dtype mismatch ({pred_masks.dtype}!={true_masks.dtype})"

    if pred_masks.dtype != torch.bool:
        pred_masks, true_masks = convert_masks(pred_masks, true_masks, correct_bg=False)
    assert pred_masks.shape[-3:] == true_masks.shape[-3:], "Mask shape mismatch"

    pred_masks = pred_masks.to(float)
    true_masks = true_masks.to(float)

    tspec = dict(device=pred_masks.device)
    iou_matrix = torch.zeros(
        pred_masks.shape[0], pred_masks.shape[1], true_masks.shape[1], **tspec
    )
    true_masks_sums = true_masks.sum((-1, -2, -3))
    pred_masks_sums = pred_masks.sum((-1, -2, -3))

    for pi in range(pred_masks.shape[1]):
        pandt = (pred_masks[:, pi : pi + 1] * true_masks).sum((-1, -2, -3))
        port = pred_masks_sums[:, pi : pi + 1] + true_masks_sums
        iou_matrix[:, pi] = (pandt + 1e-2) / (port + 1e-2)
        iou_matrix[pred_masks_sums[:, pi] == 0.0, pi] = 0.0

    for ti in range(true_masks.shape[1]):
        iou_matrix[true_masks_sums[:, ti] == 0.0, :, ti] = 0.0

    cost_matrix = iou_matrix.cpu().detach().numpy()
    inds = torch.zeros(
        pred_masks.shape[0],
        2,
        min(pred_masks.shape[1], true_masks.shape[1]),
        dtype=torch.int64,
        **tspec,
    )
    ious = torch.zeros(
        pred_masks.shape[0],
        min(pred_masks.shape[1], true_masks.shape[1]),
        dtype=float,
        **tspec,
    )
    for bi in range(cost_matrix.shape[0]):
        col_ind, row_ind = linear_sum_assignment(cost_matrix[bi].T, maximize=True)
        inds[bi, 0] = torch.tensor(row_ind, **tspec)
        inds[bi, 1] = torch.tensor(col_ind, **tspec)
        ious[bi] = torch.tensor(cost_matrix[bi, row_ind, col_ind], **tspec)
    # if torch.any(torch.isnan(inds)) or torch.any(torch.isinf(inds)): import ipdb; ipdb.set_trace()
    # if torch.any(torch.isnan(ious)) or torch.any(torch.isinf(ious)): import ipdb; ipdb.set_trace()
    return inds, ious


def align_masks_iou(
    pred_mask, true_mask, pred_mask_vis=None, true_mask_vis=None, has_bg=False
):
    pred_mask, true_mask = convert_masks(pred_mask, true_mask, correct_bg=not has_bg)
    inds, ious = iou_matching(pred_mask, true_mask)

    # Reindex the masks into alighned order
    B, S, *s = pred_mask.shape
    bias = S * torch.arange(B, device=pred_mask.device)[:, None]
    pred_mask = pred_mask.reshape(B * S, *s)[(bias + inds[:, 0]).flatten()].view(
        B, S, *s
    )
    true_mask = true_mask.reshape(B * S, *s)[(bias + inds[:, 1]).flatten()].view(
        B, S, *s
    )

    ret = pred_mask, true_mask, ious

    if pred_mask_vis is not None:
        if pred_mask_vis.dtype != torch.bool:
            pred_mask_vis = pred_mask_vis > 0.5

        if has_bg:
            pred_mask_vis = torch.cat(
                [
                    pred_mask_vis,
                    torch.zeros(
                        B,
                        S - pred_mask_vis.shape[1],
                        dtype=pred_mask_vis.dtype,
                        device=pred_mask_vis.device,
                    ),
                ],
                axis=1,
            )
        else:
            pred_mask_vis = torch.cat(
                [
                    torch.ones(
                        B, 1, dtype=pred_mask_vis.dtype, device=pred_mask_vis.device
                    ),
                    pred_mask_vis,
                    torch.zeros(
                        B,
                        S - pred_mask_vis.shape[1] - 1,
                        dtype=pred_mask_vis.dtype,
                        device=pred_mask_vis.device,
                    ),
                ],
                axis=1,
            )
        pred_mask_vis = pred_mask_vis.reshape(B * S)[
            (bias + inds[:, 0]).flatten()
        ].view(B, S)
        ret += (pred_mask_vis,)

    if true_mask_vis is not None:
        if true_mask_vis.dtype != torch.bool:
            true_mask_vis = true_mask_vis > 0.5

        true_mask_vis = torch.cat(
            [
                true_mask_vis,
                torch.zeros(
                    B,
                    S - true_mask_vis.shape[1],
                    dtype=true_mask_vis.dtype,
                    device=true_mask_vis.device,
                ),
            ],
            axis=1,
        )
        true_mask_vis = true_mask_vis.reshape(B * S)[
            (bias + inds[:, 1]).flatten()
        ].view(B, S)
        ret += (true_mask_vis,)
    # for i,r in enumerate(ret):
    #     if torch.any(torch.isnan(r)) or torch.any(torch.isinf(r)):
    #         print('/tStopcon:', i)
    #         import ipdb; ipdb.set_trace()
    return ret


def dices(pred_mask, true_mask):
    dice = (
        2
        * (pred_mask * true_mask).sum((-3, -2, -1))
        / (pred_mask.sum((-3, -2, -1)) + true_mask.sum((-3, -2, -1)))
    )
    dice = torch.where(torch.isnan(dice) | torch.isinf(dice), 0.0, dice.to(float))
    # if torch.any(torch.isnan(dice)) or torch.any(torch.isinf(dice)): import ipdb; ipdb.set_trace()
    return dice


# def ari(pred_mask, true_mask, skip_0=False):
#     B = pred_mask.shape[0]
#     pm = pred_mask.to(int).argmax(axis=1).squeeze().view(B, -1).cpu().detach().numpy()
#     tm = true_mask.to(int).argmax(axis=1).squeeze().view(B, -1).cpu().detach().numpy()
#     aris = []
#     for bi in range(B):
#         t = tm[bi]
#         p = pm[bi]
#         if skip_0:
#             p = p[t > 0]
#             t = t[t > 0]
#         ari_score = adjusted_rand_score(t, p)
#         aris.append(ari_score)
#     aris = torch.tensor(np.array(aris), device=pred_mask.device)
#     # if torch.any(torch.isnan(aris)) or torch.any(torch.isinf(aris)): import ipdb; ipdb.set_trace()
#     return aris
