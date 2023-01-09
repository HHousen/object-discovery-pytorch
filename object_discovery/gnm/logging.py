# Partially based on https://github.com/karazijal/clevrtex/blob/fe982ab224689526f5e2f83a2f542ba958d88abd/experiments/framework/vis_mixin.py

import torch
import torch.nn.functional as F
from .metrics import align_masks_iou, dices
import itertools

import torch
import torchvision as tv
from ..segmentation_metrics import adjusted_rand_index

from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap

import numpy as np

from PIL import Image, ImageFont, ImageDraw

FNT = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", 7)
CMAPSPEC = ListedColormap(
    ["black", "red", "green", "blue", "yellow"]
    + list(
        itertools.chain.from_iterable(
            itertools.chain.from_iterable(
                zip(
                    [get_cmap("tab20b")(i) for i in range(i, 20, 4)],
                    [get_cmap("tab20c")(i) for i in range(i, 20, 4)],
                )
            )
            for i in range(4)
        )
    )
    + ["cyan", "magenta"]
    + [get_cmap("Set3")(i) for i in range(12)]
    + ["white"],
    name="SemSegMap",
)


@torch.no_grad()
def _to_img(img, lim=None, dim=-3):
    if lim:
        img = img[:lim]
    img = (img.clamp(0, 1) * 255).to(torch.uint8).cpu().detach()
    if img.shape[dim] < 3:
        epd_dims = [-1 for _ in range(len(img.shape))]
        epd_dims[dim] = 3
        img = img.expand(*epd_dims)
    return img


@torch.no_grad()
def log_recons(input, output, patches, masks, background=None, pres=None):
    vis_imgs = []
    img = _to_img(input)
    vis_imgs.extend(img)
    omg = _to_img(output, lim=len(img))
    vis_imgs.extend(omg)

    if background is not None and not torch.all(background == 0.0):
        bg = _to_img(background, lim=len(img))
        vis_imgs.extend(bg)

    masks = masks[: len(img)]
    patches = patches[: len(img)]
    ms = masks * patches
    for sid in range(patches.size(1)):
        # p = (patches[:len(img), sid].clamp(0., 1.) * 255.).to(torch.uint8).detach().cpu()
        # if p.shape[1] == 3:
        #     vis_imgs.extend(p)
        m = _to_img(ms[:, sid])
        m_hat = []
        if pres is not None:
            for i in range(0, len(img)):
                if pres[i, sid][0] == 1:
                    m[i, 0, :2, :2] = 0
                    m[i, 1, :2, :2] = 255
                    m[i, 2, :2, :2] = 0
                    m_hat.append(m[i])
                else:
                    m_hat.append(m[i])
        else:
            m_hat.extend(m)
        vis_imgs.extend(m_hat)
    grid = (
        tv.utils.make_grid(vis_imgs, pad_value=128, nrow=len(img), padding=1)
        .detach()
        .cpu()
    )
    return grid.permute([1, 2, 0]).numpy()


@torch.no_grad()
def log_images(input, output, bboxes=None, pres=None):
    vis_imgs = []
    img = _to_img(input)
    omg = _to_img(output, lim=len(img))

    if bboxes is not None:
        bboxes = bboxes.detach().cpu()
        pres = pres.detach().cpu()
        for i, (i_img, o_img) in enumerate(zip(img, omg)):
            vis_imgs.append(i_img)
            img_bbox = []
            for si in range(len(bboxes[i])):
                if pres[i, si] == 1:
                    img_bbox.append(bboxes[i, si])
            if img_bbox:
                img_to_draw = Image.fromarray(o_img.permute(1, 2, 0).numpy())
                draw = ImageDraw.Draw(img_to_draw)

                for bi, bbox in enumerate(img_bbox):
                    draw.rectangle(
                        bbox.to(torch.int64).tolist(), width=1, outline="green"
                    )
                o_img = torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)

            vis_imgs.append(o_img)
    else:
        for i, (i_img, o_img) in enumerate(zip(img, omg)):
            vis_imgs.append(i_img)
            vis_imgs.append(o_img)
    grid = tv.utils.make_grid(vis_imgs, pad_value=128, nrow=16).detach().cpu()
    return grid.permute([1, 2, 0]).numpy()


@torch.no_grad()
def log_semantic_images(input, output, true_masks, pred_masks):
    assert len(true_masks.shape) == 5 and len(pred_masks.shape) == 5
    img = _to_img(input)
    omg = _to_img(output, lim=len(img))
    true_masks = true_masks[: len(img)].to(torch.float).argmax(1).squeeze(1)
    pred_masks = pred_masks[: len(img)].to(torch.float).argmax(1).squeeze(1)
    tms = (_cmap_tensor(true_masks) * 255.0).to(torch.uint8)
    pms = (_cmap_tensor(pred_masks) * 255.0).to(torch.uint8)
    vis_imgs = list(itertools.chain.from_iterable(zip(img, omg, tms, pms)))
    grid = tv.utils.make_grid(vis_imgs, pad_value=128, nrow=16).detach().cpu()
    return grid.permute([1, 2, 0]).numpy()


@torch.no_grad()
def _cmap_tensor(t):
    t_hw = t.cpu().detach().numpy()
    o_hwc = CMAPSPEC(t_hw)[..., :3]  # drop alpha
    o = torch.from_numpy(o_hwc).transpose(-1, -2).transpose(-2, -3)
    return o


def gnm_log_validation_outputs(batch, batch_idx, output, is_global_zero):
    logs = {}
    img, masks, vis = batch
    masks = masks.transpose(-1, -2).transpose(-2, -3).to(torch.float)

    mse = F.mse_loss(output["canvas"], img, reduction="none").sum((1, 2, 3))
    logs["mse"] = mse

    ali_pmasks = None
    ali_tmasks = None

    # Transforms might have changed this.
    # cnts = torch.sum(vis, dim=-1) - 1  # -1 for discounting the background from visibility
    # estimate from masks
    cnts = (
        torch.round(masks.to(torch.float)).flatten(2).any(-1).to(torch.float).sum(-1)
        - 1
    )

    if "steps" in output:
        pred_masks = output["steps"]["mask"]
        pred_vis = output["steps"]["z_pres"].squeeze(-1)

        # `align_masks_iou` adds the background pixels to `ali_pmasks` (aligned
        # predicted masks).
        ali_pmasks, ali_tmasks, ious, ali_pvis, ali_tvis = align_masks_iou(
            pred_masks, masks, pred_mask_vis=pred_vis, true_mask_vis=vis, has_bg=False
        )

        ali_cvis = ali_pvis | ali_tvis
        num_paired_slots = ali_cvis.sum(-1) - 1
        mses = F.mse_loss(
            output["canvas"][:, None] * ali_pmasks,
            img[:, None] * ali_tmasks,
            reduction="none",
        ).sum((-1, -2, -3))

        bg_mse = mses[:, 0]
        logs["bg_mse"] = bg_mse

        slot_mse = mses[:, 1:].sum(-1) / num_paired_slots
        logs["slot_mse"] = slot_mse

        mious = ious[:, 1:].sum(-1) / num_paired_slots
        # mious = torch.where(zero_mask, 0., mious)
        logs["miou"] = mious

        dice = dices(ali_pmasks, ali_tmasks)
        mdice = dice[:, 1:].sum(-1) / num_paired_slots
        logs["dice"] = mdice

        # aris = ari(ali_pmasks, ali_tmasks)
        batch_size, num_entries, channels, height, width = ali_tmasks.shape
        ali_tmasks_reshaped = (
            torch.reshape(
                ali_tmasks.squeeze(), [batch_size, num_entries, height * width]
            )
            .permute([0, 2, 1])
            .to(torch.float)
        )
        batch_size, num_entries, channels, height, width = ali_pmasks.shape
        ali_pmasks_reshaped = (
            torch.reshape(
                ali_pmasks.squeeze(), [batch_size, num_entries, height * width]
            )
            .permute([0, 2, 1])
            .to(torch.float)
        )
        logs["ari_with_background"] = adjusted_rand_index(
            ali_tmasks_reshaped, ali_pmasks_reshaped
        )

        # aris_fg = ari(ali_pmasks, ali_tmasks, True).mean().detach()
        # `[..., 1:]` removes the background pixels group from the true mask.
        logs["ari"] = adjusted_rand_index(
            ali_tmasks_reshaped[..., 1:], ali_pmasks_reshaped
        )

        # Can also calculate ari (same as above line) without using the aligned
        # masks by directly adding the background to the predicted masks. The
        # background is everything that is not predicted as an object. This is
        # done automatically when aligning the masks in `align_masks_iou`.
        # pred_masks_with_background = torch.cat([1 - pred_masks.sum(1, keepdim=True), pred_masks], 1)
        # batch_size, num_entries, channels, height, width = masks.shape
        # masks_reshaped = torch.reshape(masks, [batch_size, num_entries, height * width]).permute([0, 2, 1]).to(torch.float)
        # batch_size, num_entries, channels, height, width = pred_masks_with_background.shape
        # pred_masks_with_background_reshaped = torch.reshape(pred_masks_with_background, [batch_size, num_entries, height * width]).permute([0, 2, 1]).to(torch.float)
        # logs["ari4"] = adjusted_rand_index(masks_reshaped[..., 1:], pred_masks_with_background_reshaped)

        batch_size, num_entries, channels, height, width = masks.shape
        masks_reshaped = (
            torch.reshape(masks, [batch_size, num_entries, height * width])
            .permute([0, 2, 1])
            .to(torch.float)
        )
        batch_size, num_entries, channels, height, width = pred_masks.shape
        pred_masks_reshaped = (
            torch.reshape(pred_masks, [batch_size, num_entries, height * width])
            .permute([0, 2, 1])
            .to(torch.float)
        )
        logs["ari_no_background"] = adjusted_rand_index(
            masks_reshaped[..., 1:], pred_masks_reshaped
        )

    pred_counts = output["counts"].detach().to(torch.int)
    logs["acc"] = (pred_counts == cnts).to(float)
    logs["cnt"] = pred_counts.to(float)

    images = {}
    if batch_idx == 0 and is_global_zero:
        images["output"] = log_images(img, output["canvas_with_bbox"])
        if "steps" in output:
            images["recon"] = log_recons(
                img[:32],
                output["canvas"],
                output["steps"]["patch"],
                output["steps"]["mask"],
                output.get("background", None),
                pres=output["steps"]["z_pres"],
            )

        # If masks have been aligned; log semantic map
        if ali_pmasks is not None and ali_tmasks is not None:
            images["segmentation"] = log_semantic_images(
                img[:32], output["canvas"], ali_tmasks, ali_pmasks
            )

    return logs, images
