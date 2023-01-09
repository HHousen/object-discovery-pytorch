import torch
import torch.nn.functional as F

border_width = 3

rbox = torch.zeros(3, 42, 42)
rbox[0, :border_width, :] = 1
rbox[0, -border_width:, :] = 1
rbox[0, :, :border_width] = 1
rbox[0, :, -border_width:] = 1
rbox = rbox.view(1, 3, 42, 42)

gbox = torch.zeros(3, 42, 42)
gbox[1, :border_width, :] = 1
gbox[1, -border_width:, :] = 1
gbox[1, :, :border_width] = 1
gbox[1, :, -border_width:] = 1
gbox = gbox.view(1, 3, 42, 42)


def visualize(
    x, z_pres, z_where_scale, z_where_shift,
):
    """
        x: (bs, 3, img_h, img_w)
        z_pres: (bs, 4, 4, 1)
        z_where_scale: (bs, 4, 4, 2)
        z_where_shift: (bs, 4, 4, 2)
    """
    global gbox, rbox
    bs, _, img_h, img_w = x.size()
    z_pres = z_pres.view(-1, 1, 1, 1)
    num_obj = z_pres.size(0) // bs
    z_scale = z_where_scale.view(-1, 2)
    z_shift = z_where_shift.view(-1, 2)
    gbox = gbox.to(z_pres.device)
    rbox = rbox.to(z_pres.device)
    bbox = spatial_transform(
        z_pres * gbox + (1 - z_pres) * rbox,
        torch.cat((z_scale, z_shift), dim=1),
        torch.Size([bs * num_obj, 3, img_h, img_w]),
        inverse=True,
    )

    return bbox


def linear_schedule_tensor(step, start_step, end_step, start_value, end_value, device):
    if start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = torch.tensor(start_value + slope * (step - start_step)).to(device)
    elif step >= end_step:
        x = torch.tensor(end_value).to(device)
    else:
        x = torch.tensor(start_value).to(device)

    return x


def linear_schedule(step, start_step, end_step, start_value, end_value):
    if start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = start_value + slope * (step - start_step)
    elif step >= end_step:
        x = end_value
    else:
        x = start_value

    return x


def spatial_transform(image, z_where, out_dims, inverse=False):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = torch.zeros(2, 3).repeat(image.shape[0], 1, 1).to(image.device)
    # set scaling
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-15)
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-15)

    # set translation
    theta[:, 0, -1] = (
        z_where[:, 2] if not inverse else -z_where[:, 2] / (z_where[:, 0] + 1e-15)
    )
    theta[:, 1, -1] = (
        z_where[:, 3] if not inverse else -z_where[:, 3] / (z_where[:, 1] + 1e-15)
    )
    # 2. construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims), align_corners=False)
    # 3. sample image from grid
    return F.grid_sample(image, grid, align_corners=False)


def kl_divergence_bern_bern(q_pres_probs, p_pres_prob, eps=1e-15):
    """
    Compute kl divergence
    :param z_pres_logits: (B, ...)
    :param prior_pres_prob: float
    :return: kl divergence, (B, ...)
    """
    # z_pres_probs = torch.sigmoid(z_pres_logits)
    kl = q_pres_probs * (
        torch.log(q_pres_probs + eps) - torch.log(p_pres_prob + eps)
    ) + (1 - q_pres_probs) * (
        torch.log(1 - q_pres_probs + eps) - torch.log(1 - p_pres_prob + eps)
    )

    return kl
