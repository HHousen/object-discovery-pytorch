import torch
from torch import nn
from .module import (
    LocalLatentDecoder,
    LocalSampler,
    StructDRAW,
    BgDecoder,
    BgGenerator,
    BgEncoder,
)
from .submodule import StackConvNorm
import torch.distributions as dist
from .utils import (
    linear_schedule_tensor,
    spatial_transform,
    kl_divergence_bern_bern,
    linear_schedule,
    visualize,
)
from typing import List, Tuple


class GNM(nn.Module):
    shortname = "gnm"

    def __init__(self, args):
        super(GNM, self).__init__()
        self.args = args

        self.img_encoder = StackConvNorm(
            self.args.data.inp_channel,
            self.args.arch.conv.img_encoder_filters,
            self.args.arch.conv.img_encoder_kernel_sizes,
            self.args.arch.conv.img_encoder_strides,
            self.args.arch.conv.img_encoder_groups,
            norm_act_final=True,
        )
        self.global_struct_draw = StructDRAW(self.args)
        self.p_z_given_x_or_g_net = LocalLatentDecoder(self.args)
        # Share latent decoder for p and q
        self.local_latent_sampler = LocalSampler(self.args)

        if self.args.arch.phase_background:
            self.p_bg_decoder = BgDecoder(self.args)
            self.p_bg_given_g_net = BgGenerator(self.args)
            self.q_bg_given_x_net = BgEncoder(self.args)

        self.register_buffer("aux_p_what_mean", torch.zeros(1))
        self.register_buffer("aux_p_what_std", torch.ones(1))
        self.register_buffer("aux_p_bg_mean", torch.zeros(1))
        self.register_buffer("aux_p_bg_std", torch.ones(1))
        self.register_buffer("aux_p_depth_mean", torch.zeros(1))
        self.register_buffer("aux_p_depth_std", torch.ones(1))
        self.register_buffer(
            "aux_p_where_mean",
            torch.tensor(
                [self.args.const.scale_mean, self.args.const.ratio_mean, 0, 0]
            )[None, :],
        )
        # self.register_buffer('auxiliary_where_std', torch.ones(1))
        self.register_buffer(
            "aux_p_where_std",
            torch.tensor(
                [
                    self.args.const.scale_std,
                    self.args.const.ratio_std,
                    self.args.const.shift_std,
                    self.args.const.shift_std,
                ]
            )[None, :],
        )
        self.register_buffer(
            "aux_p_pres_probs", torch.tensor(self.args.train.p_pres_anneal_start_value)
        )
        self.register_buffer(
            "background",
            torch.zeros(
                1,
                self.args.data.inp_channel,
                self.args.data.img_h,
                self.args.data.img_w,
            ),
        )

    @property
    def aux_p_what(self):
        return dist.Normal(self.aux_p_what_mean, self.aux_p_what_std)

    @property
    def aux_p_bg(self):
        return dist.Normal(self.aux_p_bg_mean, self.aux_p_bg_std)

    @property
    def aux_p_depth(self):
        return dist.Normal(self.aux_p_depth_mean, self.aux_p_depth_std)

    @property
    def aux_p_where(self):
        return dist.Normal(self.aux_p_where_mean, self.aux_p_where_std)

    def forward(self, x: torch.Tensor, global_step) -> Tuple:
        self.args = hyperparam_anneal(self.args, global_step)
        bs = x.size(0)

        img_enc = self.img_encoder(x)
        if self.args.arch.phase_background:
            lv_q_bg, ss_q_bg = self.q_bg_given_x_net(img_enc)
            q_bg_mean, q_bg_std = ss_q_bg
        else:
            lv_q_bg = [self.background.new_zeros(1, 1)]
            q_bg_mean = self.background.new_zeros(1, 1)
            q_bg_std = self.background.new_ones(1, 1)
            ss_q_bg = [q_bg_mean, q_bg_std]

        q_bg = dist.Normal(q_bg_mean, q_bg_std)

        pa_g, lv_g, ss_g = self.global_struct_draw(img_enc)

        global_dec = pa_g[0]

        p_global_mean_all, p_global_std_all, q_global_mean_all, q_global_std_all = ss_g

        p_global_all = dist.Normal(p_global_mean_all, p_global_std_all)

        q_global_all = dist.Normal(q_global_mean_all, q_global_std_all)

        ss_p_z = self.p_z_given_x_or_g_net(global_dec)

        # (bs, dim, num_cell, num_cell)
        (
            p_pres_logits,
            p_where_mean,
            p_where_std,
            p_depth_mean,
            p_depth_std,
            p_what_mean,
            p_what_std,
        ) = ss_p_z

        p_pres_given_g_probs_reshaped = torch.sigmoid(
            p_pres_logits.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            )
        )

        p_where_given_g = dist.Normal(
            p_where_mean.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
            p_where_std.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
        )
        p_depth_given_g = dist.Normal(
            p_depth_mean.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
            p_depth_std.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
        )
        p_what_given_g = dist.Normal(
            p_what_mean.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
            p_what_std.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
        )

        ss_q_z = self.p_z_given_x_or_g_net(img_enc, ss_p_z=ss_p_z)

        # (bs, dim, num_cell, num_cell)
        (
            q_pres_logits,
            q_where_mean,
            q_where_std,
            q_depth_mean,
            q_depth_std,
            q_what_mean,
            q_what_std,
        ) = ss_q_z

        q_pres_given_x_and_g_probs_reshaped = torch.sigmoid(
            q_pres_logits.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            )
        )

        q_where_given_x_and_g = dist.Normal(
            q_where_mean.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
            q_where_std.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
        )
        q_depth_given_x_and_g = dist.Normal(
            q_depth_mean.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
            q_depth_std.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
        )
        q_what_given_x_and_g = dist.Normal(
            q_what_mean.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
            q_what_std.permute(0, 2, 3, 1).reshape(
                bs * self.args.arch.num_cell ** 2, -1
            ),
        )

        if self.args.arch.phase_background:
            # lv_p_bg, ss_p_bg = self.ss_p_bg_given_g(lv_g)
            lv_p_bg, ss_p_bg = self.p_bg_given_g_net(lv_g[0], phase_use_mode=False)
            p_bg_mean, p_bg_std = ss_p_bg
        else:
            lv_p_bg = [self.background.new_zeros(1, 1)]
            p_bg_mean = self.background.new_zeros(1, 1)
            p_bg_std = self.background.new_ones(1, 1)

        p_bg = dist.Normal(p_bg_mean, p_bg_std)

        pa_recon, lv_z = self.lv_p_x_given_z_and_bg(ss_q_z, lv_q_bg, global_step)
        *pa_recon, patches, masks = pa_recon
        canvas = pa_recon[0]
        background = pa_recon[-1]

        z_pres, z_where, z_depth, z_what, z_where_origin = lv_z

        p_dists = [
            p_global_all,
            p_pres_given_g_probs_reshaped,
            p_where_given_g,
            p_depth_given_g,
            p_what_given_g,
            p_bg,
        ]

        q_dists = [
            q_global_all,
            q_pres_given_x_and_g_probs_reshaped,
            q_where_given_x_and_g,
            q_depth_given_x_and_g,
            q_what_given_x_and_g,
            q_bg,
        ]

        log_like, kl, log_imp = self.elbo(
            x, p_dists, q_dists, lv_z, lv_g, lv_q_bg, pa_recon, global_step
        )

        self.log = {}

        if self.args.log.phase_log:
            pa_recon_from_q_g, _ = self.get_recon_from_q_g(
                global_step, global_dec=global_dec, lv_g=lv_g
            )

            z_pres_permute = z_pres.permute(0, 2, 3, 1)
            self.log = {
                "z_what": z_what.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_what_dim
                ),
                "z_where_scale": z_where.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[:, : self.args.z.z_where_scale_dim],
                "z_where_shift": z_where.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[:, self.args.z.z_where_scale_dim :],
                "z_where_origin": z_where_origin.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                ),
                "z_pres": z_pres_permute,
                "p_pres_probs": p_pres_given_g_probs_reshaped,
                "p_pres_logits": p_pres_logits,
                "p_what_std": p_what_std.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_what_dim
                )[z_pres_permute.view(-1) > 0.05],
                "p_what_mean": p_what_mean.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_what_dim
                )[z_pres_permute.view(-1) > 0.05],
                "p_where_scale_std": p_where_std.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[z_pres_permute.view(-1) > 0.05][:, : self.args.z.z_where_scale_dim],
                "p_where_scale_mean": p_where_mean.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[z_pres_permute.view(-1) > 0.05][:, : self.args.z.z_where_scale_dim],
                "p_where_shift_std": p_where_std.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[z_pres_permute.view(-1) > 0.05][:, self.args.z.z_where_scale_dim :],
                "p_where_shift_mean": p_where_mean.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[z_pres_permute.view(-1) > 0.05][:, self.args.z.z_where_scale_dim :],
                "q_pres_probs": q_pres_given_x_and_g_probs_reshaped,
                "q_pres_logits": q_pres_logits,
                "q_what_std": q_what_std.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_what_dim
                )[z_pres_permute.view(-1) > 0.05],
                "q_what_mean": q_what_mean.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_what_dim
                )[z_pres_permute.view(-1) > 0.05],
                "q_where_scale_std": q_where_std.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[z_pres_permute.view(-1) > 0.05][:, : self.args.z.z_where_scale_dim],
                "q_where_scale_mean": q_where_mean.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[z_pres_permute.view(-1) > 0.05][:, : self.args.z.z_where_scale_dim],
                "q_where_shift_std": q_where_std.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[z_pres_permute.view(-1) > 0.05][:, self.args.z.z_where_scale_dim :],
                "q_where_shift_mean": q_where_mean.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_where_dim
                )[z_pres_permute.view(-1) > 0.05][:, self.args.z.z_where_scale_dim :],
                "z_depth": z_depth.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_depth_dim
                ),
                "p_depth_std": p_depth_std.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_depth_dim
                )[z_pres_permute.view(-1) > 0.05],
                "p_depth_mean": p_depth_mean.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_depth_dim
                )[z_pres_permute.view(-1) > 0.05],
                "q_depth_std": q_depth_std.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_depth_dim
                )[z_pres_permute.view(-1) > 0.05],
                "q_depth_mean": q_depth_mean.permute(0, 2, 3, 1).reshape(
                    -1, self.args.z.z_depth_dim
                )[z_pres_permute.view(-1) > 0.05],
                "recon": pa_recon[0],
                "recon_from_q_g": pa_recon_from_q_g[0],
                "log_prob_x_given_g": dist.Normal(
                    pa_recon_from_q_g[0], self.args.const.likelihood_sigma
                )
                .log_prob(x)
                .flatten(start_dim=1)
                .sum(1),
                "global_dec": global_dec,
            }
            z_global_all = lv_g[0]
            for i in range(self.args.arch.draw_step):
                self.log[f"z_global_step_{i}"] = z_global_all[:, i]
                self.log[f"q_global_mean_step_{i}"] = q_global_mean_all[:, i]
                self.log[f"q_global_std_step_{i}"] = q_global_std_all[:, i]
                self.log[f"p_global_mean_step_{i}"] = p_global_mean_all[:, i]
                self.log[f"p_global_std_step_{i}"] = p_global_std_all[:, i]
            if self.args.arch.phase_background:
                self.log["z_bg"] = lv_q_bg[0]
                self.log["p_bg_mean"] = p_bg_mean
                self.log["p_bg_std"] = p_bg_std
                self.log["q_bg_mean"] = q_bg_mean
                self.log["q_bg_std"] = q_bg_std
                self.log["recon_from_q_g_bg"] = pa_recon_from_q_g[-1]
                self.log["recon_from_q_g_fg"] = pa_recon_from_q_g[1]
                self.log["recon_from_q_g_alpha"] = pa_recon_from_q_g[2]
                self.log["recon_bg"] = pa_recon[-1]
                self.log["recon_fg"] = pa_recon[1]
                self.log["recon_alpha"] = pa_recon[2]

        ss = [ss_q_z, ss_q_bg, ss_g[2:]]
        (
            aux_kl_pres,
            aux_kl_where,
            aux_kl_depth,
            aux_kl_what,
            aux_kl_bg,
            kl_pres,
            kl_where,
            kl_depth,
            kl_what,
            kl_global_all,
            kl_bg,
        ) = kl

        aux_kl_pres_raw = aux_kl_pres.mean(dim=0)
        aux_kl_where_raw = aux_kl_where.mean(dim=0)
        aux_kl_depth_raw = aux_kl_depth.mean(dim=0)
        aux_kl_what_raw = aux_kl_what.mean(dim=0)
        aux_kl_bg_raw = aux_kl_bg.mean(dim=0)
        kl_pres_raw = kl_pres.mean(dim=0)
        kl_where_raw = kl_where.mean(dim=0)
        kl_depth_raw = kl_depth.mean(dim=0)
        kl_what_raw = kl_what.mean(dim=0)
        kl_bg_raw = kl_bg.mean(dim=0)

        log_like = log_like.mean(dim=0)

        aux_kl_pres = aux_kl_pres_raw * self.args.train.beta_aux_pres
        aux_kl_where = aux_kl_where_raw * self.args.train.beta_aux_where
        aux_kl_depth = aux_kl_depth_raw * self.args.train.beta_aux_depth
        aux_kl_what = aux_kl_what_raw * self.args.train.beta_aux_what
        aux_kl_bg = aux_kl_bg_raw * self.args.train.beta_aux_bg
        kl_pres = kl_pres_raw * self.args.train.beta_pres
        kl_where = kl_where_raw * self.args.train.beta_where
        kl_depth = kl_depth_raw * self.args.train.beta_depth
        kl_what = kl_what_raw * self.args.train.beta_what
        kl_bg = kl_bg_raw * self.args.train.beta_bg

        kl_global_raw = kl_global_all.sum(dim=-1).mean(dim=0)
        kl_global = kl_global_raw * self.args.train.beta_global

        recon_loss = log_like
        kl = (
            kl_pres
            + kl_where
            + kl_depth
            + kl_what
            + kl_bg
            + kl_global
            + aux_kl_pres
            + aux_kl_where
            + aux_kl_depth
            + aux_kl_what
            + aux_kl_bg
        )
        elbo = recon_loss - kl
        loss = -elbo

        bbox = visualize(
            x.cpu(),
            self.log["z_pres"]
            .view(bs, self.args.arch.num_cell ** 2, -1)
            .cpu()
            .detach(),
            self.log["z_where_scale"]
            .view(bs, self.args.arch.num_cell ** 2, -1)
            .cpu()
            .detach(),
            self.log["z_where_shift"]
            .view(bs, self.args.arch.num_cell ** 2, -1)
            .cpu()
            .detach(),
            only_bbox=True,
            phase_only_display_pres=False,
        )

        bbox = (
            bbox.view(x.shape[0], -1, 3, self.args.data.img_h, self.args.data.img_w)
            .sum(1)
            .clamp(0.0, 1.0)
        )
        # bbox_img = x.cpu().expand(-1, 3, -1, -1).contiguous()
        # bbox_img[bbox.sum(dim=1, keepdim=True).expand(-1, 3, -1, -1) > 0.5] = \
        #     bbox[bbox.sum(dim=1, keepdim=True).expand(-1, 3, -1, -1) > 0.5]
        ret = {
            "canvas": canvas,
            "canvas_with_bbox": bbox,
            "background": background,
            "steps": {
                "patch": patches,
                "mask": masks,
                "z_pres": z_pres.view(bs, self.args.arch.num_cell ** 2, -1),
            },
            "counts": torch.round(z_pres).flatten(1).sum(-1),
            "loss": loss,
            "elbo": elbo,
            "kl": kl,
            "rec_loss": recon_loss,
            "kl_pres": kl_pres,
            "kl_aux_pres": aux_kl_pres,
            "kl_where": kl_where,
            "kl_aux_where": aux_kl_where,
            "kl_what": kl_what,
            "kl_aux_what": aux_kl_what,
            "kl_depth": kl_depth,
            "kl_aux_depth": aux_kl_depth,
            "kl_bg": kl_bg,
            "kl_aux_bg": aux_kl_bg,
            "kl_global": kl_global,
        }

        # return pa_recon, log_like, kl, log_imp, lv_z + lv_g + lv_q_bg, ss, self.log
        return ret

    def get_recon_from_q_g(
        self,
        global_step,
        img: torch.Tensor = None,
        global_dec: torch.Tensor = None,
        lv_g: List = None,
        phase_use_mode: bool = False,
    ) -> Tuple:

        assert img is not None or (
            global_dec is not None and lv_g is not None
        ), "Provide either image or p_l_given_g"
        if img is not None:
            img_enc = self.img_encoder(img)
            pa_g, lv_g, ss_g = self.global_struct_draw(img_enc)

            global_dec = pa_g[0]

        if self.args.arch.phase_background:
            lv_p_bg, _ = self.p_bg_given_g_net(lv_g[0], phase_use_mode=phase_use_mode)
        else:
            lv_p_bg = [self.background.new_zeros(1, 1)]

        ss_z = self.p_z_given_x_or_g_net(global_dec)

        pa, lv = self.lv_p_x_given_z_and_bg(
            ss_z, lv_p_bg, global_step, phase_use_mode=phase_use_mode
        )

        lv = lv + lv_p_bg

        return pa, lv

    def elbo(
        self,
        x: torch.Tensor,
        p_dists: List,
        q_dists: List,
        lv_z: List,
        lv_g: List,
        lv_bg: List,
        pa_recon: List,
        global_step,
    ) -> Tuple:

        bs = x.size(0)

        (
            p_global_all,
            p_pres_given_g_probs_reshaped,
            p_where_given_g,
            p_depth_given_g,
            p_what_given_g,
            p_bg,
        ) = p_dists

        (
            q_global_all,
            q_pres_given_x_and_g_probs_reshaped,
            q_where_given_x_and_g,
            q_depth_given_x_and_g,
            q_what_given_x_and_g,
            q_bg,
        ) = q_dists

        y, y_nobg, alpha_map, bg = pa_recon

        if self.args.log.phase_nll:
            # (bs, dim, num_cell, num_cell)
            z_pres, _, z_depth, z_what, z_where_origin = lv_z
            # (bs * num_cell * num_cell, dim)
            z_pres_reshape = z_pres.permute(0, 2, 3, 1).reshape(
                -1, self.args.z.z_pres_dim
            )
            z_depth_reshape = z_depth.permute(0, 2, 3, 1).reshape(
                -1, self.args.z.z_depth_dim
            )
            z_what_reshape = z_what.permute(0, 2, 3, 1).reshape(
                -1, self.args.z.z_what_dim
            )
            z_where_origin_reshape = z_where_origin.permute(0, 2, 3, 1).reshape(
                -1, self.args.z.z_where_dim
            )
            # (bs, dim, 1, 1)
            z_bg = lv_bg[0]
            # (bs, step, dim, 1, 1)
            z_g = lv_g[0]
        else:
            z_pres, _, _, _, z_where_origin = lv_z

            z_pres_reshape = z_pres.permute(0, 2, 3, 1).reshape(
                -1, self.args.z.z_pres_dim
            )

        if self.args.train.p_pres_anneal_end_step != 0:
            self.aux_p_pres_probs = linear_schedule_tensor(
                global_step,
                self.args.train.p_pres_anneal_start_step,
                self.args.train.p_pres_anneal_end_step,
                self.args.train.p_pres_anneal_start_value,
                self.args.train.p_pres_anneal_end_value,
                self.aux_p_pres_probs.device,
            )

        if self.args.train.aux_p_scale_anneal_end_step != 0:
            aux_p_scale_mean = linear_schedule_tensor(
                global_step,
                self.args.train.aux_p_scale_anneal_start_step,
                self.args.train.aux_p_scale_anneal_end_step,
                self.args.train.aux_p_scale_anneal_start_value,
                self.args.train.aux_p_scale_anneal_end_value,
                self.aux_p_where_mean.device,
            )
            self.aux_p_where_mean[:, 0] = aux_p_scale_mean

        auxiliary_prior_z_pres_probs = self.aux_p_pres_probs[None][None, :].expand(
            bs * self.args.arch.num_cell ** 2, -1
        )

        aux_kl_pres = kl_divergence_bern_bern(
            q_pres_given_x_and_g_probs_reshaped, auxiliary_prior_z_pres_probs
        )
        aux_kl_where = dist.kl_divergence(
            q_where_given_x_and_g, self.aux_p_where
        ) * z_pres_reshape.clamp(min=1e-5)
        aux_kl_depth = dist.kl_divergence(
            q_depth_given_x_and_g, self.aux_p_depth
        ) * z_pres_reshape.clamp(min=1e-5)
        aux_kl_what = dist.kl_divergence(
            q_what_given_x_and_g, self.aux_p_what
        ) * z_pres_reshape.clamp(min=1e-5)

        kl_pres = kl_divergence_bern_bern(
            q_pres_given_x_and_g_probs_reshaped, p_pres_given_g_probs_reshaped
        )

        kl_where = dist.kl_divergence(q_where_given_x_and_g, p_where_given_g)
        kl_depth = dist.kl_divergence(q_depth_given_x_and_g, p_depth_given_g)
        kl_what = dist.kl_divergence(q_what_given_x_and_g, p_what_given_g)

        kl_global_all = dist.kl_divergence(q_global_all, p_global_all)

        if self.args.arch.phase_background:
            kl_bg = dist.kl_divergence(q_bg, p_bg)
            aux_kl_bg = dist.kl_divergence(q_bg, self.aux_p_bg)
        else:
            kl_bg = self.background.new_zeros(bs, 1)
            aux_kl_bg = self.background.new_zeros(bs, 1)

        log_like = dist.Normal(y, self.args.const.likelihood_sigma).log_prob(x)

        log_imp_list = []
        if self.args.log.phase_nll:
            log_pres_prior = z_pres_reshape * torch.log(
                p_pres_given_g_probs_reshaped + self.args.const.eps
            ) + (1 - z_pres_reshape) * torch.log(
                1 - p_pres_given_g_probs_reshaped + self.args.const.eps
            )
            log_pres_pos = z_pres_reshape * torch.log(
                q_pres_given_x_and_g_probs_reshaped + self.args.const.eps
            ) + (1 - z_pres_reshape) * torch.log(
                1 - q_pres_given_x_and_g_probs_reshaped + self.args.const.eps
            )

            log_imp_pres = log_pres_prior - log_pres_pos

            log_imp_depth = p_depth_given_g.log_prob(
                z_depth_reshape
            ) - q_depth_given_x_and_g.log_prob(z_depth_reshape)

            log_imp_what = p_what_given_g.log_prob(
                z_what_reshape
            ) - q_what_given_x_and_g.log_prob(z_what_reshape)

            log_imp_where = p_where_given_g.log_prob(
                z_where_origin_reshape
            ) - q_where_given_x_and_g.log_prob(z_where_origin_reshape)

            if self.args.arch.phase_background:
                log_imp_bg = p_bg.log_prob(z_bg) - q_bg.log_prob(z_bg)
            else:
                log_imp_bg = x.new_zeros(bs, 1)

            log_imp_g = p_global_all.log_prob(z_g) - q_global_all.log_prob(z_g)

            log_imp_list = [
                log_imp_pres.view(
                    bs, self.args.arch.num_cell, self.args.arch.num_cell, -1
                )
                .flatten(start_dim=1)
                .sum(1),
                log_imp_depth.view(
                    bs, self.args.arch.num_cell, self.args.arch.num_cell, -1
                )
                .flatten(start_dim=1)
                .sum(1),
                log_imp_what.view(
                    bs, self.args.arch.num_cell, self.args.arch.num_cell, -1
                )
                .flatten(start_dim=1)
                .sum(1),
                log_imp_where.view(
                    bs, self.args.arch.num_cell, self.args.arch.num_cell, -1
                )
                .flatten(start_dim=1)
                .sum(1),
                log_imp_bg.flatten(start_dim=1).sum(1),
                log_imp_g.flatten(start_dim=1).sum(1),
            ]

        return (
            log_like.flatten(start_dim=1).sum(1),
            [
                aux_kl_pres.view(
                    bs, self.args.arch.num_cell, self.args.arch.num_cell, -1
                )
                .flatten(start_dim=1)
                .sum(-1),
                aux_kl_where.view(
                    bs, self.args.arch.num_cell, self.args.arch.num_cell, -1
                )
                .flatten(start_dim=1)
                .sum(-1),
                aux_kl_depth.view(
                    bs, self.args.arch.num_cell, self.args.arch.num_cell, -1
                )
                .flatten(start_dim=1)
                .sum(-1),
                aux_kl_what.view(
                    bs, self.args.arch.num_cell, self.args.arch.num_cell, -1
                )
                .flatten(start_dim=1)
                .sum(-1),
                aux_kl_bg.flatten(start_dim=1).sum(-1),
                kl_pres.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1)
                .flatten(start_dim=1)
                .sum(-1),
                kl_where.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1)
                .flatten(start_dim=1)
                .sum(-1),
                kl_depth.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1)
                .flatten(start_dim=1)
                .sum(-1),
                kl_what.view(bs, self.args.arch.num_cell, self.args.arch.num_cell, -1)
                .flatten(start_dim=1)
                .sum(-1),
                kl_global_all.flatten(start_dim=2).sum(-1),
                kl_bg.flatten(start_dim=1).sum(-1),
            ],
            log_imp_list,
        )

    # def get_img_enc(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     :param x: (bs, inp_channel, img_h, img_w)
    #     :return: img_enc: (bs, dim, num_cell, num_cell)
    #     """
    #
    #     img_enc = self.img_encoder(x)
    #
    #     return img_enc

    # def ss_p_z_given_g(self, global_dec: torch.Tensor) -> List:
    #     """
    #     :param x: sample of z_global variable (bs, dim, 1, 1)
    #     :return:
    #     """
    #     ss_z = self.p_z_given_g_net(global_dec)
    #
    #     return ss_z

    # def ss_q_z_given_x(self, img_enc: torch.Tensor, global_dec: torch.Tensor, ss_p_z: List) -> List:
    #     """
    #     :param x: sample of z_global variable (bs, dim, 1, 1)
    #     :return:
    #     """
    #     ss_z = self.p_z_given_x_or_g_net(img_enc, ss_p_z=ss_p_z)
    #
    #     return ss_z

    # def ss_q_bg_given_x(self, x: torch.Tensor) -> Tuple:
    #     """
    #     :param x: (bs, dim, img_h, img_w)
    #     :return:
    #     """
    #     lv_q_bg, ss_q_bg = self.q_bg_given_x_net(x)
    #
    #     return lv_q_bg, ss_q_bg

    # def ss_p_bg_given_g(self, lv_g: List, phase_use_mode: bool = False) -> Tuple:
    #     """
    #     :param x: (bs, dim, img_h, img_w)
    #     :return:
    #     """
    #     z_global_all = lv_g[0]
    #     lv_p_bg, ss_p_bg = self.p_bg_given_g_net(z_global_all, phase_use_mode=phase_use_mode)
    #
    #     return lv_p_bg, ss_p_bg

    def lv_p_x_given_z_and_bg(
        self, ss: List, lv_bg: List, global_step, phase_use_mode: bool = False
    ) -> Tuple:
        """
        :param z: (bs, z_what_dim)
        :return:
        """
        # x: (bs, inp_channel, img_h, img_w)
        pa, lv_z = self.local_latent_sampler(ss, phase_use_mode=phase_use_mode)

        o_att, a_att, *_ = pa
        z_pres, z_where, z_depth, *_ = lv_z

        if self.args.arch.phase_background:
            z_bg = lv_bg[0]
            pa_bg = self.p_bg_decoder(z_bg)
            y_bg = pa_bg[0]
        else:
            # pa_bg = [self.background.expand(lv_z[0].size(0), -1, -1, -1)]
            y_bg = self.background.expand(lv_z[0].size(0), -1, -1, -1)

        # pa = pa + pa_bg

        y, y_fg, alpha_map, patches, masks = self.render(
            o_att, a_att, y_bg, z_pres, z_where, z_depth, global_step
        )

        return [y, y_fg, alpha_map, y_bg, patches, masks], lv_z

    # def pa_bg_given_z_bg(self, lv_bg: List) -> List:
    #     """
    #     :param lv_bg[0]: (bs, z_bg_dim, 1, 1)
    #     :return:
    #     """
    #     z_bg = lv_bg[0]
    #     pa = self.p_bg_decoder(z_bg)
    #
    #     return pa

    def render(self, o_att, a_att, bg, z_pres, z_where, z_depth, global_step) -> List:
        """
        :param pa: variables with size (bs, dim, num_cell, num_cell)
        :param lv_z: o and a with size (bs * num_cell * num_cell, dim)
        :return:
        """

        bs = z_pres.size(0)

        z_pres = z_pres.permute(0, 2, 3, 1).reshape(
            bs * self.args.arch.num_cell ** 2, -1
        )
        z_where = z_where.permute(0, 2, 3, 1).reshape(
            bs * self.args.arch.num_cell ** 2, -1
        )
        z_depth = z_depth.permute(0, 2, 3, 1).reshape(
            bs * self.args.arch.num_cell ** 2, -1
        )

        if self.args.arch.phase_overlap == True:
            if (
                self.args.train.phase_bg_alpha_curriculum
                and self.args.train.bg_alpha_curriculum_period[0]
                < global_step
                < self.args.train.bg_alpha_curriculum_period[1]
            ):
                z_pres = z_pres.clamp(max=0.99)
            a_att_hat = a_att * z_pres.view(-1, 1, 1, 1)
            y_att = a_att_hat * o_att

            # (bs, self.args.arch.num_cell * self.args.arch.num_cell, 3, img_h, img_w)
            y_att_full_res = spatial_transform(
                y_att,
                z_where,
                (
                    bs * self.args.arch.num_cell ** 2,
                    self.args.data.inp_channel,
                    self.args.data.img_h,
                    self.args.data.img_w,
                ),
                inverse=True,
            ).view(
                -1,
                self.args.arch.num_cell * self.args.arch.num_cell,
                self.args.data.inp_channel,
                self.args.data.img_h,
                self.args.data.img_w,
            )
            o_att_full_res = spatial_transform(
                o_att,
                z_where,
                (
                    bs * self.args.arch.num_cell ** 2,
                    self.args.data.inp_channel,
                    self.args.data.img_h,
                    self.args.data.img_w,
                ),
                inverse=True,
            ).view(
                -1,
                self.args.arch.num_cell * self.args.arch.num_cell,
                self.args.data.inp_channel,
                self.args.data.img_h,
                self.args.data.img_w,
            )

            # (self.args.arch.num_cell * self.args.arch.num_cell * bs, 1, glimpse_size, glimpse_size)
            importance_map = a_att_hat * torch.sigmoid(-z_depth).view(-1, 1, 1, 1)
            # (self.args.arch.num_cell * self.args.arch.num_cell * bs, 1, img_h, img_w)
            importance_map_full_res = spatial_transform(
                importance_map,
                z_where,
                (
                    self.args.arch.num_cell * self.args.arch.num_cell * bs,
                    1,
                    self.args.data.img_h,
                    self.args.data.img_w,
                ),
                inverse=True,
            )
            # # (bs, self.args.arch.num_cell * self.args.arch.num_cell, 1, img_h, img_w)
            importance_map_full_res = importance_map_full_res.view(
                -1,
                self.args.arch.num_cell * self.args.arch.num_cell,
                1,
                self.args.data.img_h,
                self.args.data.img_w,
            )
            importance_map_full_res_norm = importance_map_full_res / (
                importance_map_full_res.sum(dim=1, keepdim=True) + self.args.const.eps
            )

            # (bs, 3, img_h, img_w)
            y_nobg = (y_att_full_res * importance_map_full_res_norm).sum(dim=1)

            # (bs, self.args.arch.num_cell * self.args.arch.num_cell, 1, img_h, img_w)
            a_att_hat_full_res = spatial_transform(
                a_att_hat,
                z_where,
                (
                    self.args.arch.num_cell * self.args.arch.num_cell * bs,
                    1,
                    self.args.data.img_h,
                    self.args.data.img_w,
                ),
                inverse=True,
            ).view(
                -1,
                self.args.arch.num_cell * self.args.arch.num_cell,
                1,
                self.args.data.img_h,
                self.args.data.img_w,
            )
            alpha_map = a_att_hat_full_res.sum(dim=1)
            # (bs, 1, img_h, img_w)
            alpha_map = (
                alpha_map
                + (
                    alpha_map.clamp(self.args.const.eps, 1 - self.args.const.eps)
                    - alpha_map
                ).detach()
            )

            if self.args.train.phase_bg_alpha_curriculum:
                if (
                    self.args.train.bg_alpha_curriculum_period[0]
                    < global_step
                    < self.args.train.bg_alpha_curriculum_period[1]
                ):
                    alpha_map = (
                        alpha_map.new_ones(alpha_map.size())
                        * self.args.train.bg_alpha_curriculum_value
                    )
                    # y_nobg = alpha_map * y_nobg
            y = y_nobg + (1.0 - alpha_map) * bg
        else:
            y_att = a_att * o_att

            o_att_full_res = spatial_transform(
                o_att,
                z_where,
                (
                    bs * self.args.arch.num_cell ** 2,
                    self.args.data.inp_channel,
                    self.args.data.img_h,
                    self.args.data.img_w,
                ),
                inverse=True,
            ).view(
                -1,
                self.args.arch.num_cell * self.args.arch.num_cell,
                self.args.data.inp_channel,
                self.args.data.img_h,
                self.args.data.img_w,
            )
            a_att_hat_full_res = spatial_transform(
                a_att * z_pres.view(bs * self.args.arch.num_cell ** 2, 1, 1, 1),
                z_where,
                (
                    self.args.arch.num_cell * self.args.arch.num_cell * bs,
                    1,
                    self.args.data.img_h,
                    self.args.data.img_w,
                ),
                inverse=True,
            ).view(
                -1,
                self.args.arch.num_cell * self.args.arch.num_cell,
                1,
                self.args.data.img_h,
                self.args.data.img_w,
            )

            # (self.args.arch.num_cell * self.args.arch.num_cell * bs, 3, img_h, img_w)
            y_att_full_res = spatial_transform(
                y_att,
                z_where,
                (
                    bs * self.args.arch.num_cell ** 2,
                    self.args.data.inp_channel,
                    self.args.data.img_h,
                    self.args.data.img_w,
                ),
                inverse=True,
            )
            y = (
                (
                    y_att_full_res
                    * z_pres.view(bs * self.args.arch.num_cell ** 2, 1, 1, 1)
                )
                .view(
                    bs,
                    -1,
                    self.args.data.inp_channel,
                    self.args.data.img_h,
                    self.args.data.img_w,
                )
                .sum(dim=1)
            )
            y_nobg = y
            alpha_map = y.new_ones(y.size(0), 1, y.size(2), y.size(3))

        return y, y_nobg, alpha_map, o_att_full_res, a_att_hat_full_res

    def loss_function(self, x, global_step):
        return self.forward(x, global_step)


def hyperparam_anneal(args, global_step):
    if args.train.beta_aux_pres_anneal_end_step == 0:
        args.train.beta_aux_pres = args.train.beta_aux_pres_anneal_start_value
    else:
        args.train.beta_aux_pres = linear_schedule(
            global_step,
            args.train.beta_aux_pres_anneal_start_step,
            args.train.beta_aux_pres_anneal_end_step,
            args.train.beta_aux_pres_anneal_start_value,
            args.train.beta_aux_pres_anneal_end_value,
        )

    if args.train.beta_aux_where_anneal_end_step == 0:
        args.train.beta_aux_where = args.train.beta_aux_where_anneal_start_value
    else:
        args.train.beta_aux_where = linear_schedule(
            global_step,
            args.train.beta_aux_where_anneal_start_step,
            args.train.beta_aux_where_anneal_end_step,
            args.train.beta_aux_where_anneal_start_value,
            args.train.beta_aux_where_anneal_end_value,
        )

    if args.train.beta_aux_what_anneal_end_step == 0:
        args.train.beta_aux_what = args.train.beta_aux_what_anneal_start_value
    else:
        args.train.beta_aux_what = linear_schedule(
            global_step,
            args.train.beta_aux_what_anneal_start_step,
            args.train.beta_aux_what_anneal_end_step,
            args.train.beta_aux_what_anneal_start_value,
            args.train.beta_aux_what_anneal_end_value,
        )

    if args.train.beta_aux_depth_anneal_end_step == 0:
        args.train.beta_aux_depth = args.train.beta_aux_depth_anneal_start_value
    else:
        args.train.beta_aux_depth = linear_schedule(
            global_step,
            args.train.beta_aux_depth_anneal_start_step,
            args.train.beta_aux_depth_anneal_end_step,
            args.train.beta_aux_depth_anneal_start_value,
            args.train.beta_aux_depth_anneal_end_value,
        )

    if args.train.beta_aux_global_anneal_end_step == 0:
        args.train.beta_aux_global = args.train.beta_aux_global_anneal_start_value
    else:
        args.train.beta_aux_global = linear_schedule(
            global_step,
            args.train.beta_aux_global_anneal_start_step,
            args.train.beta_aux_global_anneal_end_step,
            args.train.beta_aux_global_anneal_start_value,
            args.train.beta_aux_global_anneal_end_value,
        )

    if args.train.beta_aux_bg_anneal_end_step == 0:
        args.train.beta_aux_bg = args.train.beta_aux_bg_anneal_start_value
    else:
        args.train.beta_aux_bg = linear_schedule(
            global_step,
            args.train.beta_aux_bg_anneal_start_step,
            args.train.beta_aux_bg_anneal_end_step,
            args.train.beta_aux_bg_anneal_start_value,
            args.train.beta_aux_bg_anneal_end_value,
        )

    ########################### split here ###########################
    if args.train.beta_pres_anneal_end_step == 0:
        args.train.beta_pres = args.train.beta_pres_anneal_start_value
    else:
        args.train.beta_pres = linear_schedule(
            global_step,
            args.train.beta_pres_anneal_start_step,
            args.train.beta_pres_anneal_end_step,
            args.train.beta_pres_anneal_start_value,
            args.train.beta_pres_anneal_end_value,
        )

    if args.train.beta_where_anneal_end_step == 0:
        args.train.beta_where = args.train.beta_where_anneal_start_value
    else:
        args.train.beta_where = linear_schedule(
            global_step,
            args.train.beta_where_anneal_start_step,
            args.train.beta_where_anneal_end_step,
            args.train.beta_where_anneal_start_value,
            args.train.beta_where_anneal_end_value,
        )

    if args.train.beta_what_anneal_end_step == 0:
        args.train.beta_what = args.train.beta_what_anneal_start_value
    else:
        args.train.beta_what = linear_schedule(
            global_step,
            args.train.beta_what_anneal_start_step,
            args.train.beta_what_anneal_end_step,
            args.train.beta_what_anneal_start_value,
            args.train.beta_what_anneal_end_value,
        )

    if args.train.beta_depth_anneal_end_step == 0:
        args.train.beta_depth = args.train.beta_depth_anneal_start_value
    else:
        args.train.beta_depth = linear_schedule(
            global_step,
            args.train.beta_depth_anneal_start_step,
            args.train.beta_depth_anneal_end_step,
            args.train.beta_depth_anneal_start_value,
            args.train.beta_depth_anneal_end_value,
        )

    if args.train.beta_global_anneal_end_step == 0:
        args.train.beta_global = args.train.beta_global_anneal_start_value
    else:
        args.train.beta_global = linear_schedule(
            global_step,
            args.train.beta_global_anneal_start_step,
            args.train.beta_global_anneal_end_step,
            args.train.beta_global_anneal_start_value,
            args.train.beta_global_anneal_end_value,
        )

    if args.train.tau_pres_anneal_end_step == 0:
        args.train.tau_pres = args.train.tau_pres_anneal_start_value
    else:
        args.train.tau_pres = linear_schedule(
            global_step,
            args.train.tau_pres_anneal_start_step,
            args.train.tau_pres_anneal_end_step,
            args.train.tau_pres_anneal_start_value,
            args.train.tau_pres_anneal_end_value,
        )

    if args.train.beta_bg_anneal_end_step == 0:
        args.train.beta_bg = args.train.beta_bg_anneal_start_value
    else:
        args.train.beta_bg = linear_schedule(
            global_step,
            args.train.beta_bg_anneal_start_step,
            args.train.beta_bg_anneal_end_step,
            args.train.beta_bg_anneal_start_value,
            args.train.beta_bg_anneal_end_value,
        )

    return args
