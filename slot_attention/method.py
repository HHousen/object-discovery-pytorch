from argparse import Namespace
from typing import Union
from functools import partial
import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils

from slot_attention.slot_attention_model import SlotAttentionModel
from slot_attention.slate_model import SLATE
from slot_attention.utils import (
    to_rgb_from_tensor,
    warm_and_decay_lr_scheduler,
    cosine_anneal,
    linear_warmup,
    visualize,
    compute_ari,
)


class SlotAttentionMethod(pl.LightningModule):
    def __init__(
        self,
        model: Union[SlotAttentionModel, SLATE],
        datamodule: pl.LightningDataModule,
        params: Namespace,
    ):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params
        self.save_hyperparameters(params)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def step(self, batch):
        mask = None
        if self.params.model_type == "slate":
            self.tau = cosine_anneal(
                self.trainer.global_step,
                self.params.tau_start,
                self.params.tau_final,
                0,
                self.params.tau_steps,
            )
            loss = self.model.loss_function(batch, self.tau, self.params.hard)
        elif self.params.model_type == "sa":
            loss, mask = self.model.loss_function(batch)

        return loss, mask

    def training_step(self, batch, batch_idx):
        loss_dict, _ = self.step(batch)
        logs = {"train/" + key: val.item() for key, val in loss_dict.items()}
        self.log_dict(logs, sync_dist=True)
        return loss_dict["loss"]

    def sample_images(self, stage="validation"):
        dl = (
            self.datamodule.val_dataloader()
            if stage == "validation"
            else self.datamodule.train_dataloader()
        )
        perm = torch.randperm(self.params.batch_size)
        idx = perm[: self.params.n_samples]
        batch = next(iter(dl))
        if type(batch) == list:
            batch = batch[0][idx]
        else:
            batch = batch[idx]

        if self.params.accelerator:
            batch = batch.to(self.device)
        if self.params.model_type == "sa":
            recon_combined, recons, masks, slots = self.model.forward(batch)

            # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
            out = to_rgb_from_tensor(
                torch.cat(
                    [
                        batch.unsqueeze(1),  # original images
                        recon_combined.unsqueeze(1),  # reconstructions
                        recons * masks + (1 - masks),  # each slot
                    ],
                    dim=1,
                )
            )

            batch_size, num_slots, C, H, W = recons.shape
            images = vutils.make_grid(
                out.view(batch_size * out.shape[1], C, H, W).cpu(),
                normalize=False,
                nrow=out.shape[1],
            )
        elif self.params.model_type == "slate":
            recon, _, _, attns = self.model(batch, self.tau, True)
            gen_img = self.model.reconstruct_autoregressive(batch)
            vis_recon = visualize(batch, recon, gen_img, attns, N=32)
            images = vutils.make_grid(
                vis_recon, nrow=self.params.num_slots + 3, pad_value=0.2
            )[:, 2:-2, 2:-2]

        return images

    def validation_step(self, batch, batch_idx):
        if type(batch) == list and self.model.supports_masks:
            loss, predicted_mask = self.step(batch[0])
            predicted_mask = torch.permute(predicted_mask, [0, 3, 4, 2, 1])
            # `predicted_mask` has shape [batch_size, height, width, channels, num_entries]
            predicted_mask = torch.squeeze(predicted_mask)
            batch_size, height, width, num_entries = predicted_mask.shape
            predicted_mask = torch.reshape(
                predicted_mask, [batch_size, height * width, num_entries]
            )
            # `predicted_mask` has shape [batch_size, height * width, num_entries]
            # Scale from [0, 1] to [0, 255] to match the true mask.
            predicted_mask = (predicted_mask * 255).type(torch.int)
            ari = compute_ari(
                predicted_mask,
                batch[1],
                len(batch[0]),
                self.params.resolution[0],
                self.params.resolution[1],
                self.datamodule.max_num_entries,
            )
            loss["ari"] = ari
            return loss
        else:
            if type(batch) == list:
                batch = batch[0]
            loss, _ = self.step(batch)
            return loss

    def validation_epoch_end(self, outputs):
        logs = {
            "validation/" + key: torch.stack([x[key] for x in outputs]).mean()
            for key in outputs[0].keys()
        }
        self.log_dict(logs, sync_dist=True)

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        # https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-774265729
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.datamodule.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self):
        if self.params.model_type == "slate":
            optimizer = optim.Adam(
                [
                    {
                        "params": (
                            x[1]
                            for x in self.model.named_parameters()
                            if "dvae" in x[0]
                        ),
                        "lr": self.params.lr_dvae,
                    },
                    {
                        "params": (
                            x[1]
                            for x in self.model.named_parameters()
                            if "dvae" not in x[0]
                        ),
                        "lr": self.params.lr_main,
                    },
                ],
                weight_decay=self.params.weight_decay,
            )
        elif self.params.model_type == "sa":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.params.lr_main,
                weight_decay=self.params.weight_decay,
            )

        total_steps = self.num_training_steps()

        if self.params.scheduler == "warmup_and_decay":
            warmup_steps_pct = self.params.warmup_steps_pct
            decay_steps_pct = self.params.decay_steps_pct
            scheduler_lambda = partial(
                warm_and_decay_lr_scheduler,
                warmup_steps_pct=warmup_steps_pct,
                decay_steps_pct=decay_steps_pct,
                total_steps=total_steps,
                gamma=self.params.scheduler_gamma,
            )
        elif self.params.scheduler == "warmup":
            scheduler_lambda = partial(
                linear_warmup,
                start_value=0.0,
                final_value=1.0,
                start_step=0,
                final_step=self.params.lr_warmup_steps,
            )

        if self.params.model_type == "slate":
            lr_lambda = [lambda o: 1, scheduler_lambda]
        elif self.params.model_type == "sa":
            lr_lambda = scheduler_lambda
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lr_lambda
        )

        if self.params.model_type == "slate" and hasattr(self.params, "patience"):
            reduce_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=0.5,
                patience=self.params.patience,
            )
            return (
                [optimizer],
                [
                    {"scheduler": scheduler, "interval": "step",},
                    {
                        "scheduler": reduce_on_plateau,
                        "interval": "epoch",
                        "monitor": "validation/loss",
                    },
                ],
            )

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )
