from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

from slot_attention.data import CLEVRDataModule
from slot_attention.method import SlotAttentionMethod
from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import ImageLogCallback, rescale


def main(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    assert params.num_slots > 1, "Must have at least 2 slots."

    if params.is_verbose:
        print(
            f"INFO: limiting the dataset to only images with `num_slots - 1` ({params.num_slots - 1}) objects."
        )
        if params.num_train_images:
            print(
                f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_images}"
            )
        if params.num_val_images:
            print(
                f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_images}"
            )

    clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),  # rescale between -1 and 1
            transforms.Resize(params.resolution),
        ]
    )

    clevr_datamodule = CLEVRDataModule(
        data_root=params.data_root,
        max_n_objects=params.num_slots - 1,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clevr_transforms=clevr_transforms,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
    )

    print(
        f"Training set size (images must have {params.num_slots - 1} objects):",
        len(clevr_datamodule.train_dataset),
    )

    model = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
    )

    method = SlotAttentionMethod(
        model=model, datamodule=clevr_datamodule, params=params
    )

    logger = pl_loggers.WandbLogger(project="slot-attention-clevr6")

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator=params.accelerator,
        num_sanity_val_steps=params.num_sanity_val_steps,
        devices=params.devices,
        max_epochs=params.max_epochs,
        max_steps=params.max_steps,
        accumulate_grad_batches=params.accumulate_grad_batches,
        log_every_n_steps=50,
        callbacks=[
            LearningRateMonitor("step"),
            ImageLogCallback(),
        ]
        if params.is_logger_enabled
        else [],
    )
    trainer.fit(method, datamodule=clevr_datamodule)


if __name__ == "__main__":
    main()
