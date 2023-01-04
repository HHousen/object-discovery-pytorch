import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from slot_attention.data import CLEVRDataModule, Shapes3dDataModule
from slot_attention.method import SlotAttentionMethod
from slot_attention.slot_attention_model import SlotAttentionModel
from slot_attention.slate_model import SLATE
from slot_attention.params import merge_namespaces, training_params, slot_attention_params, slate_params
from slot_attention.utils import ImageLogCallback


def main(params = None):
    if params is None:
        params = training_params
        if params.model_type == "slate":
            params = merge_namespaces(params, slate_params)
        elif params.model_type == "sa":
            params = merge_namespaces(params, slot_attention_params)

    assert params.num_slots > 1, "Must have at least 2 slots."

    clevr_datamodule = CLEVRDataModule(
        data_root=params.data_root,
        max_n_objects=params.num_slots - 1,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
        resolution=params.resolution,
    )
    # clevr_datamodule = Shapes3dDataModule(
    #     data_root=params.data_root,
    #     train_batch_size=params.batch_size,
    #     val_batch_size=params.val_batch_size,
    #     num_workers=params.num_workers,
    # )

    print(
        f"Training set size (images must have {params.num_slots - 1} objects):",
        len(clevr_datamodule.train_dataset),
    )

    if params.model_type == "sa":
        model = SlotAttentionModel(
            resolution=params.resolution,
            num_slots=params.num_slots,
            num_iterations=params.num_iterations,
            slot_size=params.slot_size,
        )
    elif params.model_type == "slate":
        model = SLATE(
            num_slots=params.num_slots,
            vocab_size=params.vocab_size,
            d_model=params.d_model,
            resolution=params.resolution,
            num_iterations=params.num_iterations,
            slot_size=params.slot_size,
            mlp_hidden_size=params.mlp_hidden_size,
            num_heads=params.num_heads,
            dropout=params.dropout,
            num_dec_blocks=params.num_dec_blocks,
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
        gradient_clip_val=params.gradient_clip_val,
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
