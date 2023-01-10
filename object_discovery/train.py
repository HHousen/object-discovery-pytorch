import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from object_discovery.data import (
    CLEVRDataModule,
    Shapes3dDataModule,
    RAVENSRobotDataModule,
    SketchyDataModule,
    ClevrTexDataModule,
)
from object_discovery.method import SlotAttentionMethod
from object_discovery.slot_attention_model import SlotAttentionModel
from object_discovery.slate_model import SLATE
from object_discovery.params import (
    merge_namespaces,
    training_params,
    slot_attention_params,
    slate_params,
    gnm_params,
)
from object_discovery.utils import ImageLogCallback
from object_discovery.gnm.gnm_model import GNM, hyperparam_anneal
from object_discovery.gnm.config import get_arrow_args


def main(params=None):
    if params is None:
        params = training_params
        if params.model_type == "slate":
            params = merge_namespaces(params, slate_params)
        elif params.model_type == "sa":
            params = merge_namespaces(params, slot_attention_params)
        elif params.model_type == "gnm":
            params = merge_namespaces(params, gnm_params)

    assert params.num_slots > 1, "Must have at least 2 slots."
    params.neg_1_to_pos_1_scale = params.model_type == "sa"

    if params.dataset == "clevr":
        datamodule = CLEVRDataModule(
            data_root=params.data_root,
            max_n_objects=params.num_slots - 1,
            train_batch_size=params.batch_size,
            val_batch_size=params.val_batch_size,
            num_workers=params.num_workers,
            resolution=params.resolution,
            neg_1_to_pos_1_scale=params.neg_1_to_pos_1_scale,
        )
    elif params.dataset == "shapes3d":
        assert params.resolution == (
            64,
            64,
        ), "shapes3d dataset requires 64x64 resolution"
        datamodule = Shapes3dDataModule(
            data_root=params.data_root,
            train_batch_size=params.batch_size,
            val_batch_size=params.val_batch_size,
            num_workers=params.num_workers,
            neg_1_to_pos_1_scale=params.neg_1_to_pos_1_scale,
        )
    elif params.dataset == "ravens":
        datamodule = RAVENSRobotDataModule(
            data_root=params.data_root,
            # `max_n_objects` is the number of objects on the table. It does
            # not count the background, table, robot, and robot arm.
            max_n_objects=params.num_slots - 1
            if params.alternative_crop
            else params.num_slots - 4,
            train_batch_size=params.batch_size,
            val_batch_size=params.val_batch_size,
            num_workers=params.num_workers,
            resolution=params.resolution,
            alternative_crop=params.alternative_crop,
            neg_1_to_pos_1_scale=params.neg_1_to_pos_1_scale,
        )
    elif params.dataset == "sketchy":
        assert params.resolution == (
            128,
            128,
        ), "sketchy dataset requires 128x128 resolution"
        datamodule = SketchyDataModule(
            data_root=params.data_root,
            train_batch_size=params.batch_size,
            val_batch_size=params.val_batch_size,
            num_workers=params.num_workers,
            neg_1_to_pos_1_scale=params.neg_1_to_pos_1_scale,
        )
    elif params.dataset == "clevrtex":
        datamodule = ClevrTexDataModule(
            data_root=params.data_root,
            train_batch_size=params.batch_size,
            val_batch_size=params.val_batch_size,
            num_workers=params.num_workers,
            resolution=params.resolution,
            neg_1_to_pos_1_scale=params.neg_1_to_pos_1_scale,
            dataset_variant=params.clevrtex_dataset_variant,
            max_n_objects=params.num_slots - 1,
        )

    print(
        f"Training set size (images must have {params.num_slots - 1} objects):",
        len(datamodule.train_dataset),
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
    elif params.model_type == "gnm":
        model_params = get_arrow_args()
        model_params = hyperparam_anneal(model_params, 0)
        params = merge_namespaces(params, model_params)
        params.const.likelihood_sigma = params.std
        params.z.z_what_dim = params.z_what_dim
        params.z.z_bg_dim = params.z_bg_dim
        model = GNM(params)

    method = SlotAttentionMethod(model=model, datamodule=datamodule, params=params)

    logger = pl_loggers.WandbLogger(project="slot-attention-clevr6")

    callbacks = [LearningRateMonitor("step")]
    if params.model_type != "gnm":
        callbacks.append(ImageLogCallback())

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
        callbacks=callbacks if params.is_logger_enabled else [],
    )
    trainer.fit(method, datamodule=datamodule)


if __name__ == "__main__":
    main()
