from argparse import Namespace


training_params = Namespace(
    model_type="sa",
    dataset="boxworld",
    num_slots=10,
    num_iterations=3,
    accumulate_grad_batches=1,
    data_root="data/box_world_dataset.h5",
    accelerator="gpu",
    devices=-1,
    max_steps=-1,
    num_sanity_val_steps=1,
    num_workers=4,
    is_logger_enabled=True,
    gradient_clip_val=0.0,
    n_samples=16,
    clevrtex_dataset_variant="full",
    alternative_crop=True,  # Alternative crop for RAVENS dataset
)

slot_attention_params = Namespace(
    lr_main=4e-4,
    batch_size=64,
    val_batch_size=64,
    resolution=(128, 128),
    slot_size=64,
    max_epochs=1000,
    max_steps=500000,
    weight_decay=0.0,
    mlp_hidden_size=128,
    scheduler="warmup_and_decay",
    scheduler_gamma=0.5,
    warmup_steps_pct=0.02,
    decay_steps_pct=0.2,
    use_separation_loss="entropy",
    separation_tau_start=60_000,
    separation_tau_end=65_000,
    separation_tau_max_val=0.003,
    separation_tau=None,
    boxworld_group_objects=True,
    use_area_loss=True,
    area_tau_start=60_000,
    area_tau_end=65_000,
    area_tau_max_val=0.006,
    area_tau=None,
)

slate_params = Namespace(
    lr_dvae=3e-4,
    lr_main=1e-4,
    weight_decay=0.0,
    batch_size=50,
    val_batch_size=50,
    max_epochs=1000,
    patience=4,
    gradient_clip_val=1.0,
    resolution=(128, 128),
    num_dec_blocks=8,
    vocab_size=4096,
    d_model=192,
    num_heads=8,
    dropout=0.1,
    slot_size=192,
    mlp_hidden_size=192,
    tau_start=1.0,
    tau_final=0.1,
    tau_steps=30000,
    scheduler="warmup",
    lr_warmup_steps=30000,
    hard=False,
)

gnm_params = Namespace(
    std=0.2,  # 0.4 on CLEVR, 0.7 on ClevrTex, 0.2/0.3 on RAVENS
    z_what_dim=64,
    z_bg_dim=10,
    lr_main=1e-4,
    batch_size=64,
    val_batch_size=64,
    resolution=(128, 128),
    gradient_clip_val=1.0,
    max_epochs=1000,
    max_steps=5_000_000,
    weight_decay=0.0,
    scheduler=None,
)


def merge_namespaces(one: Namespace, two: Namespace):
    return Namespace(**{**vars(one), **vars(two)})
