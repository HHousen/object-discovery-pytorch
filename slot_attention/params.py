from argparse import Namespace


training_params = Namespace(
    model_type = "slate",
    num_slots = 7,
    num_iterations = 3,
    accumulate_grad_batches = 1,
    data_root = "/media/Main/Downloads/CLEVR_v1.0",
    accelerator = "gpu",
    devices = -1,
    max_steps = -1,
    num_sanity_val_steps = 1,
    num_train_images = None,
    num_val_images = None,
    num_workers = 4,
    is_logger_enabled = True,
    gradient_clip_val = 0.0,
    n_samples = 16,
)

slot_attention_params = Namespace(
    lr_main = 4e-4,
    batch_size = 32,
    val_batch_size = 32,
    resolution = (128, 128),
    slot_size = 64,
    max_epochs = 100,
    max_steps = 500_000 * 2,
    accumulate_grad_batches = 2,
    weight_decay = 0.0,
    mlp_hidden_size = 128,
    scheduler = "warmup_and_decay",
    scheduler_gamma = 0.5,
    warmup_steps_pct = 0.02,
    decay_steps_pct = 0.2,
)

slate_params = Namespace(
    lr_dvae = 3e-4,
    lr_main = 1e-4,
    weight_decay = 0.0,
    batch_size = 12,
    val_batch_size = 12,
    accumulate_grad_batches = 3,
    max_epochs = 1000,
    # patience = 4,
    gradient_clip_val = 1.0,
    resolution = (128, 128),
    num_dec_blocks = 6,
    vocab_size = 4096,
    d_model = 192,
    num_heads = 6,
    dropout = 0.1,
    slot_size = 192,
    mlp_hidden_size = 192,
    tau_start = 1.0,
    tau_final = 0.1,
    tau_steps = 30000,
    scheduler = "warmup",
    lr_warmup_steps = 30000,
    hard = False,
)

def merge_namespaces(one: Namespace, two: Namespace):
    return Namespace(**{**vars(one), **vars(two)})

