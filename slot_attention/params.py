from typing import Optional
from typing import Tuple
from argparse import Namespace


class TrainingParams(Namespace):
    model_type = "slate"
    num_slots: int = 7
    num_iterations: int = 3
    accumulate_grad_batches: int = 1
    data_root: str = "/media/Main/Downloads/CLEVR_v1.0"
    accelerator: str = "gpu"
    devices: int = -1
    max_steps: int = -1
    num_sanity_val_steps: int = 1
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    num_workers: int = 4
    is_logger_enabled: bool = True
    gradient_clip_val: int = 0.0
    n_samples: int = 16


class SlotAttentionParams(TrainingParams):
    lr_main: float = 4e-4
    batch_size: int = 32
    val_batch_size: int = 32
    resolution: Tuple[int, int] = (128, 128)
    slot_size: int = 64
    max_epochs: int = 100
    max_steps: int = 500_000 * 2
    accumulate_grad_batches: int = 2
    weight_decay: float = 0.0
    mlp_hidden_size = 128
    scheduler: str = "warmup_and_decay"
    scheduler_gamma: float = 0.5
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2


class SLATEParams(TrainingParams):
    lr_dvae: float = 3e-4
    lr_main: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 12
    val_batch_size: int = 12
    max_epochs: int = 1000
    # patience: int = 4  # not implemented
    gradient_clip_val: float = 1.0
    resolution: Tuple[int, int] = (128, 128)
    num_dec_blocks: int = 8
    vocab_size: int = 4096
    d_model: int = 192
    num_heads: int = 8
    dropout: float = 0.1
    slot_size: int = 192
    mlp_hidden_size: int = 192
    tau_start: float = 1.0
    tau_final: float = 0.1
    tau_steps: int = 30000
    scheduler: str = "warmup"
    lr_warmup_steps: int = 30000
    hard: bool = False
