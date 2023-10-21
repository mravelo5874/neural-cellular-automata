# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from pydantic.dataclasses import dataclass
from omegaconf import MISSING  # Do not confuse with dataclass.MISSING
from typing import Dict, Any, Optional, List

"""
    Optimizer + Scheduler config
"""
@dataclass
class AdamConf:
    _target_: str = "torch.optim.adam.Adam"
    params: Any = MISSING
    lr: Any = 0.001
    betas: Any = (0.9, 0.999)
    eps: Any = 1e-08
    weight_decay: Any = 0
    amsgrad: Any = False

@dataclass
class LambdaLRConf:
    _target_: str = "torch.optim.lr_scheduler.LambdaLR"
    optimizer: Any = MISSING
    lr_lambda: Any = MISSING
    last_epoch: Any = -1


@dataclass
class MultiplicativeLRConf:
    _target_: str = "torch.optim.lr_scheduler.MultiplicativeLR"
    optimizer: Any = MISSING
    lr_lambda: Any = MISSING
    last_epoch: Any = -1


@dataclass
class StepLRConf:
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    optimizer: Any = MISSING
    step_size: Any = 0.1
    gamma: Any = 0.1
    last_epoch: Any = -1


@dataclass
class MultiStepLRConf:
    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    optimizer: Any = MISSING
    milestones: Any = MISSING
    gamma: Any = 0.1
    last_epoch: Any = -1


@dataclass
class ExponentialLRConf:
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    optimizer: Any = MISSING
    gamma: Any = 0.9999
    last_epoch: Any = -1


@dataclass
class CosineAnnealingLRConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    optimizer: Any = MISSING
    T_max: Any = MISSING
    eta_min: Any = 0
    last_epoch: Any = -1


@dataclass
class ReduceLROnPlateauConf:
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    optimizer: Any = MISSING
    mode: Any = "min"
    factor: Any = 0.1
    patience: Any = 10
    verbose: Any = False
    threshold: Any = 0.0001
    threshold_mode: Any = "rel"
    cooldown: Any = 0
    min_lr: Any = 0
    eps: Any = 1e-08


@dataclass
class CyclicLRConf:
    _target_: str = "torch.optim.lr_scheduler.CyclicLR"
    optimizer: Any = MISSING
    base_lr: Any = MISSING
    max_lr: Any = MISSING
    step_size_up: Any = 2000
    step_size_down: Any = None
    mode: Any = "triangular"
    gamma: Any = 1.0
    scale_fn: Any = None
    scale_mode: Any = "cycle"
    cycle_momentum: Any = True
    base_momentum: Any = 0.8
    max_momentum: Any = 0.9
    last_epoch: Any = -1


@dataclass
class CosineAnnealingWarmRestartsConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    optimizer: Any = MISSING
    T_0: Any = MISSING
    T_mult: Any = 1
    eta_min: Any = 0
    last_epoch: Any = -1


@dataclass
class OneCycleLRConf:
    _target_: str = "torch.optim.lr_scheduler.OneCycleLR"
    optimizer: Any = MISSING
    max_lr: Any = MISSING
    total_steps: Any = None
    epochs: Any = None
    steps_per_epoch: Any = None
    pct_start: Any = 0.3
    anneal_strategy: Any = "cos"
    cycle_momentum: Any = True
    base_momentum: Any = 0.85
    max_momentum: Any = 0.95
    div_factor: Any = 25.0
    final_div_factor: Any = 10000.0
    last_epoch: Any = -1

"""
    Dataset + Dataloader Config
"""
@dataclass
class DataLoaderConf:
    _target_: str = "torch.utils.data.dataloader.DataLoader"
    dataset: Any = MISSING
    batch_size: Any = 1
    shuffle: Any = False
    sampler: Any = None
    batch_sampler: Any = None
    num_workers: Any = 0
    collate_fn: Any = None
    pin_memory: Any = False
    drop_last: Any = False
    timeout: Any = 0
    worker_init_fn: Any = None
    multiprocessing_context: Any = None
    generator: Any = None

@dataclass
class DatasetConf:
    _target_: str = "torch.utils.data.dataset.Dataset"


@dataclass
class ChainDatasetConf:
    _target_: str = "torch.utils.data.dataset.ChainDataset"
    datasets: Any = MISSING


@dataclass
class ConcatDatasetConf:
    _target_: str = "torch.utils.data.dataset.ConcatDataset"
    datasets: Any = MISSING


@dataclass
class IterableDatasetConf:
    _target_: str = "torch.utils.data.dataset.IterableDataset"


@dataclass
class TensorDatasetConf:
    _target_: str = "torch.utils.data.dataset.TensorDataset"
    tensors: Any = MISSING


@dataclass
class SubsetConf:
    _target_: str = "torch.utils.data.dataset.Subset"
    dataset: Any = MISSING
    indices: Any = MISSING
