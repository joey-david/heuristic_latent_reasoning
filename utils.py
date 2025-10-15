# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_transformers_no_torchvision() -> None:
    """
    Prevent transformers from importing torchvision, which can require CUDA ops
    missing in the execution environment.
    """
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
    try:
        from transformers.utils import import_utils as _import_utils
    except Exception:
        return

    try:
        _import_utils._torchvision_available = False
        _import_utils._torchvision_version = "N/A"
    except Exception:
        pass
