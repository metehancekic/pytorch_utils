from typing import Dict, Iterable, Callable
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

__all__ = ["get_lr", "count_parameter", "check_model_parameters_sparsity"]


def get_lr(optimizer: nn.optim.Optimizer) -> None:
    """
    Get the Learning rate from optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameter(model: nn.Module, logger: Callable = print, verbose: bool = True) -> int:
    """
    Outputs the number of trainable parameters on the model
    """

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    if verbose:
        logger(f" Number of total trainable parameters: {params}")
    return params


def check_model_parameters_sparsity(model: nn.Module, logger: Callable = print) -> None:
    """
    Check and print model layer's sparsity

    Note: prints only the sparsity of the convolutinal and linear layers
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            logger("Sparsity in {:}.weight: {:.2f}%".format(module, 100. *
                                                            float(torch.sum(module.weight == 0)) / float(module.weight.nelement())))
        elif isinstance(module, torch.nn.Linear):
            logger("Sparsity in {:}.weight: {:.2f}%".format(module, 100. *
                                                            float(torch.sum(module.weight == 0)) / float(module.weight.nelement())))
