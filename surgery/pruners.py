from typing import Dict, Iterable, Callable

import torch
from torch import nn
import torch.nn.utils.prune as prune


def prune_model(model: nn.Module, amount: float = 0.6, permanent: bool = False):
    """
    Prunes given neural network

    Note: it only prunes convolutinal and linear layers
    for permanent deletion set permanent true
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            if permanent:
                prune.remove(module, "weight")

        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            if permanent:
                prune.remove(module, "weight")


def prune_layers(model: nn.Module, layer_names: Iterable[str] = ["conv1"], amount: float = 0.8, permanent: bool = False):
	"""
    Prunes specified layers inside given neural network
    """
    model_layer_dict = dict(model.named_modules())
    for layer_name in layer_names:
        module = model_layer_dict[layer_name]
        prune.l1_unstructured(module, name='weight', amount=amount)
        if permanent:
            prune.remove(module, "weight")
