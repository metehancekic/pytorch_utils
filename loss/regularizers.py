from typing import Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["hoyer_loss", "hoyer_channel_loss", "l1_loss", "l2_loss"]


def hoyer_loss(features: Dict[str, torch.Tensor] = {}, epsilon: float = 0.000000000001) -> torch.float:
    """
    Hoyer square loss: https://arxiv.org/pdf/1908.09979.pdf

    Takes a dictionary of tensors

    returns (|feature|_1)^2/((|feature|_2)^2 + epsilon)
    """
    loss = 0
    for feature in features:
        if features[feature].ndim == 4:
            l1_norm = torch.sum(torch.abs(features[feature]), dim=(0, 1, 2, 3))
            l2_norm_square = torch.sum(features[feature]**2, dim=(0, 1, 2, 3))
        elif features[feature].ndim == 2:
            l1_norm = torch.sum(torch.abs(features[feature]), dim=(0, 1))
            l2_norm_square = torch.sum(features[feature]**2, dim=(0, 1))
        else:
            raise NotImplementedError
        loss += torch.sum((l1_norm**2)/(l2_norm_square+epsilon))
    return loss


def hoyer_channel_loss(features: Dict[str, torch.Tensor] = {}, epsilon: float = 0.000000000001) -> torch.float:
    """
    Hoyer square loss on each channel: https://arxiv.org/pdf/1908.09979.pdf

    Takes a dictionary of tensors

    returns (|feature|_1)^2/((|feature|_2)^2 + epsilon)
    """
    loss = 0
    for feature in features:
        if features[feature].ndim == 4:
            l1_norm = torch.sum(torch.abs(features[feature]), dim=(0, 2, 3))
            l2_norm_square = torch.sum(features[feature]**2, dim=(0, 2, 3))
        elif features[feature].ndim == 2:
            l1_norm = torch.sum(torch.abs(features[feature]), dim=0)
            l2_norm_square = torch.sum(features[feature]**2, dim=0)
        else:
            raise NotImplementedError
        loss += torch.sum((l1_norm**2)/(l2_norm_square+epsilon))
    return loss


def l1_loss(features: Dict[str, torch.Tensor] = {}) -> torch.float:
    """
    L1 loss: L1 norm of the given tensors disctionary

    Takes a dictionary of tensors

    returns |feature|_1
    """
    loss = 0
    for feature in features:
        if features[feature].ndim == 4:
            l1_norm = torch.sum(torch.abs(features[feature]), dim=(0, 1, 2, 3))
        elif features[feature].ndim == 2:
            l1_norm = torch.sum(torch.abs(features[feature]), dim=(0, 1))
        elif features[feature].ndim == 3:
            l1_norm = torch.sum(torch.abs(features[feature]), dim=(0, 1, 2))
        else:
            raise NotImplementedError
        loss += l1_norm
    return loss


def l2_loss(features: Dict[str, torch.Tensor] = {}) -> torch.float:
    """
    L2 loss: L2 norm of the given tensors disctionary

    Takes a dictionary of tensors

    returns |feature|_2
    """
    loss = 0
    for feature in features:
        if features[feature].ndim == 4:
            l2_norm = torch.sqrt(
                torch.sum(features[feature]**2, dim=(0, 1, 2, 3)))
        elif features[feature].ndim == 2:
            l2_norm = torch.sqrt(torch.sum(features[feature]**2, dim=(0, 1)))
        elif features[feature].ndim == 3:
            l2_norm = torch.sqrt(torch.sum(features[feature]**2, dim=(0, 1, 2)))
        else:
            raise NotImplementedError
        loss += l2_norm
    return loss
