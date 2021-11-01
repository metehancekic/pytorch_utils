from typing import Dict, Iterable, Callable
from torch import nn, Tensor
import torch


__all__ = ["LayerOutputExtractor_ctxmgr", "LayerOutputExtractor_wrapper"]


class LayerOutputExtractor_ctxmgr():
    def __init__(self, model: nn.Module, layer_names: Iterable[str]):
        self.model = model
        self.layer_names = layer_names
        self.layer_outputs = {layer: torch.empty(0) for layer in layer_names}
        self.hook_handles = {}

    def __enter__(self):
        model_layer_dict = dict(self.model.named_modules())
        for layer_id in self.layer_names:
            layer = model_layer_dict[layer_id]
            self.hook_handles[layer_id] = layer.register_forward_hook(
                self.generate_hook_fn(layer_id))
        return self

    def __exit__(self, *_):
        [hook_handle.remove() for hook_handle in self.hook_handles.values()]

    def generate_hook_fn(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self.layer_outputs[layer_id] = output
        return fn


class LayerOutputExtractor_wrapper(nn.Module):
    def __init__(self, model: nn.Module, layer_names: Iterable[str]):
        super().__init__()
        self._model = model
        # self.__class__ = type(model.__class__.__name__,
        #                       (self.__class__, model.__class__),
        #                       {})
        # self.__dict__ = model.__dict__

        self.layer_outputs = {layer: torch.empty(0) for layer in layer_names}
        self.hook_handles = {}
        model_layer_dict = dict([*model.named_modules()])

        for layer_id in layer_names:
            layer = model_layer_dict[layer_id]
            self.hook_handles[layer_id] = layer.register_forward_hook(
                self.generate_hook_fn(layer_id))

    def generate_hook_fn(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self.layer_outputs[layer_id] = output
        return fn

    def close(self):
        [hook_handle.remove() for hook_handle in self.hook_handles.values()]

    def forward(self, x):
        return self._model(x)

    def __getattribute__(self, name: str):
        # the last three are used in nn.Module.__setattr__
        if name in ["_model", "layer_outputs", "hook_handles", "generate_hook_fn", "close", "__dict__", "_parameters", "_buffers", "_non_persistent_buffers_set"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self._model, name)


def main():
    model = torch.nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
    model = LayerOutputExtractor_wrapper(model, ["0"])


if __name__ == '__main__':
    main()
