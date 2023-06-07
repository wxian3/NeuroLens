from typing import Callable
import torch


def batched_func(func: Callable, tensor: torch.Tensor, batch_size: int):
    return torch.cat(tuple(func(batch) for batch in batched(tensor, batch_size)), dim=0)


class batched(object):
    def __init__(self, tensor: torch.Tensor, batch_size: int):
        self.tensor = tensor
        self.batch_size = batch_size
        self._position = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self._position + 1 >= self.tensor.shape[0]:
            raise StopIteration()
        next_batch = self.tensor[
            self._position + 1 : self._position + 1 + self.batch_size
        ]
        self._position += self.batch_size
        return next_batch
