import typing

import torch

def no_grad(fun: typing.Callable)->typing.Callable:
    """
    Decorator function to disable gradients.

    Usage:
        @no_grad
        def fun(*args, **kwargs):
            ...
    """
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fun(*args, **kwargs)

    return wrapper