
from .diceloss import DiceLoss


__all__ = [
    'DiceLoss'
]

__LOSS = {
    'DiceLoss': DiceLoss
}


def build_loss(name, *args, **kwargs):
    if name not in __LOSS:
        raise KeyError(f"Unknown Loss: {name}")
    return __LOSS[name](*args, **kwargs)
