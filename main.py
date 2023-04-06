import inspect

import torch.nn as nn

from src.models.modules import MultiheadAttention


class TrainerBase:
    r"""
    Base abstract class for the training. To implement a certain training procedure
        extent this class and implement the abstract methods 'process_batch' and
        'evaluate'.
    """


class yo(nn.Module):
    r"""pass"""

    def __init__(self, a, b) -> None:
        super().__init__()


ba = TrainerBase()
ma = yo()
print(inspect.getdoc(ma))
