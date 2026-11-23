"""Weight initialisation, the same for every part of the model.

    Linear      Xavier (Glorot) uniform weights and zero biases: the original Transformer's choice
                and what torch.nn.MultiheadAttention uses. It keeps a signal's size steady going
                forward and gradients' size steady going back (Glorot & Bengio 2010).
    LayerNorm   gain 1, bias 0: each norm starts as a plain standardisation.

Every module calls ``self.apply(init_weights)`` at the end of its constructor. A module with a
reason to start one layer differently sets that layer itself afterwards, and says why.
"""

from torch import nn


def init_weights(module: nn.Module) -> None:
    """Initialise one module's own weights (use with ``Module.apply``)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
