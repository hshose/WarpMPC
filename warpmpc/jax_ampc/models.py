"""Standard neural policy models provided by ``warpmpc.jax_ampc``."""

from __future__ import annotations

from flax import linen as nn


class MLP(nn.Module):
    """A small fully connected network for AMPC imitation targets."""

    hidden_sizes: tuple[int, ...] = (32, 32, 32)
    output_dim: int = 4
    activation: str = "leaky_relu"
    negative_slope: float = 0.01

    def _activate(self, x):
        if self.activation == "leaky_relu":
            return nn.leaky_relu(x, negative_slope=self.negative_slope)
        if self.activation == "relu":
            return nn.relu(x)
        if self.activation == "tanh":
            return nn.tanh(x)
        if self.activation == "gelu":
            return nn.gelu(x)
        if self.activation == "elu":
            return nn.elu(x)
        if self.activation in ("silu", "swish"):
            return nn.silu(x)
        raise ValueError(f"unsupported activation {self.activation!r}")

    @nn.compact
    def __call__(self, x):
        for width in self.hidden_sizes:
            x = nn.Dense(width)(x)
            x = self._activate(x)
        return nn.Dense(self.output_dim)(x)
