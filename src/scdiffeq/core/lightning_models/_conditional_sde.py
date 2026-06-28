"""Perturbation-conditioned neural SDE.

Adds a (constant-along-trajectory) perturbation vector ``p`` as an additional
input to BOTH the drift and diffusion networks, so the learned dynamics depend
on the experimental condition. Mirrors ``neural_diffeqs.NeuralSDE`` /
``DiffEqConfig`` exactly, except the drift/diffusion MLPs take ``state_size +
pert_dim`` inputs and a per-cell ``p`` is concatenated to the state inside
``drift``/``diffusion``.

torchsde calls ``f(t, y)`` / ``g(t, y)`` with ``y`` of shape ``[batch, state]``
and preserves batch order across solver steps, so the buffer ``self._pert`` set
once before integration stays row-aligned with ``y`` throughout.

Diagonal noise (``noise_type="diagonal"``) is used so the diffusion is an
expressive per-dimension term (g: ``[batch, state]``) — better suited to matching
each target's spread than scDiffEq's default rank-1 ``general`` noise.
"""
from __future__ import annotations

from typing import List, Union

import torch
import torch_nets


class ConditionalNeuralSDE(torch.nn.Module):
    def __init__(
        self,
        state_size: int,
        pert_dim: int,
        mu_hidden: Union[List[int], int] = [400, 400],
        sigma_hidden: Union[List[int], int] = [400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        sigma_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.1,
        sigma_dropout: Union[float, List[float]] = 0.1,
        mu_bias: bool = True,
        sigma_bias: bool = True,
        mu_output_bias: bool = True,
        sigma_output_bias: bool = True,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
        diffusion_floor: float = 0.0,
        sde_type: str = "ito",
        noise_type: str = "diagonal",
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.pert_dim = pert_dim
        self.sde_type = sde_type      # required by torchsde
        self.noise_type = noise_type  # required by torchsde
        self._coef_drift = coef_drift
        self._coef_diffusion = coef_diffusion
        self._diffusion_floor = diffusion_floor   # additive noise the network cannot zero out

        self.mu = torch_nets.TorchNet(
            in_features=state_size + pert_dim, out_features=state_size,
            hidden=mu_hidden, activation=mu_activation, dropout=mu_dropout,
            bias=mu_bias, output_bias=mu_output_bias,
        )
        self.sigma = torch_nets.TorchNet(
            in_features=state_size + pert_dim, out_features=state_size,
            hidden=sigma_hidden, activation=sigma_activation, dropout=sigma_dropout,
            bias=sigma_bias, output_bias=sigma_output_bias,
        )
        self._pert: torch.Tensor | None = None

    # -- conditioning: --------------------------------------------------------
    def set_perturbation(self, pert: torch.Tensor) -> None:
        """pert: [batch, pert_dim], row-aligned with the y0 passed to sdeint."""
        self._pert = pert

    def _condition(self, y: torch.Tensor) -> torch.Tensor:
        if self._pert is None:
            raise RuntimeError("set_perturbation(p) must be called before integration.")
        return torch.cat([y, self._pert], dim=-1)

    # -- drift / diffusion: ---------------------------------------------------
    def drift(self, y: torch.Tensor) -> torch.Tensor:
        return self.mu(self._condition(y)) * self._coef_drift

    def diffusion(self, y: torch.Tensor) -> torch.Tensor:
        # softplus keeps the diagonal diffusion non-negative; the floor is an additive
        # minimum noise level the network cannot suppress to zero (it otherwise does).
        sig = torch.nn.functional.softplus(self.sigma(self._condition(y))) * self._coef_diffusion
        return sig + self._diffusion_floor

    # -- torchsde interface: --------------------------------------------------
    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.drift(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.diffusion(y)
