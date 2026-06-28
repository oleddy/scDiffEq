"""Autoencoder + perturbation-conditioned latent SDE (scDiffEq VAE-SDE variant).

Mirrors scDiffEq's ``LightningSDE_VAE`` architecture — ``torch_nets.Encoder``
(genes -> latent), a latent neural SDE, ``torch_nets.Decoder`` (latent -> genes) —
but the latent SDE is the :class:`ConditionalNeuralSDE` whose drift and diffusion
are conditioned on the perturbation vector. Kept as a plain ``nn.Module`` so the
training loop (pretrain autoencoder, then per-condition Sinkhorn on decoded
predictions) can be driven explicitly.
"""
from __future__ import annotations

from typing import List, Union

import torch
import torch_nets
import torchsde

from ._conditional_sde import ConditionalNeuralSDE


class ConditionalSDE_VAE(torch.nn.Module):
    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        pert_dim: int,
        dt: float = 0.1,
        encoder_n_hidden: int = 4,
        decoder_n_hidden: int = 4,
        encoder_power: float = 2,
        decoder_power: float = 2,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        mu_hidden: Union[List[int], int] = [400, 400],
        sigma_hidden: Union[List[int], int] = [400, 400],
        coef_diffusion: float = 1.0,
        diffusion_floor: float = 0.0,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.Encoder = torch_nets.Encoder(
            data_dim=data_dim, latent_dim=latent_dim,
            n_hidden=encoder_n_hidden, power=encoder_power, dropout=encoder_dropout,
        )
        self.Decoder = torch_nets.Decoder(
            data_dim=data_dim, latent_dim=latent_dim,
            n_hidden=decoder_n_hidden, power=decoder_power, dropout=decoder_dropout,
        )
        self.DiffEq = ConditionalNeuralSDE(
            state_size=latent_dim, pert_dim=pert_dim,
            mu_hidden=mu_hidden, sigma_hidden=sigma_hidden, coef_diffusion=coef_diffusion,
            diffusion_floor=diffusion_floor,
        )

    # -- autoencoder: ---------------------------------------------------------
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.Encoder(X)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        return self.Decoder(Z)

    def reconstruct(self, X: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(X))

    # -- conditioned latent integration: --------------------------------------
    def integrate(self, Z0: torch.Tensor, pert: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Integrate the conditioned SDE. Returns latent path [len(ts), batch, latent]."""
        self.DiffEq.set_perturbation(pert)
        return torchsde.sdeint(self.DiffEq, Z0, ts, dt=self.dt)

    def simulate(self, X0: torch.Tensor, pert: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Source genes -> encode -> conditioned SDE -> decode endpoint. Returns predicted genes."""
        Z_path = self.integrate(self.encode(X0), pert, ts)
        return self.decode(Z_path[-1])
