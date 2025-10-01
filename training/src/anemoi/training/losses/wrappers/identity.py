# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
from torch import nn

from anemoi.training.losses.wrappers.base import BaseLossWrapper


class IdentityWrapper(BaseLossWrapper):
    """Identity wrapper that passes through to the underlying loss function.

    This wrapper provides no additional functionality and is used as the
    default when no special loss computation is needed.
    """

    def __init__(self) -> None:
        """Initialize the identity wrapper."""
        super().__init__()

    @property
    def num_outputs(self) -> int:
        """Return the number of loss outputs.

        Returns
        -------
        int
            Always returns 1 for identity wrapper.
        """
        return 1

    def forward(
        self,
        loss_fn: nn.Module,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Pass through to the underlying loss function.

        Parameters
        ----------
        loss_fn : nn.Module
            The underlying loss function
        y_pred : torch.Tensor
            Predicted tensor
        y : torch.Tensor
            Target tensor
        **kwargs
            Additional keyword arguments passed to the loss function

        Returns
        -------
        torch.Tensor
            The computed loss value
        """
        return loss_fn(y_pred, y, **kwargs)
