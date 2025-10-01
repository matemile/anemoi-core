# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod

import torch
from torch import nn


class BaseLossWrapper(nn.Module, ABC):
    """Abstract base class for loss wrappers.

    Loss wrappers provide additional functionality around loss functions,
    such as multi-scale computation, temporal weighting, or adaptive scaling.
    """

    def __init__(self) -> None:
        """Initialize the base loss wrapper."""
        super().__init__()

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        """Return the number of loss outputs.

        Returns
        -------
        int
            Number of loss values that will be computed.
            1 for standard losses, >1 for multi-scale losses.
        """

    @abstractmethod
    def forward(
        self,
        loss_fn: nn.Module,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Apply the wrapper to compute the loss.

        Parameters
        ----------
        loss_fn : nn.Module
            The underlying loss function to wrap
        y_pred : torch.Tensor
            Predicted tensor
        y : torch.Tensor
            Target tensor
        **kwargs
            Additional keyword arguments passed to the loss function

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            The computed loss value(s)
        """

    def prepare_inputs(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare inputs before loss computation.

        Can be overridden by subclasses to perform preprocessing.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted tensor
        y : torch.Tensor
            Target tensor

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Processed prediction and target tensors
        """
        return y_pred, y

    def aggregate_losses(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate multiple loss values into a single scalar.

        Default implementation sums the losses.

        Parameters
        ----------
        losses : list[torch.Tensor]
            List of loss tensors to aggregate

        Returns
        -------
        torch.Tensor
            Aggregated loss value
        """
        return torch.stack(losses).sum()
