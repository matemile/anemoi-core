# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.losses.wrappers.base import BaseLossWrapper
from anemoi.training.losses.wrappers.identity import IdentityWrapper
from anemoi.training.losses.wrappers.multiscale import MultiScaleWrapper

LOGGER = logging.getLogger(__name__)


def get_loss_wrapper(config: DictConfig | dict | None = None) -> BaseLossWrapper:
    """Get a loss wrapper based on configuration.

    Parameters
    ----------
    config : DictConfig | dict | None, optional
        Configuration for the loss wrapper. If None, returns IdentityWrapper.
        Can be either:
        - A dictionary/DictConfig with '_target_' key for Hydra instantiation
        - A dictionary with 'type' key and additional parameters
        - None for default IdentityWrapper

    Returns
    -------
    BaseLossWrapper
        The instantiated loss wrapper

    Raises
    ------
    ValueError
        If the wrapper type is not recognized or configuration is invalid
    TypeError
        If the instantiated object is not a BaseLossWrapper

    Examples
    --------
    >>> # Default wrapper
    >>> wrapper = get_loss_wrapper()

    >>> # Using type specification
    >>> wrapper = get_loss_wrapper({"type": "multiscale", "incremental": True})

    >>> # Using Hydra target
    >>> wrapper = get_loss_wrapper({
    ...     "_target_": "anemoi.training.losses.wrappers.MultiScaleWrapper",
    ...     "incremental": False
    ... })
    """
    if config is None:
        LOGGER.debug("No wrapper configuration provided, using IdentityWrapper")
        return IdentityWrapper()

    # Convert to container if it's a DictConfig
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # Try Hydra instantiation first (if _target_ is present)
    if isinstance(config, dict) and "_target_" in config:
        LOGGER.debug("Instantiating wrapper with Hydra from target: %s", config["_target_"])
        wrapper = instantiate(config)
        if not isinstance(wrapper, BaseLossWrapper):
            error_msg = f"Instantiated wrapper must be a BaseLossWrapper, got {type(wrapper)}"
            raise TypeError(error_msg)
        return wrapper

    # Handle dictionary with 'type' specification
    if isinstance(config, dict):
        wrapper_type = config.pop("type", "identity").lower()

        wrapper_map = {
            "identity": IdentityWrapper,
            "none": IdentityWrapper,
            "default": IdentityWrapper,
            "multiscale": MultiScaleWrapper,
            "multi_scale": MultiScaleWrapper,
            "multi-scale": MultiScaleWrapper,
        }

        if wrapper_type not in wrapper_map:
            error_msg = f"Unknown wrapper type: {wrapper_type}. Available: {list(wrapper_map.keys())}"
            raise ValueError(error_msg)

        wrapper_class = wrapper_map[wrapper_type]
        LOGGER.debug("Creating wrapper of type: %s with params: %s", wrapper_type, config)

        # Filter out None values and instantiate
        filtered_config = {k: v for k, v in config.items() if v is not None}
        return wrapper_class(**filtered_config)

    error_msg = f"Invalid wrapper configuration type: {type(config)}"
    raise ValueError(error_msg)


class WrappedLoss(BaseLossWrapper):
    """Convenience class that combines a loss function with a wrapper.

    This allows for a cleaner interface where the loss and wrapper
    are combined into a single callable object.
    """

    def __init__(self, loss_fn, wrapper: BaseLossWrapper | None = None) -> None:
        """Initialize the wrapped loss.

        Parameters
        ----------
        loss_fn : nn.Module
            The underlying loss function
        wrapper : BaseLossWrapper | None, optional
            The wrapper to apply. If None, uses IdentityWrapper.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.wrapper = wrapper or IdentityWrapper()

    def forward(
        self,
        y_pred,
        y,
        **kwargs,
    ):
        """Compute the wrapped loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted tensor
        y : torch.Tensor
            Target tensor
        **kwargs
            Additional arguments passed to the loss function

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            The computed loss value(s)
        """
        # Note: this is a bit different - WrappedLoss itself is a wrapper
        # that contains both the loss and another wrapper
        return self.wrapper.forward(self.loss_fn, y_pred, y, **kwargs)

    def __call__(self, y_pred, y, **kwargs):
        """Make the wrapped loss directly callable."""
        return self.forward(y_pred, y, **kwargs)
