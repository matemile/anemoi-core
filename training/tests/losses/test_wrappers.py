# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.wrappers import IdentityWrapper
from anemoi.training.losses.wrappers import MultiScaleWrapper
from anemoi.training.losses.wrappers import get_loss_wrapper
from anemoi.training.losses.wrappers.factory import WrappedLoss


class SimpleLoss(BaseLoss):
    """Simple test loss function."""

    def forward(self, y_pred, y, **kwargs):
        return torch.mean((y_pred - y) ** 2)


class TestIdentityWrapper:
    """Tests for IdentityWrapper."""

    def test_identity_passthrough(self):
        """Test that IdentityWrapper passes through correctly."""
        loss_fn = SimpleLoss()
        wrapper = IdentityWrapper()

        y_pred = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        # Direct loss computation
        direct_loss = loss_fn(y_pred, y)

        # Wrapped loss computation
        wrapped_loss = wrapper.forward(loss_fn, y_pred, y)

        torch.testing.assert_close(direct_loss, wrapped_loss)


class TestMultiScaleWrapper:
    """Tests for MultiScaleWrapper."""

    def test_single_scale_no_interpolation(self):
        """Test MultiScaleWrapper with single scale (no interpolation)."""
        loss_fn = SimpleLoss()
        wrapper = MultiScaleWrapper(
            interpolation_matrices=[None],
            incremental=False,
            scale_weights=[1.0],
        )

        y_pred = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        # Direct loss
        direct_loss = loss_fn(y_pred, y)

        # Wrapped loss
        wrapped_loss = wrapper.forward(loss_fn, y_pred, y)

        torch.testing.assert_close(direct_loss, wrapped_loss)

    def test_multi_scale_with_weights(self):
        """Test MultiScaleWrapper with multiple scales and weights."""
        loss_fn = SimpleLoss()
        wrapper = MultiScaleWrapper(
            interpolation_matrices=[None, None],  # Two scales, no actual interpolation
            incremental=False,
            scale_weights=[0.5, 0.5],
        )

        y_pred = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        # Expected: sum of weighted losses
        direct_loss = loss_fn(y_pred, y)
        expected = direct_loss * 0.5 + direct_loss * 0.5

        wrapped_loss = wrapper.forward(loss_fn, y_pred, y)

        torch.testing.assert_close(wrapped_loss, expected)

    def test_incremental_mode(self):
        """Test incremental mode computes differences."""
        loss_fn = SimpleLoss()
        wrapper = MultiScaleWrapper(
            interpolation_matrices=[None, None],
            incremental=True,
            scale_weights=[1.0, 1.0],
        )

        y_pred = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        losses = wrapper.forward(loss_fn, y_pred, y, return_all_scales=True)

        # First scale should be normal loss
        first_scale_loss = loss_fn(y_pred, y)
        torch.testing.assert_close(losses[0], first_scale_loss)

        # Second scale should be loss of zero tensor (y_pred - y_pred = 0)
        zero_loss = loss_fn(torch.zeros_like(y_pred), torch.zeros_like(y))
        torch.testing.assert_close(losses[1], zero_loss)

    def test_custom_interpolation_function(self):
        """Test custom interpolation function."""

        def custom_interp(x, matrix, scale_idx):
            # Simple scaling as test interpolation
            return x * 0.5

        loss_fn = SimpleLoss()
        wrapper = MultiScaleWrapper(
            interpolation_matrices=[torch.eye(4)],  # Dummy matrix
            incremental=False,
            scale_weights=[1.0],
            interpolation_fn=custom_interp,
        )

        y_pred = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        wrapped_loss = wrapper.forward(loss_fn, y_pred, y)

        # Expected: loss of scaled tensors
        expected = loss_fn(y_pred * 0.5, y * 0.5)

        torch.testing.assert_close(wrapped_loss, expected)

    def test_return_all_scales(self):
        """Test returning losses for all scales."""
        loss_fn = SimpleLoss()
        num_scales = 3
        wrapper = MultiScaleWrapper(
            interpolation_matrices=[None] * num_scales,
            incremental=False,
            scale_weights=[1.0] * num_scales,
        )

        y_pred = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        losses = wrapper.forward(loss_fn, y_pred, y, return_all_scales=True)

        assert len(losses) == num_scales
        for loss in losses:
            assert isinstance(loss, torch.Tensor)


class TestWrapperFactory:
    """Tests for the wrapper factory."""

    def test_get_identity_wrapper_default(self):
        """Test default wrapper is IdentityWrapper."""
        wrapper = get_loss_wrapper(None)
        assert isinstance(wrapper, IdentityWrapper)

    def test_get_wrapper_by_type(self):
        """Test getting wrapper by type string."""
        configs = [
            {"type": "identity"},
            {"type": "multiscale"},
            {"type": "multi_scale"},
            {"type": "multi-scale"},
        ]

        expected_types = [
            IdentityWrapper,
            MultiScaleWrapper,
            MultiScaleWrapper,
            MultiScaleWrapper,
        ]

        for config, expected_type in zip(configs, expected_types, strict=False):
            wrapper = get_loss_wrapper(config)
            assert isinstance(wrapper, expected_type)

    def test_get_wrapper_with_params(self):
        """Test getting wrapper with parameters."""
        config = {
            "type": "multiscale",
            "incremental": False,
            "scale_weights": [1.0, 0.5],
        }

        wrapper = get_loss_wrapper(config)
        assert isinstance(wrapper, MultiScaleWrapper)
        assert wrapper.incremental is False
        assert wrapper.scale_weights == [1.0, 0.5]

    def test_invalid_wrapper_type(self):
        """Test error on invalid wrapper type."""
        with pytest.raises(ValueError, match="Unknown wrapper type"):
            get_loss_wrapper({"type": "invalid_wrapper"})


class TestWrappedLoss:
    """Tests for WrappedLoss convenience class."""

    def test_wrapped_loss_callable(self):
        """Test WrappedLoss is callable."""
        loss_fn = SimpleLoss()
        wrapper = IdentityWrapper()
        wrapped_loss = WrappedLoss(loss_fn, wrapper)

        y_pred = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        # Test both forward and __call__
        loss1 = wrapped_loss.forward(y_pred, y)
        loss2 = wrapped_loss(y_pred, y)

        torch.testing.assert_close(loss1, loss2)

    def test_wrapped_loss_with_multiscale(self):
        """Test WrappedLoss with MultiScaleWrapper."""
        loss_fn = SimpleLoss()
        wrapper = MultiScaleWrapper(
            interpolation_matrices=[None, None],
            incremental=False,
            scale_weights=[0.7, 0.3],
        )
        wrapped_loss = WrappedLoss(loss_fn, wrapper)

        y_pred = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        loss = wrapped_loss(y_pred, y)

        # Expected
        direct_loss = loss_fn(y_pred, y)
        expected = direct_loss * 0.7 + direct_loss * 0.3

        torch.testing.assert_close(loss, expected)
