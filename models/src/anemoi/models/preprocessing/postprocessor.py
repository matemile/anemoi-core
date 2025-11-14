# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod
from typing import Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.activations import CustomRelu
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing.normalizer import InputNormalizer

LOGGER = logging.getLogger(__name__)


class Postprocessor(BasePreprocessor):
    """Class for Basic Postprocessors.

    For Postprocessors just the inverse_transform method is implemented.
    transform is not needed and corresponds to the identity function.
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the Postprocessor.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        """
        super().__init__(config, data_indices, statistics)

        self._prepare_postprocessing_indices_list()
        self._create_postprocessing_indices()

        self._validate_indices()

    def _validate_indices(self):
        assert (
            len(self.index_training_output) == len(self.index_inference_output) <= len(self.postprocessorfunctions)
        ), (
            f"Error creating postprocessing indices {len(self.index_training_output)}, "
            f"{len(self.index_inference_output)}, {len(self.postprocessorfunctions)}"
        )

    def _prepare_postprocessing_indices_list(self):
        """Prepare the postprocessor indices list."""
        self.num_training_output_vars = len(self.data_indices.data.output.name_to_index)
        self.num_inference_output_vars = len(self.data_indices.model.output.name_to_index)

        (
            self.index_training_output,
            self.index_inference_output,
            self.postprocessorfunctions,
        ) = ([], [], [])

    def _create_postprocessing_indices(self):
        """Create the indices for postprocessing."""

        # Create indices for postprocessing
        for name in self.data_indices.data.output.name_to_index:

            method = self.methods.get(name, self.default)
            if method == "none":
                LOGGER.debug(f"Postprocessor: skipping {name} as no postprocessing method is specified")
                continue
            assert name in self.data_indices.model.output.name_to_index, (
                f"Postprocessor: {name} not found in inference output indices. "
                f"Postprocessors cannot be applied to forcing variables."
            )

            self.index_training_output.append(self._get_index(self.data_indices.data.output.name_to_index, name))
            self.index_inference_output.append(self._get_index(self.data_indices.model.output.name_to_index, name))
            self.postprocessorfunctions.append(self._get_postprocessor_function(method, name))

    def _get_index(self, name_to_index_dict, name):
        return name_to_index_dict.get(name, None)

    def _get_postprocessor_function(self, method, name):
        if method == "relu":
            postprocessor_function = torch.nn.functional.relu
        elif method == "hardtanh":
            postprocessor_function = torch.nn.Hardtanh(min_val=-1, max_val=1)  # default hardtanh
        elif method == "hardtanh_0_1":
            postprocessor_function = torch.nn.Hardtanh(min_val=0, max_val=1)
        else:
            raise ValueError(f"Unknown postprocessing method: {method}")

        LOGGER.info(f"Postprocessor: applying {method} to {name}")
        return postprocessor_function

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        """Postprocess model output tensor."""
        if not in_place:
            x = x.clone()

        if x.shape[-1] == self.num_training_output_vars:
            index = self.index_training_output
        elif x.shape[-1] == self.num_inference_output_vars:
            index = self.index_inference_output
        else:
            raise ValueError(
                f"Input tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
            )

        # Replace values
        for postprocessor, idx_dst in zip(self.postprocessorfunctions, index):
            if idx_dst is not None:
                x[..., idx_dst] = postprocessor(x[..., idx_dst])
        return x


class NormalizedReluPostprocessor(Postprocessor):
    """Postprocess with a ReLU activation and customizable thresholds.

    Expects the config to have keys corresponding to customizable thresholds and lists of variables to postprocess and a normalizer to apply to thresholds.:
    ```
    normalizer: 'mean-std'
    1:
        - y
    0:
        - x
    3.14:
        - q
    ```
    Thresholds are in un-normalized space. If normalizer is specified, the threshold values are normalized.
    This is necessary if in config file the normalizer is specified before the postprocessor, e.g.:
    ```
    data:
        processors:
          normalizer:
            _target_: anemoi.models.preprocessing.normalizer.InputNormalizer
            config:
              default: "mean-std"
          normalized_relu_postprocessor:
            _target_: anemoi.models.preprocessing.postprocessor.NormalizedReluPostprocessor
            config:
              271.15:
              - x1
              0:
              - x2
              normalizer: 'mean-std'
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:

        self.statistics = statistics

        super().__init__(config, data_indices, statistics)

        # Validate normalizer input
        if self.normalizer not in {"none", "mean-std", "min-max", "max", "std"}:
            raise ValueError(
                "Normalizer must be one of: 'none', 'mean-std', 'min-max', 'max', 'std' in NormalizedReluBounding."
            )

    def _get_postprocessor_function(self, method: float, name: str) -> CustomRelu:
        """Get the relu function class for the specified threshold and name."""
        stat_index = self.data_indices.data.input.name_to_index[name]
        normalized_value = method
        if self.normalizer == "mean-std":
            mean = self.statistics["mean"][stat_index]
            std = self.statistics["stdev"][stat_index]
            normalized_value = (method - mean) / std
        elif self.normalizer == "min-max":
            min_stat = self.statistics["minimum"][stat_index]
            max_stat = self.statistics["maximum"][stat_index]
            normalized_value = (method - min_stat) / (max_stat - min_stat)
        elif self.normalizer == "max":
            max_stat = self.statistics["maximum"][stat_index]
            normalized_value = method / max_stat
        elif self.normalizer == "std":
            std = self.statistics["stdev"][stat_index]
            normalized_value = method / std
        postprocessor_function = CustomRelu(normalized_value)

        LOGGER.info(
            f"NormalizedReluPostprocessor: applying NormalizedRelu with threshold {normalized_value} after {self.normalizer} normalization to {name}."
        )
        return postprocessor_function


class ConditionalPostprocessor(Postprocessor):
    """Base class for postprocessors that conditionally apply a transformation based on another variable.

    This class is intended to be subclassed for specific implementations.
    It expects the config to have keys corresponding to customizable values and lists of variables to postprocess.
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

    def _prepare_postprocessing_indices_list(self):
        """Prepare the postprocessor indices list."""

        super()._prepare_postprocessing_indices_list()

        # retrieve index of masking variable
        self.masking_variable = self.remap
        self.masking_variable_training_output = self.data_indices.data.output.name_to_index.get(
            self.masking_variable, None
        )
        self.masking_variable_inference_output = self.data_indices.model.output.name_to_index.get(
            self.masking_variable, None
        )

    def fill_with_value(self, x: torch.Tensor, index: list[int], fill_mask: torch.tensor):
        for idx_dst, value in zip(index, self.postprocessorfunctions):
            if idx_dst is not None:
                x[..., idx_dst][fill_mask] = value
        return x

    @abstractmethod
    def get_locations(self, x: torch.Tensor) -> torch.Tensor:
        """Get a mask from data for conditional postprocessing.
        This method must be implemented by subclasses.

        Parameters:
            x (torch.Tensor): The output for reference variable.

        Returns:
            torch.Tensor: A mask tensor indicating the locations for postprocessing of shape x.shape.
        """
        pass

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        """Set values in the output tensor."""
        if not in_place:
            x = x.clone()

        # Replace with value if masking variable is zero
        if x.shape[-1] == self.num_training_output_vars:
            index = self.index_training_output
            masking_variable = self.masking_variable_training_output
        elif x.shape[-1] == self.num_inference_output_vars:
            index = self.index_inference_output
            masking_variable = self.masking_variable_inference_output
        else:
            raise ValueError(
                f"Output tensor ({x.shape[-1]}) does not match the training "
                f"({self.num_training_output_vars}) or inference shape ({self.num_inference_output_vars})",
            )

        postprocessor_mask = self.get_locations(x[..., masking_variable])

        # Replace values
        return self.fill_with_value(x, index, postprocessor_mask)


class ConditionalZeroPostprocessor(ConditionalPostprocessor):
    """Sets values to specified value where another variable is zero.

    Expects the config to have keys corresponding to customizable values and
    lists of variables to postprocess and a masking/reference variable to use for postprocessing.:

    ```
    default: "none"
    remap: "x"
    0:
        - y
    5.0:
        - x
    3.14:
        - q
    ```

    If "x" is zero, "y" will be postprocessed with 0, "x" with 5.0 and "q" with 3.14.
    """

    def _get_postprocessor_function(self, method: float, name: str):
        """For ConditionalZeroPostprocessor, the 'method' is the constant value to fill
        when the masking variable is zero. This function simply returns the value.
        """
        LOGGER.info(
            f"ConditionalZeroPostprocessor: replacing valus in {name} with value {method} if {self.masking_variable} is zero."
        )
        return method

    def get_locations(self, x: torch.Tensor) -> torch.Tensor:
        """Get zero mask from data"""
        # reference/masking variable is already selected. Mask covers all remaining dimensions.
        return x == 0


class ConditionalNaNPostprocessor(ConditionalPostprocessor):
    """Sets values to NaNs where another variable is NaN.

    Expects the config to have list of variables to postprocess and a
    masking/reference variable to use for postprocessing.:

    ```
    default: "none"
    remap: "x"
    nan:
        - y
    ```

    The module sets "y" NaN, at NaN locations of "x".
    """

    def _get_postprocessor_function(self, method: float, name: str):
        """For ConditionalNaNPostprocessor, the 'method' is a NaN to fill
        when the masking variable is NaN. This function simply returns a NaN.
        """
        LOGGER.info(
            f"ConditionalNaNPostprocessor: replacing values in {name} with value NaN if {self.masking_variable} is NaN."
        )
        return torch.nan

    def get_locations(self, x: torch.Tensor) -> torch.Tensor:
        """Get NaN mask from data"""
        # reference/masking variable is already selected. Mask covers all remaining dimensions.
        return torch.isnan(x)


class TemporalRescaleByTotals(BasePreprocessor):
    """Rescale accumulated output time series in REAL space to match right-boundary totals.

    This is an inverse-only post-processor. It expects the caller to pass the real-space
    boundary tensor via keyword argument `x_boundaries_real` or `x_boundaries_normalized` and operates only on tensors
    whose last dimension matches the model output size.

    Config expected fields (all optional with defaults):
      - enabled: bool (default: True)
      - enforcement: str (default: "predict"); only active if equals "predict"
      - mapping: dict[target_var -> constraint_var]; typically `${training.mass_conservation.accumulations}`
      - minimum_abs_total: float (default 0.0)
      - minimum_abs_pred_sum: float (default 1e-6)
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        # Keep a direct handle to statistics for optional de-normalization when receiving normalized inputs
        self.statistics = statistics
        self.config = config
        super().__init__({}, data_indices, statistics)

        # Initialize a matching InputNormalizer to reuse inverse_transform for selective de-normalization
        self._input_normalizer: InputNormalizer | None = None
        try:
            cfg = self.data_indices.config
            normalizer_cfg = None
            if isinstance(cfg, dict):
                if "normalizer" in cfg:
                    normalizer_cfg = cfg["normalizer"]
                elif "data" in cfg and isinstance(cfg["data"], dict) and "normalizer" in cfg["data"]:
                    normalizer_cfg = cfg["data"]["normalizer"]
            if normalizer_cfg is not None:
                self._input_normalizer = InputNormalizer(
                    config=normalizer_cfg,
                    data_indices=self.data_indices,
                    statistics=self.statistics,
                )
        except Exception as e:
            LOGGER.warning(f"TemporalRescaleByTotals: could not initialize InputNormalizer for denorm: {e}")
            raise e

        # Flags and thresholds
        self.enabled = bool(getattr(config, "enabled", True)) if config is not None else True
        self.enforcement = str(getattr(config, "enforcement", "predict")).lower() if config is not None else "predict"
        self.minimum_abs_total = float(getattr(config, "minimum_abs_total", 0.0)) if config is not None else 0.0
        self.minimum_abs_pred_sum = (
            float(getattr(config, "minimum_abs_pred_sum", 1.0e-6)) if config is not None else 1.0e-6
        )

        # Build aligned index lists from mapping dict[target -> constraint]
        mapping = getattr(config, "mapping", {}) if config is not None else {}
        if isinstance(mapping, dict):
            targets = list(mapping.keys())
            constraints = [mapping[t] for t in targets]
        else:
            targets, constraints = [], []

        self._target_indices = []
        self._constraint_indices = []
        self._target_names: list[str] = []
        self._constraint_names: list[str] = []

        name_to_output = self.data_indices.model.output.name_to_index
        name_to_input = self.data_indices.model.input.name_to_index

        for t_name, c_name in zip(targets, constraints):
            if t_name in name_to_output and c_name in name_to_input:
                self._target_indices.append(name_to_output[t_name])
                self._constraint_indices.append(name_to_input[c_name])
                self._target_names.append(t_name)
                self._constraint_names.append(c_name)
            else:
                LOGGER.warning(
                    f"TemporalRescaleByTotals: skipping mapping {t_name}->{c_name} (indices not found in model indices)."
                )

        # Cache sizes for quick guards
        self._num_output_vars = len(name_to_output)
        self._num_input_vars = len(name_to_input)

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        if not in_place:
            x = x.clone()

        # Quick guards
        if not self.enabled:
            return x
        if self.enforcement != "predict":
            return x
        if x.ndim < 2 or x.shape[-1] != self._num_output_vars:
            # Only operate on tensors shaped like model outputs
            return x

        # Determine boundaries context: prefer real if provided, else try normalized and denormalize
        x_boundaries_real: Optional[torch.Tensor] = kwargs.get("x_boundaries_real", None)
        x_boundaries_normalized: Optional[torch.Tensor] = kwargs.get("x_boundaries_normalized", None)

        if x_boundaries_real is None and x_boundaries_normalized is not None:
            # Denormalize normalized boundaries in-place to obtain real-space totals for constraints
            # Only denormalize the constraint channels we need, using data (training) input statistics
            # Shapes: x_boundaries_normalized[..., C], C = full model input vars; select constraint_idx later
            # We reconstruct real-space slice for constraints below directly
            pass
        elif x_boundaries_real is None and x_boundaries_normalized is None:
            # No context provided -> no-op
            return x

        if not self._target_indices:
            # Nothing to rescale
            return x

        device = x.device
        dtype = x.dtype

        target_idx = torch.tensor(self._target_indices, device=device, dtype=torch.long)
        constraint_idx = torch.tensor(self._constraint_indices, device=device, dtype=torch.long)

        if x_boundaries_real is None:
            # We have normalized boundaries; denormalize constraints slice to real space
            constraints_norm = x_boundaries_normalized[:, -1:, ..., constraint_idx]
            # Prefer using the InputNormalizer for selective inverse_transform (std-only per config)

            train_name_to_input = self.data_indices.data.input.name_to_index
            train_constraint_indices = torch.tensor(
                [train_name_to_input[name] for name in self._constraint_names], device=device, dtype=torch.long
            )
            constraints = self._input_normalizer.inverse_transform(
                constraints_norm, in_place=False, data_index=train_constraint_indices
            )

        else:
            # Use provided real-space boundaries directly
            constraints = x_boundaries_real[:, -1:, ..., constraint_idx]
        # Predicted sums over time (REAL space, since this runs after denorm): (B, 1, E, G, V_acc)
        pred_sum = x[..., target_idx].sum(dim=1, keepdim=True)

        # Compute scaling factors with numerical guards (abs-based as configured elsewhere)
        alpha = constraints.abs() / pred_sum.abs().clamp_min(self.minimum_abs_pred_sum)
        if self.minimum_abs_total > 0.0:
            skip_mask = constraints.abs() <= self.minimum_abs_total
            alpha = torch.where(skip_mask, torch.ones_like(alpha, dtype=dtype, device=device), alpha)

        # Apply broadcast multiply across time
        x_scaled = x[..., target_idx] * alpha.to(dtype=dtype, device=device)
        x[..., target_idx] = x_scaled

        return x
