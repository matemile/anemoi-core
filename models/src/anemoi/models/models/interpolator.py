# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import einops
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecInterpolator(AnemoiModelEncProcDec):
    """Message passing interpolating graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DotDict
            Job configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        model_config = DotDict(model_config)
        self.num_target_forcings = (
            len(model_config.training.target_forcing.data) + model_config.training.target_forcing.time_fraction * 8
        )
        self.num_input_times = len(model_config.training.explicit_times.input)
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )

        self.latent_skip = model_config.model.latent_skip
        self.grid_skip = model_config.model.grid_skip

        self.setup_mass_conserving_accumulations(data_indices, model_config)

    def _calculate_input_dim(self, model_config):
        return (
            self.num_input_times * self.num_input_channels
            + self.node_attributes.attr_ndims[self._graph_name_data]
            + self.num_target_forcings
        )

    def _assemble_input(self, x, target_forcing, batch_size, grid_shard_shapes=None, model_comm_group=None):
        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=batch_size)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = self._get_shard_shapes(node_attributes_data, 0, grid_shard_shapes, model_comm_group)
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(target_forcing, "batch ensemble grid vars -> (batch ensemble grid) (vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = self._get_shard_shapes(x_data_latent, 0, grid_shard_shapes, model_comm_group)

        if self.grid_skip is not None:
            x_skip = x[:, self.grid_skip, ...]
            if self.A_down is not None or self.A_up is not None:
                x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")
                x_skip = self._apply_truncation(x_skip, grid_shard_shapes, model_comm_group)
                x_skip = einops.rearrange(
                    x_skip, "(batch ensemble) grid vars -> batch ensemble grid vars", batch=batch_size
                )
        else:
            x_skip = None

        return x_data_latent, x_skip, shard_shapes_data

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        if x_skip is not None:
            x_out[..., self._internal_output_idx] += x_skip[..., self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def forward(
        self,
        x: Tensor,
        *,
        target_forcing: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, target_forcing, batch_size, grid_shard_shapes, model_comm_group
        )
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x.dtype)

        return x_out

    def predict_step(
        self,
        batch: torch.Tensor,
        pre_processors: nn.Module,
        post_processors: nn.Module,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> Tensor:
        """Prediction step for the model.

        Base implementation applies pre-processing, performs a forward pass, and applies post-processing.
        Subclasses can override this for different behavior (e.g., sampling for diffusion models).

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data (before pre-processing)
        pre_processors : nn.Module,
            Pre-processing module
        post_processors : nn.Module,
            Post-processing module
        multi_step : int,
            Number of input timesteps
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        gather_out : bool
            Whether to gather output tensors across distributed processes
        **kwargs
            Additional arguments

        Returns
        -------
        Tensor
            Model output (after post-processing)
        """
        with torch.no_grad():

            assert (
                len(batch.shape) == 5
            ), f"The input tensor has an incorrect shape: expected a 5-dimensional tensor, got {batch.shape}!"

            x_boundaries = pre_processors(batch, in_place=False)  # batch should be the input variables only already

            # Handle distributed processing
            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x_boundaries, -2, model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x_boundaries = shard_tensor(x_boundaries, -2, shard_shapes, model_comm_group)

            target_forcing = kwargs.get(
                "target_forcing", None
            )  # shape(bs, interpolation_steps, ens, grid, forcing_dim)
            interpolation_steps = target_forcing.shape[1]

            output_shape = (
                batch.shape[0],
                target_forcing.shape[1],
                batch.shape[2],
                batch.shape[3],
            )
            # Perform forward pass
            # TODO: add the same logic as in _step here e.g. iterative forwards to get the multiple y_hats

            for i in range(interpolation_steps):
                y_pred = self.forward(
                    x_boundaries,
                    model_comm_group=model_comm_group,
                    grid_shard_shapes=grid_shard_shapes,
                    target_forcing=target_forcing[:, i],
                )

                if i == 0:
                    output_shape = output_shape = (
                        batch.shape[0],
                        target_forcing.shape[1],
                        batch.shape[2],
                        batch.shape[3],
                        y_pred.shape[-1],
                    )
                    y_preds = batch.new_zeros(output_shape)

                y_preds[:, i] = y_pred

            include_right_boundary = kwargs.get("include_right_boundary", False)
            raise NotImplementedError("Mass conservation is not implemented for the full rollout version.")
            if self.map_accum_indices is not None:

                y_preds = self.resolve_mass_conservations(
                    y_preds, x_boundaries, include_right_boundary=include_right_boundary
                )
            elif include_right_boundary:
                y_preds = torch.cat([y_preds, x_boundaries[:, -1:, ...]], dim=1)

            # Apply post-processing
            y_preds = post_processors(y_preds, in_place=False)

            # Gather output if needed
            if gather_out and model_comm_group is not None:
                y_preds = gather_tensor(
                    y_preds, -2, apply_shard_shapes(y_preds, -2, grid_shard_shapes), model_comm_group
                )

        return y_preds

    def resolve_mass_conservations(
        self,
        y_preds,
        x_input,
        include_right_boundary: bool = True,
        *,
        return_weights: bool = False,
        return_logits: bool = False,
        return_log_weights: bool = False,
        return_scales: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Enforce a "mass conservation" style constraint on a subset of output variables by
        redistributing a known total (taken from the input constraints) across the time
        dimension using softmax weights derived from the model's logits.

        Args:
            y_preds (torch.Tensor):
                Model outputs with shape (B, T, E, G, V_out) where:
                - B: batch
                - T: time / interpolation steps
                - E, G: extra dims (e.g., ensemble, grid) — passed through unchanged
                - V_out: total number of output variables
                The subset `target_indices` inside V_out are the "accumulated" variables whose
                per-time-step values must sum to a constraint.
            x_input (torch.Tensor):
                Model inputs with shape compatible with y_preds selection. We read the
                *right-boundary* (last time slice) constraint values from:
                    x_input[:, -1:, ..., input_constraint_indxs]
                yielding shape (B, 1, E, G, V_acc).
            include_right_boundary (bool): (this is now whether or not to update the right boundary)
                If False: distribute the constraint over the existing T steps.
                If True:  append an extra (T+1)-th step representing the right boundary and
                        distribute over T+1 steps; also copy non-target outputs at that
                        boundary from inputs.
            return_weights (bool): If True, include softmax weights across time under key "weights".
            return_logits (bool): If True, include raw logits under key "logits".
            return_log_weights (bool): If True, include log-softmax weights across time under key "log_weights".
            return_scales (bool): If True, include the final per-variable softmax scales under key "scales".

        Returns:
            dict[str, torch.Tensor]: A dictionary that always contains key "y_preds" (updated outputs),
            and optionally includes any of "weights", "log_weights", and "logits" depending on the flags.
        """
        update_right_boundary = include_right_boundary

        # Indices mapping:
        # - input_constraint_indxs: channels in x_input containing the total "mass" to conserve
        # - target_indices: corresponding output channels in y_preds that must sum to that mass
        input_constraint_indxs = self.map_accum_indices["constraint_idxs"]
        target_indices = self.map_accum_indices["target_idxs"]

        # Extract logits for the "accumulated" target variables: (B, T, E, G, V_acc)
        logits = y_preds[..., target_indices].clone()
        # Stabilize softmax by removing per-sample time-mean; improves gradient flow
        logits = logits - logits.mean(dim=1, keepdim=True)

        # Apply per-variable softmax scale (inverse temperature) via sigmoid-bounded parameterization
        # scales: (V_acc,) -> broadcast to (1, 1, 1, 1, V_acc)
        if all(k in self.map_accum_indices for k in ("scale_unconstrained", "scale_min", "scale_max")):
            w = self.map_accum_indices["scale_unconstrained"]
            s_min = self.map_accum_indices["scale_min"]
            s_max = self.map_accum_indices["scale_max"]
            scales = s_min + (s_max - s_min) * torch.sigmoid(w)
        else:
            # Fallback to scale 1.0 if not configured
            scales = torch.ones((logits.shape[-1],), device=logits.device, dtype=logits.dtype)
        scaled_logits = logits * scales.view(*(1,) * (logits.ndim - 1), -1)

        # Compute normalized weights along time on scaled logits: (B, T, E, G, V_acc)
        # Note: softmax dim=1 (time). Weights across time sum to 1 per (B, E, G, V_acc).
        weights = F.softmax(scaled_logits, dim=1)

        # The constraint "total mass" comes from the *last* input time slice:
        # shape (B, 1, E, G, V_acc). This broadcasts over time when multiplied by weights.
        # Detach to prevent gradients flowing into x_input/boundaries.
        constraints = x_input[:, -1:, ..., input_constraint_indxs].detach()

        # eps = 1e-6  # gate tiny totals
        # small = constraints.abs() < eps

        if update_right_boundary:
            y_constrained = constraints * weights
            # y_constrained = torch.where(small, 0.0, y_constrained) # When overall amount is 0, we don't need a gradient signal since any dissagregation will be 0 at all times

            y_preds[..., target_indices] = y_constrained
        else:
            y_constrained = weights[:, :-1] * constraints
            # y_constrained = torch.where(small.expand_as(y_constrained), y_preds[:, :-1, ..., target_indices], y_constrained)
            y_preds[:, :-1, ..., target_indices] = y_constrained

        # Compose return dictionary
        result: dict[str, torch.Tensor] = {"y_preds": y_preds}
        if return_weights:
            result["weights"] = weights
        if return_log_weights:
            result["log_weights"] = F.log_softmax(scaled_logits, dim=1)
        if return_logits:
            # Return the centered, scaled logits used to compute the weights
            result["logits"] = scaled_logits
        if return_scales:
            result["scales"] = scales

        return result

    def setup_mass_conserving_accumulations(self, data_indices: dict, config: dict):

        # Mass-conserving accumulations: expose the config mapping on the underlying model and
        # prepare aligned index lists. Each mapping pairs an output variable (prediction target)
        # with an input constraint variable (accumulation/forcing), which we validate and index below.
        self.map_mass_conserving_accums = getattr(config.model, "mass_conserving_accumulations", None)
        if self.map_mass_conserving_accums is None:
            self.map_accum_indices = None
        else:
            target_idx_list: list[int] = []
            constraint_idx_list: list[int] = []
            scale_list: list[float] = []
            scale_min_list: list[float] = []
            scale_max_list: list[float] = []

            # Read per-variable scale configuration (inverse temperature for softmax)
            default_scale = getattr(config.model, "mass_conserving_softmax_default_scale", 1.0)
            per_var_scales = getattr(config.model, "mass_conserving_softmax_scales", {}) or {}

            # Default bounds and per-variable bounds for sigmoid-bounded scales
            default_min = getattr(config.model, "mass_conserving_softmax_default_min_scale", 0.5)
            default_max = getattr(config.model, "mass_conserving_softmax_default_max_scale", 10.0)
            bounds_map = getattr(config.model, "mass_conserving_softmax_scale_bounds", {}) or {}

            for output_varname, input_constraint_varname in self.map_mass_conserving_accums.items():
                assert (
                    input_constraint_varname in data_indices.data._forcing
                ), f"Input constraint variable {input_constraint_varname} not found in data indices forcing variables."
                assert (
                    output_varname in data_indices.model.output.name_to_index
                ), f"Output variable {output_varname} not found in data indices output variables."

                target_idx_list.append(data_indices.model.output.name_to_index[output_varname])
                constraint_idx_list.append(data_indices.model.input.name_to_index[input_constraint_varname])

                # Determine bounds and initial scale for this variable
                var_bounds = bounds_map.get(output_varname, {}) or {}
                var_min = float(var_bounds.get("min", default_min))
                var_max = float(var_bounds.get("max", default_max))
                assert var_max > var_min, f"Invalid bounds for {output_varname}: min={var_min}, max={var_max}"

                init_scale = float(per_var_scales.get(output_varname, default_scale))
                # Ensure init_scale within bounds for a valid logit
                if init_scale <= var_min:
                    init_scale = var_min + 1e-4
                if init_scale >= var_max:
                    init_scale = var_max - 1e-4

                scale_list.append(init_scale)
                scale_min_list.append(var_min)
                scale_max_list.append(var_max)

            # # Compute complementary (non-accumulated) indices for outputs and inputs
            # all_output_indices = list(data_indices.model.output.name_to_index.values())

            # all_input_indices = list(data_indices.model.input.name_to_index.values())

            # prognostic_input_indices = list(data_indices.model.input.prognostic)

            # non_target_idx_list = [idx for idx in all_output_indices if idx not in set(target_idx_list)]
            # non_constraint_idx_list = [idx for idx in all_input_indices if idx not in set(constraint_idx_list)]

            # Compute unconstrained initial values w0 via logit of normalized scale
            scale_min_tensor = torch.tensor(scale_min_list, dtype=torch.float32)
            scale_max_tensor = torch.tensor(scale_max_list, dtype=torch.float32)
            scale_tensor = torch.tensor(scale_list, dtype=torch.float32)
            ratio = (scale_tensor - scale_min_tensor) / (scale_max_tensor - scale_min_tensor)
            ratio = torch.clamp(ratio, 1e-4, 1 - 1e-4)
            w0 = torch.log(ratio) - torch.log(1 - ratio)

            self.map_accum_indices = torch.nn.ParameterDict(
                {
                    "target_idxs": torch.nn.Parameter(
                        torch.tensor(target_idx_list, dtype=torch.long), requires_grad=False
                    ),
                    "constraint_idxs": torch.nn.Parameter(
                        torch.tensor(constraint_idx_list, dtype=torch.long),
                        requires_grad=False,
                    ),
                    # Learnable per-variable sigmoid-bounded scale parameterization
                    "scale_unconstrained": torch.nn.Parameter(w0, requires_grad=True),
                    "scale_min": torch.nn.Parameter(scale_min_tensor, requires_grad=False),
                    "scale_max": torch.nn.Parameter(scale_max_tensor, requires_grad=False),
                },
            )
