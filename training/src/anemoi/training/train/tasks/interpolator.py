# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping
from operator import itemgetter

import torch
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import get_loss_function
from anemoi.training.train.tasks.base import BaseGraphModule

LOGGER = logging.getLogger(__name__)


class GraphInterpolator(BaseGraphModule):
    """Graph neural network interpolator for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            truncation_data=truncation_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )
        if len(config.training.target_forcing.data) >= 1:
            self.target_forcing_indices = itemgetter(*config.training.target_forcing.data)(
                data_indices.data.input.name_to_index,
            )
            if isinstance(self.target_forcing_indices, int):
                self.target_forcing_indices = [self.target_forcing_indices]
        else:
            self.target_forcing_indices = []

        self.use_time_fraction = config.training.target_forcing.time_fraction

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.rollout = 1

        # Auxiliary temporal distribution CE loss (configurable via training.temporal_distribution_loss)
        td_cfg = getattr(config.model_dump(by_alias=True).training, "temporal_distribution_loss", None)
        if td_cfg is not None:
            self.temporal_dist_loss = get_loss_function(
                config.training.temporal_distribution_loss,
                scalers=self.scalers,
                data_indices=self.data_indices,
            )
            self.temporal_dist_loss_weight = getattr(config.training, "temporal_distribution_loss_weight", 0.0)

    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], list[torch.Tensor]]:

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        x_bound = batch[:, itemgetter(*self.boundary_times)(self.imap)][
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, time, ens, latlon, nvar)

        num_tfi = len(self.target_forcing_indices)
        target_forcing = torch.empty(
            batch.shape[0],
            batch.shape[2],
            batch.shape[3],
            num_tfi if not self.use_time_fraction else num_tfi + 8,
            device=self.device,
            dtype=batch.dtype,
        )

        target_shape = (
            batch.shape[0],
            len(self.interp_times),
            batch.shape[2],
            batch.shape[3],
            len(self.data_indices.model.output.name_to_index),
        )
        y_preds = batch.new_zeros(target_shape)

        for idx, interp_step in enumerate(self.interp_times):
            # get the forcing information for the target interpolation time:
            if num_tfi >= 1:
                target_forcing[..., :num_tfi] = batch[:, self.imap[interp_step], :, :, self.target_forcing_indices]
            if self.use_time_fraction:
                target_forcing[..., -8:] = (
                    2 * (interp_step - self.boundary_times[-2]) / (self.boundary_times[-1] - self.boundary_times[-2])
                ) - 1

            y_pred = self(x_bound, target_forcing)
            y_preds[:, idx] = y_pred

        # If the last interpolation time is a boundary time, copy boundary values to the last step
        # for the NON-accumulated variables (complement of the accumulated mapping)
        # NOTE: Currently this method does not support the existence of any diagnostic variables - because they would have to also be predicted at the last step - but here we assume that only prognostic variables need to have their predicted values replaced with the actual boundary values
        if (
            self.interp_times[-1] in self.boundary_times
            and getattr(self.model.model, "map_accum_indices", None) is not None
        ):
            # non_accum_out = self.model.model.map_accum_indices["non_target_idxs"].tolist()
            # non_accum_in = self.model.model.map_accum_indices["non_constraint_idxs"].tolist()
            # y_preds[:, -1, ..., non_accum_out] = x_bound[:, -1, ..., non_accum_in]

            y_preds[:, -1, ..., self.data_indices.model.output.prognostic] = x_bound[
                :,
                -1,
                ...,
                self.data_indices.model.input.prognostic,
            ]

        if self.model.model.map_accum_indices is not None:
            # Enforce mass conservation and retrieve temporal distribution in log-space to avoid recomputation
            mass_conservation_result = self.model.model.resolve_mass_conservations(
                y_preds,
                x_bound,
                return_weights=True,
                return_log_weights=True,
                return_scales=True,
            )
            y_preds = mass_conservation_result["y_preds"]
            log_weights = mass_conservation_result["log_weights"]
            weights = mass_conservation_result["weights"]
            scales = mass_conservation_result["scales"]

            # # Training-only: check and log temporal alignment of accumulations with boundary constraints
            if not validation_mode:
                self.check_accum_alignment(y_preds, x_bound)

            # Optional auxiliary temporal distribution CE loss over accumulated variables
            if self.temporal_dist_loss_weight and self.temporal_dist_loss_weight > 0.0:
                accum_target_idxs = self.model.model.map_accum_indices["target_idxs"].tolist()
                accum_constraint_idxs = self.model.model.map_accum_indices["constraint_idxs"].tolist()

                # Predicted distribution over time in log-space (directly from resolve_mass_conservations)
                constraints = x_bound[:, -1:, ..., accum_constraint_idxs].detach()
                pred_log_probs = log_weights

                # Target distribution over time from ground-truth sequence
                # Collect targets for all interpolation times and subset to output vars
                y_true_seq = torch.stack(
                    [
                        batch[:, self.imap[interp_step], :, :, self.data_indices.data.output.full]
                        for interp_step in self.interp_times
                    ],
                    dim=1,
                )
                target_amounts = y_true_seq[..., accum_target_idxs]
                target_totals = target_amounts.sum(dim=1, keepdim=True)
                target_probs = target_amounts / (target_totals)

                # Validation-only: compute and log temporal entropy for accumulated vars (per variable)
                if validation_mode:
                    self.log_entropy_on_accumulated_vars(
                        pred_log_probs=pred_log_probs,
                        pred_probs=weights,
                        target_probs=target_probs,
                        accum_target_idxs=accum_target_idxs,
                        batch_size=batch.shape[0],
                    )

                # Training-only: Log the per-variable softmax scale used for temporal distribution
                if not validation_mode:
                    self.log_softmax_scale_on_accumulated_vars(
                        scales=scales,
                        accum_target_idxs=accum_target_idxs,
                        batch_size=batch.shape[0],
                    )

                # Mask out tiny/undefined totals in either prediction constraint or target total
                minimum_total = 0.0
                zero_constraint = constraints.abs().squeeze(1) > minimum_total  # (bs, ensemble, grid, v_acc)
                # tiny_tgt = target_totals.abs().squeeze(1) < eps
                # valid_mask = (~(tiny_pred | tiny_tgt)).to(pred_probs.dtype)

                valid_mask = zero_constraint.to(pred_log_probs.dtype)

                # Compute CE over time; scalers for CE are configured in the loss (grid-level only)
                ce_loss = self.temporal_dist_loss(
                    pred_log_probs,
                    target_probs,
                    mask=valid_mask,
                    grid_shard_slice=self.grid_shard_slice,
                    group=self.model_comm_group,
                    inputs_are_log_probs=True,
                    full_variable_size=len(self.data_indices.model.output.full),
                    target_full_indices=accum_target_idxs,
                )
                metrics.update({"temporal_cross_entropy_loss": ce_loss})

                loss = loss + self.temporal_dist_loss_weight * ce_loss

        # During Training and Validation we don't need to compute the loss for the sixth step for non accumulated variables so we skip it
        _inter_step_losses = (
            self.interp_times[:-1] if self.interp_times[-1] in self.boundary_times else self.interp_times
        )

        for idx, interp_step in enumerate(_inter_step_losses):
            y = batch[:, self.imap[interp_step], :, :, self.data_indices.data.output.full].clone()
            y_pred = y_preds[:, idx]

            loss_step, metrics_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                interp_step,
                training_mode=True,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            loss += loss_step / len(self.interp_times)
            metrics.update(metrics_next)

        y_preds_list = list(y_preds.unbind(dim=1))
        return loss, metrics, y_preds_list

    def check_accum_alignment(self, y_preds: torch.Tensor, x_bound: torch.Tensor) -> bool:
        """Check and log temporal alignment between per-hour predictions and 6-hour boundary totals.

        Sums predictions over the time dimension for accumulated output variables and compares
        them to the corresponding boundary constraint values at the last input time.
        Logs a per-variable mean signed difference for monitoring during training.
        Returns a boolean indicating whether sums are close to constraints within tolerances.
        """
        if getattr(self.model.model, "map_accum_indices", None) is None:
            return True

        accum_target_idxs = self.model.model.map_accum_indices["target_idxs"].tolist()
        accum_constraint_idxs = self.model.model.map_accum_indices["constraint_idxs"].tolist()

        # Build reverse map: output index -> varname
        name_to_index = self.data_indices.model.output.name_to_index
        index_to_name = {idx: name for name, idx in name_to_index.items()}

        # Sum over interpolation time dimension (dim=1) for accumulated outputs
        # Shapes:
        #   pred_sum:     (B, E, G, V_acc)
        #   constraints:  (B, E, G, V_acc)
        pred_sum = y_preds[..., accum_target_idxs].sum(dim=1)
        constraints = x_bound[:, -1, ..., accum_constraint_idxs]
        diff = pred_sum - constraints
        # Minimum constraint threshold for bucketing
        min_constraint_value = 0.0

        # Log per-variable mean absolute differences (overall and segmented) on-step (training only)
        for j, out_idx in enumerate(accum_target_idxs):
            varname = index_to_name.get(out_idx, f"var_{out_idx}")

            # Overall absolute difference
            diff_mae_all = diff[..., j].abs().mean()
            self.log(
                f"accum_align_mae_diff_{varname}",
                diff_mae_all,
                on_epoch=False,
                on_step=True,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=x_bound.shape[0],
                sync_dist=True,
            )

            # Overall relative absolute difference (exclude zero-denominator cases)
            denom = constraints[..., j]
            nonzero_mask = denom != 0
            if torch.any(nonzero_mask):
                rel_mae_all = (diff[..., j][nonzero_mask] / denom[nonzero_mask]).abs().mean()

                self.log(
                    f"accum_align_mae_rel_diff_{varname}",
                    rel_mae_all,
                    on_epoch=False,
                    on_step=True,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=x_bound.shape[0],
                    sync_dist=True,
                )

            # Segmented by constraint threshold
            low_mask = denom <= min_constraint_value
            high_mask = denom > min_constraint_value

            # Low bucket: absolute difference
            if torch.any(low_mask):
                diff_mae_low = diff[..., j][low_mask].abs().mean()

                self.log(
                    f"accum_align_mae_diff_low_{varname}",
                    diff_mae_low,
                    on_epoch=False,
                    on_step=True,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=x_bound.shape[0],
                    sync_dist=True,
                )

            # Low bucket: relative absolute difference (exclude zero denominators)
            low_nonzero_mask = low_mask & nonzero_mask
            if torch.any(low_nonzero_mask):
                rel_mae_low = (diff[..., j][low_nonzero_mask] / denom[low_nonzero_mask]).abs().mean()
                self.log(
                    f"accum_align_mae_rel_diff_low_{varname}",
                    rel_mae_low,
                    on_epoch=False,
                    on_step=True,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=x_bound.shape[0],
                    sync_dist=True,
                )

            # High bucket: absolute difference
            if torch.any(high_mask):
                diff_mae_high = diff[..., j][high_mask].abs().mean()

                self.log(
                    f"accum_align_mae_diff_high_{varname}",
                    diff_mae_high,
                    on_epoch=False,
                    on_step=True,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=x_bound.shape[0],
                    sync_dist=True,
                )

            # High bucket: relative absolute difference (exclude zero denominators)
            high_nonzero_mask = high_mask & nonzero_mask
            if torch.any(high_nonzero_mask):
                rel_mae_high = (diff[..., j][high_nonzero_mask] / denom[high_nonzero_mask]).abs().mean()

                self.log(
                    f"accum_align_mae_rel_diff_high_{varname}",
                    rel_mae_high,
                    on_epoch=False,
                    on_step=True,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=x_bound.shape[0],
                    sync_dist=True,
                )

        # Return a simple closeness check as a health indicator
        # return torch.allclose(pred_sum, constraints, rtol=1e-4, atol=1e-5)

    def log_entropy_on_accumulated_vars(
        self,
        *,
        pred_log_probs: torch.Tensor,
        pred_probs: torch.Tensor,
        target_probs: torch.Tensor,
        accum_target_idxs: list[int],
        batch_size: int,
    ) -> None:
        """Log predicted and target temporal entropies for accumulated variables (validation only)."""
        T = pred_log_probs.shape[1]
        # Guard against log(0)
        target_log_probs = torch.log(target_probs + 1e-6)

        # Build reverse map: output index -> varname
        name_to_index = self.data_indices.model.output.name_to_index
        index_to_name = {idx: name for name, idx in name_to_index.items()}

        for j, out_idx in enumerate(accum_target_idxs):
            varname = index_to_name.get(out_idx, f"var_{out_idx}")

            # Predicted entropy: sum_t p_t log p_t / T, average over batch/ens/grid
            p_logp_pred = pred_probs[..., j] * pred_log_probs[..., j]
            pred_entropy_mean = (p_logp_pred.sum(dim=1) / T).mean()

            # Target entropy: sum_t p_t log p_t / T, average over batch/ens/grid
            p_logp_true = target_probs[..., j] * target_log_probs[..., j]
            target_entropy_mean = (p_logp_true.sum(dim=1) / T).mean()

            # Log scalars via Lightning logger
            self.log(
                f"val_entropy_{varname}_pred",
                pred_entropy_mean,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                f"val_entropy_{varname}_true",
                target_entropy_mean,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch_size,
                sync_dist=True,
            )

    def log_softmax_scale_on_accumulated_vars(
        self,
        *,
        scales: torch.Tensor,
        accum_target_idxs: list[int],
        batch_size: int,
    ) -> None:
        """Log the per-variable softmax scale used for temporal distribution (training only)."""
        # Build reverse map: output index -> varname
        name_to_index = self.data_indices.model.output.name_to_index
        index_to_name = {idx: name for name, idx in name_to_index.items()}

        for j, out_idx in enumerate(accum_target_idxs):
            varname = index_to_name.get(out_idx, f"var_{out_idx}")
            self.log(
                f"logit_scale_{varname}",
                scales[j],
                on_epoch=False,
                on_step=True,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        grid_shard_slice: slice | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute base loss excluding accumulated variables (handled by CE loss).

        Uses scaler_indices to restrict the training loss to non-accumulated variables.
        Falls back to the default implementation if no accumulation mapping is set.
        """
        if getattr(self.model.model, "map_accum_indices", None) is not None:
            # non_accum_idxs = self.model.model.map_accum_indices["non_target_idxs"].tolist()
            prognostic_idxs = self.data_indices.model.output.prognostic.tolist()
            return self.loss(
                y_pred,
                y,
                scaler_indices=[..., prognostic_idxs],
                grid_shard_slice=grid_shard_slice,
                group=self.model_comm_group,
            )

        return super()._compute_loss(y_pred=y_pred, y=y, grid_shard_slice=grid_shard_slice, **_kwargs)

    def forward(self, x: torch.Tensor, target_forcing: torch.Tensor) -> torch.Tensor:
        return self.model(
            x,
            target_forcing=target_forcing,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
        )
