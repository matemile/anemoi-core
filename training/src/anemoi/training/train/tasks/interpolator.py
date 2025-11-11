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
from anemoi.training.losses.base import BaseLoss
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.training.utils.enums import TensorDim

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
        self.time_fraction_size = getattr(config.training.target_forcing, "time_fraction_size", 1)

        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        self.rollout = 1

        # Auxiliary temporal distribution losses (separate CE and rates)
        self.temporal_ce_loss = None
        self.temporal_rates_loss = None

        # Mass conservation enforcement mode: 'train' | 'predict' | None
        self.enforcement = str(getattr(config.training.mass_conservation, "enforcement", None)).lower()

        td_losses = getattr(config.model_dump(by_alias=True).training, "temporal_distribution_losses", None)
        if td_losses is not None:
            # CE loss
            ce_cfg = td_losses.get("cross_entropy", None)
            if ce_cfg is not None:
                self.temporal_ce_loss = get_loss_function(
                    ce_cfg,
                    scalers=self.scalers,
                    data_indices=self.data_indices,
                )

            # Rates loss
            rates_cfg = config.training.temporal_distribution_losses.get("rates", None)
            if rates_cfg is not None:
                self.temporal_rates_loss = get_loss_function(
                    rates_cfg,
                    scalers=self.scalers,
                    data_indices=self.data_indices,
                )

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
        # Cache for downstream validation metrics post-processing context
        self._last_x_boundaries = x_bound

        num_tfi = len(self.target_forcing_indices)
        target_forcing = torch.empty(
            batch.shape[0],
            batch.shape[2],
            batch.shape[3],
            num_tfi if not self.use_time_fraction else num_tfi + self.time_fraction_size,
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
                target_forcing[..., -self.time_fraction_size :] = (
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

        if self.enforcement == "train":
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

            # Optional auxiliary temporal losses (CE and/or rates) over accumulated variables
            accum_target_idxs = self.model.model.map_accum_indices["target_idxs"].tolist()
            accum_constraint_idxs = self.model.model.map_accum_indices["constraint_idxs"].tolist()

            # Predicted temporal diagnostics
            constraints = x_bound[:, -1:, ..., accum_constraint_idxs].detach()
            pred_log_probs = log_weights

            # Target sequence over time (fields) subset to accumulated outputs
            y_true_seq = torch.stack(
                [
                    batch[:, self.imap[interp_step], :, :, self.data_indices.data.output.full]
                    for interp_step in self.interp_times
                ],
                dim=1,
            )
            target_amounts = y_true_seq[..., accum_target_idxs]
            target_totals = target_amounts.sum(dim=1, keepdim=True)
            target_probs = target_amounts / (target_totals.clamp_min(torch.finfo(target_totals.dtype).eps))

            # Validation-only: compute and log temporal entropy for accumulated vars (per variable)
            if validation_mode:
                self.log_entropy_on_accumulated_vars(
                    pred_log_probs=pred_log_probs,
                    pred_probs=weights,
                    target_probs=target_probs,
                    accum_target_idxs=accum_target_idxs,
                    batch_size=batch.shape[0],
                )

            # Training-only: Log the per-variable softmax scale used for temporal distribution (logits mode)
            if not validation_mode and scales is not None:
                use_rates = self.model.model.map_accum_indices["use_rates"].to(device=scales.device, dtype=torch.bool)
                ce_mask = ~use_rates
                if ce_mask.any():
                    scales_ce = scales[ce_mask] if scales.shape[0] == use_rates.shape[0] else scales
                    ce_indices = torch.nonzero(ce_mask, as_tuple=False).squeeze(1).tolist()
                    accum_target_idxs_ce = [accum_target_idxs[i] for i in ce_indices]
                    self.log_softmax_scale_on_accumulated_vars(
                        scales=scales_ce,
                        accum_target_idxs=accum_target_idxs_ce,
                        batch_size=batch.shape[0],
                    )

            # Mask out tiny/undefined totals in either prediction constraint or target total
            minimum_total = 0.0
            zero_constraint = constraints.abs().squeeze(1) > minimum_total  # (bs, ensemble, grid, v_acc)
            valid_mask_all = zero_constraint.to(pred_log_probs.dtype)

            # Split vars by mode
            try:
                use_rates = self.model.model.map_accum_indices["use_rates"].to(
                    device=pred_log_probs.device, dtype=torch.bool,
                )
            except Exception:
                use_rates = torch.zeros((len(accum_target_idxs),), dtype=torch.bool, device=pred_log_probs.device)

            # CE loss on non-rates variables
            if self.temporal_ce_loss is not None and (~use_rates).any():
                ce_mask = ~use_rates
                ce_idx = torch.nonzero(ce_mask, as_tuple=False).squeeze(1).tolist()
                ce_full_indices = [accum_target_idxs[i] for i in ce_idx]
                pred_ce = pred_log_probs[..., ce_mask]
                tgt_ce = target_probs[..., ce_mask]
                mask_ce = valid_mask_all[..., ce_mask]
                ce_value = self.temporal_ce_loss(
                    pred_ce,
                    tgt_ce,
                    mask=mask_ce,
                    grid_shard_slice=self.grid_shard_slice,
                    group=self.model_comm_group,
                    inputs_are_log_probs=True,
                    full_variable_size=len(self.data_indices.model.output.full),
                    target_full_indices=ce_full_indices,
                )
                metrics.update({"temporal_cross_entropy_loss": ce_value})
                loss = loss + ce_value

            # Rates loss on rates variables
            if self.temporal_rates_loss is not None and use_rates.any():
                rates_mask = use_rates
                rates_idx = torch.nonzero(rates_mask, as_tuple=False).squeeze(1).tolist()
                rates_full_indices = [accum_target_idxs[i] for i in rates_idx]
                pred_fields = y_preds[..., accum_target_idxs][..., rates_mask]
                tgt_fields = target_amounts[..., rates_mask]
                mask_rates = valid_mask_all[..., rates_mask]
                rates_value = self.temporal_rates_loss(
                    pred_fields,
                    tgt_fields,
                    mask=mask_rates,
                    grid_shard_slice=self.grid_shard_slice,
                    group=self.model_comm_group,
                    inputs_are_log_probs=False,
                    full_variable_size=len(self.data_indices.model.output.full),
                    target_full_indices=rates_full_indices,
                )
                metrics.update({"temporal_reconstruction_loss": rates_value})
                loss = loss + rates_value

                # Optional additional soft-conservation penalty for rates-soft variables
                try:
                    use_soft = self.model.model.map_accum_indices["use_soft_conservation"].to(dtype=torch.bool)
                    soft_lambda = self.model.model.map_accum_indices["soft_penalty_lambda"].to(dtype=y_preds.dtype)
                    soft_mask = use_rates & use_soft
                    if torch.any(soft_mask):
                        sel = [i for i, m in enumerate(accum_target_idxs) if soft_mask[i]]
                        pred_sum = y_preds[..., sel].sum(dim=1, keepdim=True)
                        cons = constraints[..., sel]
                        lam = soft_lambda[soft_mask].view(*(1,) * (pred_sum.ndim - 1), -1)
                        penalty_tensor = lam * (pred_sum - cons) ** 2
                        soft_penalty = penalty_tensor.mean()
                        metrics.update({"temporal_soft_conservation_penalty": soft_penalty})
                        loss = loss + soft_penalty
                except Exception:
                    pass

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
        if self.enforcement is None:
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
            else:
                rel_mae_all = torch.zeros((), device=diff.device, dtype=diff.dtype)

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
            else:
                diff_mae_low = torch.zeros((), device=diff.device, dtype=diff.dtype)

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
            else:
                rel_mae_low = torch.zeros((), device=diff.device, dtype=diff.dtype)
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
            else:
                diff_mae_high = torch.zeros((), device=diff.device, dtype=diff.dtype)

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
            else:
                rel_mae_high = torch.zeros((), device=diff.device, dtype=diff.dtype)

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
        if self.enforcement == "train":
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

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int = 0,
        grid_shard_slice: slice | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate metrics on the validation output (post-processed, real-space where applicable)

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted ensemble
        y: torch.Tensor
            Ground truth (target).
        rollout_step: int
            Rollout step

        Returns
        -------
        val_metrics : dict[str, torch.Tensor]
            validation metrics and predictions.
        """
        metrics = {}
        y_postprocessed = self.model.post_processors(y, in_place=False)

        # Provide boundaries context if available; processor will no-op if not needed
        if hasattr(self, "_last_x_boundaries") and self._last_x_boundaries is not None:
            y_pred_postprocessed = self.model.post_processors(
                y_pred,
                in_place=False,
                x_boundaries_normalized=self._last_x_boundaries,
            )
        else:
            y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False)

        for metric_name, metric in self.metrics.items():
            if not isinstance(metric, BaseLoss):
                # If not a loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}_metric/{rollout_step + 1}"] = metric(y_pred_postprocessed, y_postprocessed)
                continue

            for mkey, indices in self.val_metric_ranges.items():
                metric_step_name = f"{metric_name}_metric/{mkey}/{rollout_step + 1}"
                if len(metric.scaler.subset_by_dim(TensorDim.VARIABLE.value)):
                    exception_msg = (
                        "Validation metrics cannot be scaled over the variable dimension"
                        " in the post processed space."
                    )
                    raise ValueError(exception_msg)

                metrics[metric_step_name] = metric(
                    y_pred_postprocessed,
                    y_postprocessed,
                    scaler_indices=[..., indices],
                    grid_shard_slice=grid_shard_slice,
                    group=self.model_comm_group,
                )

        return metrics
