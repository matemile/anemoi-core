# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from math import log

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


class TemporalDistributionLoss(BaseLoss):
    """Temporal loss over sub-steps with optional shift alignment.

    Modes:
    - "ce": Cross-entropy over the time dimension for normalized temporal distributions
      (inputs are probabilities or log-probabilities), with optional shift-tolerant alignment.
    - "rates": Reconstruction loss (MAE/MSE/Huber) between predicted and target substep fields
      (inputs are physical fields, not probabilities). Supports shift-tolerant alignment.

    The loss reduces the time axis internally (summing over time) and returns a per-variable tensor
    with shape (bs, ensemble, grid, n_vars), to which standard scalers and reduction apply.
    """

    name: str = "temporal_distribution_ce"

    def __init__(
        self,
        *,
        ignore_nans: bool = False,
        # Mode: "ce" or "rates"
        mode: str = "ce",
        # Reconstruction loss for rates mode: "mae" | "mse" | "huber"
        reconstruction: str = "mae",
        huber_delta: float = 1.0,
        # Shift-tolerant alignment params
        shift_alignment: dict | None = None,
        # Optionally scale per-sample loss by the sample size (sum over time of target)
        scale_by_sample_size: bool = False,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)

        # Mode and reconstruction configuration
        self.mode: str = str(mode).lower()
        self.reconstruction: str = str(reconstruction).lower()
        self.huber_delta: float = float(huber_delta)
        self.scale_by_sample_size: bool = bool(scale_by_sample_size)

        # Shift alignment configuration
        shift_cfg = shift_alignment or {}
        self.shift_enabled: bool = bool(shift_cfg.get("enabled", False))
        shifts = shift_cfg.get("shifts", [0])
        if isinstance(shifts, int):
            radius = abs(int(shifts))
            shifts = list(range(-radius, radius + 1))
        if 0 not in shifts:
            shifts = sorted(set(list(shifts) + [0]))
        self.shift_values: list[int] = [int(s) for s in shifts]
        self.shift_aggregation: str = str(shift_cfg.get("aggregation", "softmin"))  # 'softmin'|'min'|'mean'
        self.shift_beta: float = float(shift_cfg.get("beta", 5.0))
        self.shift_renormalize: bool = bool(shift_cfg.get("renormalize", True))
        # Coverage-weighted conditional CE over overlap bins (Variant B)
        self.shift_coverage_weighted_conditional: bool = bool(shift_cfg.get("coverage_weighted_conditional", False))
        # Rates normalization strategy over overlap: 'sum' | 'mean' | 'mass_mean'
        self.shift_rates_normalization: str = str(shift_cfg.get("rates_normalization", "sum")).lower()
        # Variables (by output names) to which shift alignment applies
        self.shift_variable_names: list[str] = list(shift_cfg.get("variables", []))

        # Will be populated via set_data_indices (called by get_loss_function)
        self._output_name_to_full_idx: dict[str, int] | None = None
        self._output_idx_to_name: dict[int, str] | None = None

    def set_data_indices(self, data_indices) -> None:  # type: ignore[override]
        try:
            name_to_index = data_indices.model.output.name_to_index
            self._output_name_to_full_idx = dict(name_to_index)
            self._output_idx_to_name = {idx: name for name, idx in name_to_index.items()}
        except Exception:  # pragma: no cover - optional path depending on training pipeline
            self._output_name_to_full_idx = None
            self._output_idx_to_name = None

    def _build_overlap_mask(self, T: int, delta: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask_1d = torch.ones((T,), device=device, dtype=dtype)
        if delta > 0:
            mask_1d[:delta] = 0.0
        elif delta < 0:
            d = -delta
            mask_1d[T - d :] = 0.0
        return mask_1d.view(1, T, 1, 1, 1)

    def _aggregate_shifts(self, stack: torch.Tensor) -> torch.Tensor:
        if self.shift_aggregation == "min":
            return stack.min(dim=0).values
        if self.shift_aggregation == "mean":
            return stack.mean(dim=0)
        beta = max(self.shift_beta, 1e-6)
        return -(1.0 / beta) * (torch.logsumexp(-beta * stack, dim=0) - log(stack.shape[0]))

    def _compute_temporal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        inputs_are_log_probs: bool,
        shift_rel_indices: list[int],
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        pred_probs: torch.Tensor,
        target_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        inputs_are_log_probs: bool = False,
        full_variable_size: int | None = None,
        target_full_indices: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute temporal loss (CE or reconstruction) with optional shift alignment.

        Parameters
        ----------
        pred_probs : torch.Tensor
            For mode=="ce": predicted distributions or log-distributions across time.
            For mode=="rates": predicted sub-step fields in physical units.
            Shape (bs, time, ensemble, grid, n_vars)
        target_probs : torch.Tensor
            For mode=="ce": target distributions across time (sum over time == 1).
            For mode=="rates": target sub-step fields.
            Same shape as pred_probs.
        mask : torch.Tensor | None
            Optional mask with shape (bs, ensemble, grid, n_vars) applied after time reduction.
        """
        # Ensure shape (bs, time, ensemble, grid, vars)
        assert (
            pred_probs.ndim == 5 and target_probs.ndim == 5
        ), f"Expected 5D tensors (bs, time, ensemble, grid, vars); got {pred_probs.shape} and {target_probs.shape}"

        # Map which variables should get shift alignment (by relative indices of incoming V-subset)
        desired_full_idxs: set[int] = set()
        if self._output_name_to_full_idx is not None and len(self.shift_variable_names) > 0:
            for n in self.shift_variable_names:
                if n in self._output_name_to_full_idx:
                    desired_full_idxs.add(self._output_name_to_full_idx[n])

        rel_map: dict[int, int] = {}
        if target_full_indices is not None and len(desired_full_idxs) > 0:
            if isinstance(target_full_indices, torch.Tensor):
                tf_idx = target_full_indices.detach().cpu().tolist()
            else:
                tf_idx = list(target_full_indices)
            rel_map = {full_idx: pos for pos, full_idx in enumerate(tf_idx) if full_idx in desired_full_idxs}
        shift_rel_indices: list[int] = list(rel_map.values())

        loss_per_entity = self._compute_temporal_loss(
            pred_probs,
            target_probs,
            inputs_are_log_probs=inputs_are_log_probs,
            shift_rel_indices=shift_rel_indices,
        )

        if mask is not None:
            loss_per_entity = loss_per_entity * mask

        # Optional magnitude-aware scaling by sample size (sum over time of target)
        if self.scale_by_sample_size:
            sample_size = target_probs.sum(dim=1)
            loss_per_entity = loss_per_entity * sample_size

        if full_variable_size is not None and target_full_indices is not None:
            if isinstance(target_full_indices, torch.Tensor):
                idx = target_full_indices.to(device=loss_per_entity.device)
            else:
                idx = torch.tensor(target_full_indices, device=loss_per_entity.device)
            loss_full = loss_per_entity.new_zeros((*loss_per_entity.shape[:-1], full_variable_size))
            loss_full.index_copy_(-1, idx, loss_per_entity)
            loss_tensor = loss_full
        else:
            loss_tensor = loss_per_entity

        loss_tensor = self.scale(
            loss_tensor,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        is_sharded = grid_shard_slice is not None
        return self.reduce(loss_tensor, squash=squash, group=group if is_sharded else None)


class TemporalDistributionCrossEntropyLoss(TemporalDistributionLoss):
    """Cross-entropy over the time dimension for accumulated variables with optional shift alignment.

    This is a thin wrapper around TemporalDistributionLoss with mode fixed to "ce".
    """

    name: str = "temporal_distribution_ce"

    def __init__(
        self,
        *,
        ignore_nans: bool = False,
        shift_alignment: dict | None = None,
        scale_by_sample_size: bool = False,
    ) -> None:
        super().__init__(
            ignore_nans=ignore_nans,
            mode="ce",
            reconstruction="mae",
            huber_delta=1.0,
            shift_alignment=shift_alignment,
            scale_by_sample_size=scale_by_sample_size,
        )

    def _compute_temporal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        inputs_are_log_probs: bool,
        shift_rel_indices: list[int],
    ) -> torch.Tensor:
        eps = torch.finfo(pred.dtype).eps
        if inputs_are_log_probs:
            log_pred = pred
            prob_pred = log_pred.exp()
        else:
            prob_pred = pred
            log_pred = (prob_pred + eps).log()

        def ce_over_time(p: torch.Tensor, log_p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            term = -y * log_p
            return term.sum(dim=1)

        ce_base = ce_over_time(prob_pred, log_pred, target)
        if not self.shift_enabled or len(shift_rel_indices) == 0:
            return ce_base

        T = prob_pred.shape[1]
        ce_shifts: list[torch.Tensor] = []
        for delta in self.shift_values:
            if delta == 0:
                p_shift = prob_pred
            elif delta > 0:
                p_shift = torch.zeros_like(prob_pred)
                p_shift[:, delta:, ...] = prob_pred[:, : T - delta, ...]
            else:
                d = -delta
                p_shift = torch.zeros_like(prob_pred)
                p_shift[:, : T - d, ...] = prob_pred[:, d:, ...]

            overlap_mask = self._build_overlap_mask(T, delta, prob_pred.device, prob_pred.dtype)
            y_masked = target * overlap_mask
            if self.shift_coverage_weighted_conditional:
                p_masked = p_shift * overlap_mask
                cy_unclamped = y_masked.sum(dim=1, keepdim=True)
                cy = cy_unclamped.clamp_min(eps)
                cp = p_masked.sum(dim=1, keepdim=True).clamp_min(eps)
                y_hat = y_masked / cy
                p_hat = p_masked / cp
                log_p_hat = (p_hat + eps).log()
                ce_cond = ce_over_time(p_hat, log_p_hat, y_hat)
                ce_shifts.append(cy_unclamped.squeeze(1) * ce_cond)
            else:
                # Partial CE over overlap: -sum_{overlap} y_t log p_t
                log_p_full = log_pred
                ce_shift = -(y_masked * log_p_full).sum(dim=1)
                ce_shifts.append(ce_shift)

        ce_agg = self._aggregate_shifts(torch.stack(ce_shifts, dim=0))
        ce_per_entity = ce_base.clone()
        ce_per_entity[..., shift_rel_indices] = ce_agg[..., shift_rel_indices]
        return ce_per_entity


class TemporalDistributionRatesLoss(TemporalDistributionLoss):
    """Reconstruction loss over sub-steps for rates-mode variables with optional shift alignment.

    This is a thin wrapper around TemporalDistributionLoss with mode fixed to "rates".
    """

    name: str = "temporal_distribution_rates"

    def __init__(
        self,
        *,
        ignore_nans: bool = False,
        reconstruction: str = "mae",
        huber_delta: float = 1.0,
        shift_alignment: dict | None = None,
        scale_by_sample_size: bool = False,
    ) -> None:
        super().__init__(
            ignore_nans=ignore_nans,
            mode="rates",
            reconstruction=reconstruction,
            huber_delta=huber_delta,
            shift_alignment=shift_alignment,
            scale_by_sample_size=scale_by_sample_size,
        )

    def _compute_temporal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        inputs_are_log_probs: bool,
        shift_rel_indices: list[int],
    ) -> torch.Tensor:
        eps = torch.finfo(pred.dtype).eps

        def rec_over_time(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if self.reconstruction == "mse":
                err = (p - y) ** 2
            elif self.reconstruction == "huber":
                diff = p - y
                abs_diff = diff.abs()
                quadratic = torch.minimum(
                    abs_diff,
                    torch.tensor(self.huber_delta, device=diff.device, dtype=diff.dtype),
                )
                err = 0.5 * quadratic**2 + self.huber_delta * (abs_diff - quadratic)
            else:
                err = (p - y).abs()
            return err.sum(dim=1)

        # Base (no shift) reconstruction
        rec_base = rec_over_time(pred, target)
        if not self.shift_enabled or len(shift_rel_indices) == 0:
            return rec_base

        T = pred.shape[1]
        rec_shifts: list[torch.Tensor] = []

        for delta in self.shift_values:
            if delta == 0:
                p_shift = pred
            elif delta > 0:
                p_shift = torch.zeros_like(pred)
                p_shift[:, delta:, ...] = pred[:, : T - delta, ...]
            else:
                d = -delta
                p_shift = torch.zeros_like(pred)
                p_shift[:, : T - d, ...] = pred[:, d:, ...]

            # Restrict reconstruction to overlap
            overlap_mask = self._build_overlap_mask(T, delta, pred.device, pred.dtype)
            err = p_shift - target
            if self.reconstruction == "mse":
                err = err**2
            elif self.reconstruction == "huber":
                diff = err
                abs_diff = diff.abs()
                quadratic = torch.minimum(
                    abs_diff,
                    torch.tensor(self.huber_delta, device=diff.device, dtype=diff.dtype),
                )
                err = 0.5 * quadratic**2 + self.huber_delta * (abs_diff - quadratic)
            else:
                err = err.abs()

            err_masked = err * overlap_mask

            if self.shift_rates_normalization == "mean":
                denom_bins = overlap_mask.sum(dim=1, keepdim=False).clamp_min(1.0)
                rec_shift = err_masked.sum(dim=1) / denom_bins
            elif self.shift_rates_normalization == "mass_mean":
                y_masked = target * overlap_mask
                cy = y_masked.sum(dim=1, keepdim=True).clamp_min(eps)
                # mass-weighted mean: sum(y * err) / sum(y)
                rec_shift = (err_masked * (y_masked / cy)).sum(dim=1)
            else:  # 'sum'
                rec_shift = err_masked.sum(dim=1)

            rec_shifts.append(rec_shift)

        rec_agg = self._aggregate_shifts(torch.stack(rec_shifts, dim=0))
        loss_per_entity = rec_base.clone()
        loss_per_entity[..., shift_rel_indices] = rec_agg[..., shift_rel_indices]
        return loss_per_entity
