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


class TemporalDistributionCrossEntropyLoss(BaseLoss):
    """Cross-entropy over the time dimension for accumulated variables, with optional ratio-focal weighting
    (gamma) and shift-tolerant temporal alignment.

    - Inputs are distributions over the time axis (sum along time = 1) or their log-probabilities.
    - Optionally applies focal loss modulation with gamma (and alpha) to emphasize sharp events.
    - Optionally performs small-window temporal shift alignment for a subset of variables
      (e.g., tp, cp, sf), aggregating the cross-entropy across shifts via min/softmin/mean.

    The loss reduces the time axis internally and returns a per-variable tensor with
    shape (bs, ensemble, grid, n_vars), to which standard scalers and reduction apply.
    """

    name: str = "temporal_distribution_ce"

    def __init__(
        self,
        *,
        ignore_nans: bool = False,
        # Focal loss params
        focal_gamma: float = 0.0,
        focal_gamma_per_var: dict | None = None,
        # Shift-tolerant alignment params
        shift_alignment: dict | None = None,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)

        # Focal
        self.focal_gamma: float = float(focal_gamma) if focal_gamma is not None else 0.0
        self.focal_gamma_per_var: dict[str, float] = (
            {str(k): float(v) for k, v in focal_gamma_per_var.items()} if focal_gamma_per_var else {}
        )

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
        self.shift_coverage_weighted_conditional: bool = bool(
            shift_cfg.get("coverage_weighted_conditional", False)
        )
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
        """Compute cross-entropy over time and apply scaling and reduction.

        Parameters
        ----------
        pred_probs : torch.Tensor
            Predicted distributions over time, shape (bs, time, ensemble, grid, n_vars)
            or any permutation which will be rearranged internally.
        target_probs : torch.Tensor
            Target distributions over time (sum over time == 1), same shape as pred_probs.
        mask : torch.Tensor | None, optional
            Optional mask with shape (bs, ensemble, grid, n_vars) to zero out contributions
            for positions where the total mass is tiny or undefined.
        squash : bool, optional
            Average last dimension during reduction, by default True
        scaler_indices : tuple[int, ...] | None, optional
            Indices to subset the scaler application, by default None
        without_scalers : list[str] | list[int] | None, optional
            Scalers to exclude, by default None
        grid_shard_slice : slice | None, optional
            Grid shard slice for distributed training, by default None
        group : ProcessGroup | None, optional
            Distributed group to reduce over, by default None
        """

        # Ensure shape (bs, time, ensemble, grid, vars)
        assert pred_probs.ndim == 5 and target_probs.ndim == 5, (
            f"Expected 5D tensors (bs, time, ensemble, grid, vars); got {pred_probs.shape} and {target_probs.shape}"
        )

        # Time dimension is at dim=1; we reduce along it below

        # Numerics
        eps = torch.finfo(pred_probs.dtype).eps

        # Get probabilities and log-probabilities
        if inputs_are_log_probs:
            log_pred = pred_probs
            prob_pred = log_pred.exp()
        else:
            prob_pred = pred_probs
            log_pred = (prob_pred + eps).log()

        # Helper: focal factor
        V = pred_probs.shape[-1]

        # Build per-variable gamma/alpha aligned to incoming variable subset using full indices -> names
        if target_full_indices is not None and self._output_idx_to_name is not None:
            if isinstance(target_full_indices, torch.Tensor):
                tf_idx_list = target_full_indices.detach().cpu().tolist()
            else:
                tf_idx_list = list(target_full_indices)
        else:
            tf_idx_list = [None] * V

        gamma_values: list[float] = []

        for j in range(V):
            full_idx = tf_idx_list[j] if j < len(tf_idx_list) else None
            name = (
                self._output_idx_to_name.get(full_idx)
                if (self._output_idx_to_name is not None and full_idx is not None)
                else None
            )

            g = self.focal_gamma
            if name is not None and name in self.focal_gamma_per_var:
                g = float(self.focal_gamma_per_var[name])
            gamma_values.append(g)

        gamma_v = pred_probs.new_tensor(gamma_values)  # (V,)

        def ce_over_time(
            p: torch.Tensor, log_p: torch.Tensor, y: torch.Tensor, gamma_vec: torch.Tensor
        ) -> torch.Tensor:
            # p, log_p, y shapes: (B, T, E, G, V)
            gamma_b = gamma_vec.view(*(1,) * (p.ndim - 1), -1)  # (1,1,1,1,V)
            # Ratio focal modulation: ((y+eps)/(p+eps))^gamma
            # When gamma == 0, the modulation is 1 (no focal effect)
            ratio = (y + eps) / (p + eps)
            mod = ratio.pow(gamma_b)
            term = -y * mod * log_p
            return term.sum(dim=1)  # (B, E, G, V)

        # Base CE (no shift alignment)
        ce_base = ce_over_time(prob_pred, log_pred, target_probs, gamma_v)

        # If shift alignment disabled or no variables specified, use base CE
        if not self.shift_enabled or (self.shift_variable_names is not None and len(self.shift_variable_names) == 0):
            ce_per_entity = ce_base
        else:
            # Map desired shift variables (by full output index) to relative indices in the incoming V-subset
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

            if len(shift_rel_indices) == 0:
                # None of the requested variables are in this subset; fall back to base
                ce_per_entity = ce_base
            else:
                # Compute CE for each temporal shift
                T = prob_pred.shape[1]
                ce_shifts: list[torch.Tensor] = []
                for delta in self.shift_values:
                    if delta == 0:
                        p_shift = prob_pred
                    elif delta > 0:
                        # shift right by delta (later in time); zero-pad at start
                        p_shift = torch.zeros_like(prob_pred)
                        p_shift[:, delta:, ...] = prob_pred[:, : T - delta, ...]
                    else:  # delta < 0, shift left; zero-pad at end
                        d = -delta
                        p_shift = torch.zeros_like(prob_pred)
                        p_shift[:, : T - d, ...] = prob_pred[:, d:, ...]

                    if self.shift_coverage_weighted_conditional:
                        # Mask for overlap bins only (drop shifted-out edges)
                        mask_1d = p_shift.new_ones((T,))
                        if delta > 0:
                            mask_1d[:delta] = 0.0
                        elif delta < 0:
                            d = -delta
                            mask_1d[T - d :] = 0.0
                        overlap_mask = mask_1d.view(1, T, 1, 1, 1)  # broadcastable

                        y_masked = target_probs * overlap_mask
                        p_masked = p_shift * overlap_mask

                        # Coverage of target on overlap; renormalize both on overlap
                        cy = y_masked.sum(dim=1, keepdim=True).clamp_min(eps)  # (B,1,E,G,V)
                        cp = p_masked.sum(dim=1, keepdim=True).clamp_min(eps)

                        y_hat = y_masked / cy
                        p_hat = p_masked / cp
                        log_p_hat = (p_hat + eps).log()

                        # Conditional CE on overlap with ratio-focal modulation
                        ce_cond = ce_over_time(p_hat, log_p_hat, y_hat, gamma_v)  # (B,E,G,V)
                        # Coverage-weighted conditional CE
                        ce_cov = cy.squeeze(1) * ce_cond  # (B,E,G,V)
                        ce_shifts.append(ce_cov)
                    else:
                        # Original behavior: renormalize across full window after zero-padding
                        if self.shift_renormalize:
                            denom = p_shift.sum(dim=1, keepdim=True).clamp_min(eps)
                            p_shift = p_shift / denom

                        log_p_shift = (p_shift + eps).log()
                        ce_shift = ce_over_time(p_shift, log_p_shift, target_probs, gamma_v)  # (B,E,G,V)
                        ce_shifts.append(ce_shift)

                # Stack over shifts
                ce_stack = torch.stack(ce_shifts, dim=0)  # (S,B,E,G,V)

                # Aggregate across shifts per variable
                if self.shift_aggregation == "min":
                    ce_agg = ce_stack.min(dim=0).values
                elif self.shift_aggregation == "mean":
                    ce_agg = ce_stack.mean(dim=0)
                else:
                    # softmin with temperature beta
                    beta = max(self.shift_beta, 1e-6)
                    # softmin(x) = -1/beta * log( (1/S) * sum exp(-beta * x) )
                    ce_agg = - (1.0 / beta) * (
                        torch.logsumexp(-beta * ce_stack, dim=0) - log(ce_stack.shape[0])
                    )

                # Compose final CE: use aggregated values for selected vars, base CE for the rest
                ce_per_entity = ce_base.clone()
                ce_per_entity[..., shift_rel_indices] = ce_agg[..., shift_rel_indices]

        if mask is not None:
            # Expect mask shape (bs, ensemble, grid, vars)
            ce_per_entity = ce_per_entity * mask

        # Optionally expand to full variable dimension so variable-wise scalers
        # defined over all outputs (e.g., accumulated_variable) can be applied.
        if full_variable_size is not None and target_full_indices is not None:
            if isinstance(target_full_indices, torch.Tensor):
                idx = target_full_indices.to(device=ce_per_entity.device)
            else:
                idx = torch.tensor(target_full_indices, device=ce_per_entity.device)
            ce_full = ce_per_entity.new_zeros((*ce_per_entity.shape[:-1], full_variable_size))
            ce_full.index_copy_(-1, idx, ce_per_entity)
            ce = ce_full
        else:
            ce = ce_per_entity

        # Apply scalers (variable/grid) and reduce as in other BaseLoss losses
        ce = self.scale(
            ce,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        is_sharded = grid_shard_slice is not None
        return self.reduce(ce, squash=squash, group=group if is_sharded else None)


