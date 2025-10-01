# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from pathlib import Path
from typing import TYPE_CHECKING

import einops
import numpy as np
import torch
from torch import nn

from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.training.losses.wrappers.base import BaseLossWrapper

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)


class MultiScaleWrapper(BaseLossWrapper):
    """Multi-scale loss wrapper that computes losses at different resolutions.

    This wrapper enables computing losses at multiple spatial scales by
    applying interpolation/smoothing operations to both predictions and targets.
    It supports incremental loss computation where coarser scales compute
    the difference from finer scales.
    """

    def __init__(
        self,
        truncation_files: list[str | None] | None = None,
        truncation_path: str | None = None,
        incremental: bool = True,
        scale_weights: list[float] | None = None,
    ) -> None:
        """Initialize the multi-scale wrapper.

        Parameters
        ----------
        truncation_files : list[str | None] | None, optional
            List of NPZ file names for each scale.
            None/False entries indicate no interpolation for that scale.
            If None, defaults to single scale with no interpolation.
        truncation_path : str | None, optional
            Base directory path where NPZ files are stored.
            Used with truncation_files to construct full paths.
        incremental : bool, optional
            If True, compute incremental differences between scales.
            Default is True.
        scale_weights : list[float] | None, optional
            Weights for each scale's loss contribution.
            If None, all scales are weighted equally.
        """
        super().__init__()

        # Load interpolation matrices from NPZ files
        if truncation_files is not None:
            self.interpolation_matrices = self._load_truncation_matrices(truncation_files, truncation_path)
        else:
            # Default: single scale with no interpolation
            self.interpolation_matrices = [None]

        self.incremental = incremental
        self.scale_weights = scale_weights or [1.0] * len(self.interpolation_matrices)

        if len(self.scale_weights) != len(self.interpolation_matrices):
            error_msg = (
                f"Number of scale weights ({len(self.scale_weights)}) must match "
                f"number of interpolation matrices ({len(self.interpolation_matrices)})"
            )
            raise ValueError(error_msg)

    @property
    def num_outputs(self) -> int:
        """Return the number of loss outputs (scales).

        Returns
        -------
        int
            Number of scales at which losses are computed.
        """
        return len(self.interpolation_matrices)

    def _make_truncation_matrix(self, A, data_type=torch.float32):
        A_ = torch.sparse_coo_tensor(
            torch.tensor(np.vstack(A.nonzero()), dtype=torch.long),
            torch.tensor(A.data, dtype=data_type),
            size=A.shape,
        ).coalesce()
        return A_

    def _multiply_sparse(self, x, A):
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type="cuda", enabled=False):
                out = torch.sparse.mm(A, x)
        else:
            with torch.amp.autocast(device_type="cpu", enabled=False):
                out = torch.sparse.mm(A, x)
        return out

    def _truncate_fields(self, x, A, batch_size=None, auto_cast=False):
        if not batch_size:
            batch_size = x.shape[0]
        out = []
        with torch.amp.autocast(device_type="cuda", enabled=auto_cast):
            for i in range(batch_size):
                out.append(self._multiply_sparse(x[i, ...], A))
        return torch.stack(out)

    def _load_truncation_matrices(
        self, truncation_files: list[str | None], truncation_path: str | None = None,
    ) -> list[torch.Tensor | None]:
        """Load truncation matrices from NPZ files.

        Parameters
        ----------
        truncation_files : list[str | None]
            List of NPZ file names for each scale.
            None or False entries mean no interpolation for that scale.
        truncation_path : str | None, optional
            Base directory path for NPZ files.

        Returns
        -------
        list[torch.Tensor | None]
            List of sparse truncation matrices
        """
        matrices = []

        for i, file_name in enumerate(truncation_files):
            if not file_name or file_name == "False" or file_name == "false":
                matrices.append(None)
                LOGGER.info("Scale %d: no truncation matrix", i)
            else:
                # Construct full path
                if truncation_path:
                    full_path = Path(truncation_path) / file_name
                else:
                    full_path = Path(file_name)

                # Only support NPZ files
                if not full_path.suffix == ".npz":
                    error_msg = f"Only NPZ files are supported for truncation matrices, got: {full_path}"
                    raise ValueError(error_msg)

                try:
                    # Load NPZ file and pass directly to _make_truncation_matrix
                    npz_data = np.load(full_path)
                    matrix = self._make_truncation_matrix(npz_data)
                    matrices.append(matrix)
                    LOGGER.info("Scale %d: loaded truncation matrix from %s, shape %s", i, full_path, matrix.shape)
                except Exception as e:
                    LOGGER.error("Failed to load truncation matrix from %s: %s", full_path, e)
                    raise

        if not matrices:
            matrices = [None]
            LOGGER.info("No truncation files provided, using single scale")

        return matrices

    def prepare_for_distributed_truncation(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        model_comm_group: ProcessGroup | None = None,
        grid_dim: int | None = None,
        grid_shard_shapes: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None, tuple | None]:
        """Prepare tensors for distributed interpolation/smoothing.

        This method handles the sharding/unsharding needed for distributed
        multi-scale loss computation.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted tensor (potentially with ensemble dimension)
        y : torch.Tensor
            Target tensor
        model_comm_group : ProcessGroup | None, optional
            Model communication group for distributed operations
        grid_dim : int | None, optional
            Grid dimension index for sharding
        grid_shard_shapes : tuple | None, optional
            Shard shapes for distributed computation

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, tuple | None, tuple | None]
            Prepared y_pred, y, and their shard shapes for later gathering
        """
        # If not using distributed truncation or single scale, return as-is
        if model_comm_group is None or self.num_outputs == 1:
            return y_pred, y, None, None

        # Handle ensemble dimension if present
        # we need to fix this here ... later ignore for now
        if y_pred.ndim == 4 and y_pred.shape[1] > 1:  # Has ensemble dimension
            batch_size, ensemble_size = y_pred.shape[0], y_pred.shape[1]

            # Flatten batch and ensemble dimensions
            y_pred_flat = einops.rearrange(y_pred, "b e g c -> (b e) g c")

            # Apply sharding
            shard_shapes = apply_shard_shapes(y_pred_flat, grid_dim, grid_shard_shapes)
            y_pred_sharded = shard_channels(y_pred_flat, shard_shapes, model_comm_group)

            # Restore ensemble dimension
            y_pred_prepared = einops.rearrange(
                y_pred_sharded,
                "(b e) g c -> b e g c",
                b=batch_size,
                e=ensemble_size,
            )
        else:
            # No ensemble dimension
            shard_shapes = apply_shard_shapes(y_pred, grid_dim, grid_shard_shapes)
            y_pred_prepared = shard_channels(y_pred, shard_shapes, model_comm_group)

        # Prepare target tensor
        shard_shapes_y = apply_shard_shapes(y, grid_dim, grid_shard_shapes)
        y_prepared = shard_channels(y, shard_shapes_y, model_comm_group)

        return y_pred_prepared, y_prepared, shard_shapes, shard_shapes_y

    def forward(
        self,
        loss_fn: nn.Module,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        return_all_scales: bool = False,
        model_comm_group: ProcessGroup | None = None,
        grid_dim: int | None = None,
        grid_shard_shapes: tuple | None = None,
        keep_batch_sharded: bool = False,
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Compute multi-scale loss.

        Parameters
        ----------
        loss_fn : nn.Module
            The underlying loss function to use at each scale
        y_pred : torch.Tensor
            Predicted tensor
        y : torch.Tensor
            Target tensor
        return_all_scales : bool, optional
            If True, return list of losses for each scale.
            If False, return aggregated loss. Default is False.
        model_comm_group : ProcessGroup | None, optional
            Model communication group for distributed operations
        grid_dim : int | None, optional
            Grid dimension index for sharding
        grid_shard_shapes : tuple | None, optional
            Shard shapes for distributed computation
        keep_batch_sharded : bool, optional
            Whether to use distributed truncation preparation
        **kwargs
            Additional keyword arguments passed to the loss function

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            Aggregated loss value or list of losses per scale
        """
        # Handle distributed truncation if needed
        if keep_batch_sharded and self.num_outputs > 1:
            y_pred, y, shard_shapes, shard_shapes_y = self.prepare_for_distributed_truncation(
                y_pred, y, model_comm_group, grid_dim, grid_shard_shapes,
            )
        losses = []
        y_preds_at_scale = []
        ys_at_scale = []

        for i, interp_matrix in enumerate(self.interpolation_matrices):
            LOGGER.debug(
                "Computing loss at scale %d with matrix shape %s",
                i,
                interp_matrix.shape if interp_matrix is not None else None,
            )

            # Interpolate predictions and targets to current scale
            y_pred_scale = self._interpolate(y_pred, interp_matrix, i)
            y_scale = self._interpolate(y, interp_matrix, i)

            # Store for incremental computation
            y_preds_at_scale.append(y_pred_scale)
            ys_at_scale.append(y_scale)

            # Compute incremental difference if requested and not first scale
            if self.incremental and i > 0:
                y_pred_scale = y_pred_scale - y_preds_at_scale[i - 1]
                y_scale = y_scale - ys_at_scale[i - 1]

            # Compute loss at this scale
            scale_loss = loss_fn(y_pred_scale, y_scale, **kwargs)

            # Apply scale weight
            weighted_loss = scale_loss * self.scale_weights[i]
            losses.append(weighted_loss)

        if return_all_scales:
            return losses
        return self.aggregate_losses(losses)

    def _interpolate(
        self,
        x: torch.Tensor,
        interp_matrix: torch.Tensor | None,
        scale_idx: int,
    ) -> torch.Tensor:
        """Apply interpolation to tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to interpolate
        interp_matrix : torch.Tensor | None
            Interpolation matrix or None for no interpolation
        scale_idx : int
            Index of the current scale (for custom interpolation functions)

        Returns
        -------
        torch.Tensor
            Interpolated tensor
        """
        if interp_matrix is None:
            return x

        # Use custom interpolation function if provided (for backward compatibility)
        if self.interpolation_fn is not None:
            return self.interpolation_fn(x, interp_matrix, scale_idx)

        # Default: use built-in interpolation that handles sparse matrices
        return self._interpolate_batch(x, interp_matrix)

    def _interpolate_batch(self, batch: torch.Tensor, interp_matrix: torch.Tensor) -> torch.Tensor:
        """Apply interpolation/truncation to a batch of fields.

        This replaces the need for model._truncate_fields and handles both
        sparse and dense matrices appropriately.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch tensor (e.g., batch x ensemble x spatial x features)
        interp_matrix : torch.Tensor
            Interpolation/truncation matrix (can be sparse or dense)

        Returns
        -------
        torch.Tensor
            Interpolated batch with same shape structure but different spatial dimension
        """
        input_shape = batch.shape
        # Reshape to (-1, spatial, features) for matrix multiplication
        batch_2d = batch.reshape(-1, *input_shape[-2:])

        interp_matrix = interp_matrix.to(batch.device)

        # Check if matrix is sparse
        if hasattr(interp_matrix, "is_sparse") and interp_matrix.is_sparse:
            # Sparse matrix multiplication (efficient for large truncation matrices)
            out = []
            for i in range(batch_2d.shape[0]):
                # Disable autocast for sparse operations (as in original model code)
                if torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        out.append(torch.sparse.mm(interp_matrix, batch_2d[i]))
                else:
                    with torch.amp.autocast(device_type="cpu", enabled=False):
                        out.append(torch.sparse.mm(interp_matrix, batch_2d[i]))
            batch_truncated = torch.stack(out)
        else:
            # Dense matrix multiplication
            # Apply to each element in the batch
            batch_truncated = torch.stack([torch.matmul(interp_matrix, batch_2d[i]) for i in range(batch_2d.shape[0])])

        # Reshape back to original shape structure with new spatial dimension
        new_spatial = batch_truncated.shape[-2]
        output_shape = list(input_shape)
        output_shape[-2] = new_spatial
        return batch_truncated.reshape(*output_shape)

    def set_interpolation_matrices(self, matrices: list[torch.Tensor | None]) -> None:
        """Update the interpolation matrices.

        Parameters
        ----------
        matrices : list[torch.Tensor | None]
            New interpolation matrices
        """
        # Ensure scale weights match the number of matrices
        assert len(self.scale_weights) == len(matrices), (
            f"Number of scale weights ({len(self.scale_weights)}) must match "
            f"number of interpolation matrices ({len(matrices)}). "
            "Please update scale_weights before setting new interpolation matrices."
        )
        self.interpolation_matrices = matrices

    def aggregate_losses(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate multiple scale losses.

        Parameters
        ----------
        losses : list[torch.Tensor]
            List of weighted loss tensors from each scale

        Returns
        -------
        torch.Tensor
            Aggregated loss value
        """
        return torch.stack(losses).sum()


############################################# do not remove yet #####################################################


# def _prepare_for_truncation(
#     self,
#     y_pred_ens: torch.Tensor,
#     y: torch.Tensor,
#     model_comm_group: ProcessGroup,
# ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
#     """Prepare tensors for interpolation/smoothing.

#     Args:
#         y_pred_ens: torch.Tensor
#             Ensemble predictions
#         y: torch.Tensor
#             Ground truth
#         model_comm_group: ProcessGroup
#             Model communication group

#     Returns
#     -------
#         y_pred_ens_interp: torch.Tensor
#             Predictions for interpolation
#         y_interp: torch.Tensor
#             Ground truth for interpolation
#         shard_info: tuple
#             Shard shapes for later gathering
#     """
#     batch_size, ensemble_size = y_pred_ens.shape[0], y_pred_ens.shape[1]

#     y_pred_ens_interp = einops.rearrange(y_pred_ens, "b e g c -> (b e) g c")
#     shard_shapes = apply_shard_shapes(y_pred_ens_interp, self.grid_dim, self.grid_shard_shapes)
#     y_pred_ens_interp = shard_channels(y_pred_ens_interp, shard_shapes, model_comm_group)
#     y_pred_ens_interp = einops.rearrange(
#         y_pred_ens_interp,
#         "(b e) g c -> b e g c",
#         b=batch_size,
#         e=ensemble_size,
#     )

#     shard_shapes_y = apply_shard_shapes(y, self.grid_dim, self.grid_shard_shapes)
#     y_interp = shard_channels(y, shard_shapes_y, model_comm_group)

#     return y_pred_ens_interp, y_interp, shard_shapes, shard_shapes_y

# def gather_and_compute_loss(
#     self,
#     y_pred: torch.Tensor,
#     y: torch.Tensor,
#     loss: torch.nn.Module,
#     ens_comm_subgroup_size: int,
#     ens_comm_subgroup: ProcessGroup,
#     model_comm_group: ProcessGroup,
#     return_pred_ens: bool = False,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
#     """Gather the ensemble members from all devices in my group.

#     Eliminate duplicates (if any) and compute the loss.

#     Args:
#         y_pred: torch.Tensor
#             Predicted state tensor, calculated on self.device
#         y: torch.Tensor
#             Ground truth
#         loss: torch.nn.Module
#             Loss function
#         ens_comm_group_size: int
#             Size of the ensemble communication group
#         ens_comm_subgroup: ProcessGroup
#             Ensemble communication subgroup
#         model_comm_group: ProcessGroup
#             Model communication group
#         return_pred_ens: bool
#             Validation flag: if True, we return the predicted ensemble (post-gather)

#     Returns
#     -------
#         loss_inc:
#             Loss
#         y_pred_ens:
#             Predictions if validation mode
#     """
#     # gather ensemble members
#     y_pred_ens = gather_tensor(
#         y_pred.clone(),  # for bwd because we checkpoint this region
#         dim=1,
#         shapes=[y_pred.shape] * ens_comm_subgroup_size,
#         mgroup=ens_comm_subgroup,
#     )

#     is_multi_scale_loss = any(x is not None for x in self.loss_trunc_matrices)
#     shard_shapes, shard_shapes_y = None, None
#     if self.keep_batch_sharded and is_multi_scale_loss:
#         # go to full sequence dimension for interpolation / smoothing
#         y_pred_ens_interp, y_for_interp, shard_shapes, shard_shapes_y = self._prepare_for_truncation(
#             y_pred_ens,
#             y,
#             model_comm_group,
#         )
#     else:
#         y_pred_ens_interp = y_pred_ens
#         y_for_interp = y

#     loss_inc = []
#     y_preds_ens = []
#     ys = []
#     for i, trunc_matrix in enumerate(self.loss_trunc_matrices):
#         LOGGER.debug(
#             "Loss: %s %s %s",
#             i,
#             trunc_matrix.shape if trunc_matrix is not None else None,
#             trunc_matrix.device if trunc_matrix is not None else None,
#         )

#         # interpolate / smooth the predictions and the truth for loss computation
#         y_pred_ens_tmp, y_tmp = self._interp_for_loss(y_pred_ens_interp, y_for_interp, i)

#         if self.keep_batch_sharded and is_multi_scale_loss:
#             y_pred_ens_tmp = gather_channels(y_pred_ens_tmp, shard_shapes, model_comm_group)
#             y_tmp = gather_channels(y_tmp, shard_shapes_y, model_comm_group)

#         # save for next loss scale
#         y_preds_ens.append(y_pred_ens_tmp)
#         ys.append(y_tmp)

#         if i > 0:  # assumption, resol 0 < 1 < 2 < ... < n
#             y_pred_ens_tmp = y_pred_ens_tmp - y_preds_ens[i - 1]
#             y_tmp = y_tmp - ys[i - 1]

#         # compute the loss
#         loss_inc.append(
#             loss(
#                 y_pred_ens_tmp,
#                 y_tmp,
#                 squash=True,
#                 grid_shard_slice=self.grid_shard_slice,
#                 group=model_comm_group,
#             ),
#         )

#     return loss_inc, y_pred_ens if return_pred_ens else None

# def _interp_for_loss(self, x: torch.Tensor, y: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
#     if self.loss_trunc_matrices[i] is not None:
#         self.loss_trunc_matrices[i] = self.loss_trunc_matrices[i].to(x.device)
#         x = self._interpolate_batch(x, self.loss_trunc_matrices[i])
#         y = self._interpolate_batch(y, self.loss_trunc_matrices[i])
#     return x, y

# def _interpolate_batch(self, batch: torch.Tensor, intp_matrix: torch.Tensor) -> torch.Tensor:
#     input_shape = batch.shape  # e.g. (batch steps ensemble grid vars) or (batch steps grid vars)
#     batch = batch.reshape(-1, *input_shape[-2:])
#     batch = self.model.model._truncate_fields(batch, intp_matrix)  # to coarse resolution
#     return batch.reshape(*input_shape)
