# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.losses.wrappers.base import BaseLossWrapper
from anemoi.training.losses.wrappers.factory import get_loss_wrapper
from anemoi.training.losses.wrappers.identity import IdentityWrapper
from anemoi.training.losses.wrappers.multiscale import MultiScaleWrapper

__all__ = ["BaseLossWrapper", "IdentityWrapper", "MultiScaleWrapper", "get_loss_wrapper"]
