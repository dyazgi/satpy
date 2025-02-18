#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2021 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Composite classes for the AHI instrument."""

import logging

from satpy.composites import GenericCompositor
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)


class GreenCorrector(GenericCompositor):
    """Corrector of the AHI green band to compensate for the deficit of chlorophyll signal."""

    def __init__(self, *args, fractions=(0.85, 0.15), **kwargs):
        """Set default keyword argument values."""
        # XXX: Should this be 0.93 and 0.07
        self.fractions = fractions
        super(GreenCorrector, self).__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Boost vegetation effect thanks to NIR (0.8µm) band."""
        LOG.info('Boosting vegetation on green band')

        projectables = self.match_data_arrays(projectables)
        new_green = sum(fraction * value for fraction, value in zip(self.fractions, projectables))
        new_green.attrs = combine_metadata(*projectables)
        return super(GreenCorrector, self).__call__((new_green,), **attrs)
