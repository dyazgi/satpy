#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2022 Satpy developers
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
"""Eumetsat MHS L1B NetCDF format reader.

Sample:
   - W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,NOAA19+MHS_C_EUMP_20200624080901_58637_eps_o_l1.nc

"""

import functools
import logging
import os
from datetime import datetime

import dask.array as da
import numpy as np
import xarray as xr
from pyproj import CRS
from pyresample.geometry import AreaDefinition

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import unzip_file

logger = logging.getLogger(__name__)


class NcMHS1B(BaseFileHandler):
    """MHS from Eumetsat NetCDF reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(NcMHS1B, self).__init__(filename, filename_info,
                                      filetype_info)

        self._unzipped = unzip_file(self.filename)
        if self._unzipped:
            self.filename = self._unzipped

        self.cache = {}

        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  decode_times = False,
                                  chunks=CHUNK_SIZE)


        # self.nc = self.nc.rename({'nx': 'x', 'ny': 'y'})
        # Daniel Y: Eumetsat has a problem in attribute values for lat
        #        int lat(along_track, across_track) ;
        #        lat:long_name = "latitude" ;  This should be grid_latitude
        #        lat:standard_name = "grid_latitude" ; This should be latitude

        if 'lat' in self.nc.keys():
            lat_attrs = self.nc.lat.attrs
            if 'standard_name' in self.nc.lat.attrs.keys():
                if lat_attrs['standard_name'] == 'grid_latitude':
                    self.nc.lat.attrs['standard_name'] = 'latitude'
                    self.nc.lat.attrs['long_name'] = 'grid_latitude'

        if 'record_start_time' in self.nc.keys():
            record_start_time_attrs = self.nc.record_start_time.attrs
            if 'units' in record_start_time_attrs.keys():
                if record_start_time_attrs['units'] == 's since 01-01-2000 0:0:0':
                    self.nc.record_start_time.attrs['original_units'] = 's since 01-01-2000 0:0:0'
                    self.nc.record_stop_time.attrs['orignial_units'] = 's since 01-01-2000 0:0:0'
                    self.nc.record_start_time.attrs['new_units'] = 'seconds'
                    self.nc.record_stop_time.attrs['new_units'] = 'seconds'
                    self.nc.record_start_time.attrs['unix_add'] = 946684800  # seconds between 01-01-1970 UTC and  01-01-2000 0:0:0 UTC
                    self.nc.record_stop_time.attrs['unix_add'] = 946684800
                    self.nc.record_start_time.attrs['units'] = 'removed'




            attrs = record_start_time_attrs.copy()

            across_track = self.nc.dims['across_track']
            along_track = self.nc.dims['along_track']
            attrs['valid_max'] = along_track * across_track
            var = self.nc.record_start_time.compute().data

            #var = np.array(range(0,along_track*across_track))
            #var.shape = (1,along_track)
            var = np.repeat(var, across_track,axis = 0)

            var.shape = (along_track,across_track)

            #print(var)
            var = da.from_array(var,chunks=self.nc.lon.chunks)
            data = xr.DataArray(
                data = var,
                name = 'record_start_time',
                dims = ["along_track","across_track"],
                coords = dict(
                lon = (["along_track","across_track"], self.nc.lon.data),
                lat = (["along_track","across_track"], self.nc.lat.data)),
                attrs = dict(**attrs))
            self.nc['record_start_time'] = data




        self.nc = xr.decode_cf(self.nc,
                                   decode_times=True,
                                   mask_and_scale=False)

        self.nc = self.nc.rename({'along_track': 'y', 'across_track': 'x'})
        self.platform_long_name = self.nc.attrs['platform_long_name']
        self.platform = self.nc.attrs['platform']
        self.sensor = self.nc.attrs['sensor']


    def __del__(self):
        """Delete the instance."""
        if self._unzipped:
            try:
                os.remove(self._unzipped)
            except OSError:
                pass

    def get_dataset(self, dsid, info):
        """Load a dataset."""
        dsid_name = dsid['name']
        if dsid_name in self.cache:
            logger.debug('Get the data set from cache: %s.', dsid_name)
            return self.cache[dsid_name]

        logger.debug('Reading %s.', dsid_name)
        file_key = self._get_filekeys(dsid_name, info)
        variable = self.nc[file_key]
        # variable = self.remove_timedim(variable)
        variable = self.scale_dataset(variable, info)

        return variable

    def _get_varname_in_file(self, info, info_type="file_key"):
        if isinstance(info[info_type], list):
            for key in info[info_type]:
                if key in self.nc:
                    return key
        return info[info_type]

    def _get_filekeys(self, dsid_name, info):
        try:
            file_key = self._get_varname_in_file(info, info_type="file_key")
        except KeyError:
            file_key = dsid_name
        return file_key

    def scale_dataset(self, variable, info):
        """Scale the data set, applying the attributes from the netCDF file.

        The scale and offset attributes will then be removed from the resulting variable.
        """
        # variable = remove_empties(variable)

        scale = variable.attrs.get('scale_factor', np.array(1))
        offset = variable.attrs.get('add_offset', np.array(0))
        if np.issubdtype((scale + offset).dtype, np.floating) or np.issubdtype(variable.dtype, np.floating):
            variable = self._mask_variable(variable)

        # Keep Order of the two lines below
        attrs = variable.attrs.copy()


        if scale != 1 and offset != 0:
            variable = variable * scale + offset
        elif scale != 1:
            variable = variable * scale
        elif offset != 0:
            variable = variable + offset


        variable.attrs = attrs
        if 'valid_range' in variable.attrs:
            variable.attrs['valid_range'] = variable.attrs['valid_range'] * scale + offset

        variable.attrs.pop('add_offset', None)
        variable.attrs.pop('scale_factor', None)

        # variable.attrs.update({'platform_name': self.platform_name,
        #                       'sensor': self.sensor})

        # if 'palette_meanings' in variable.attrs:
        #     variable = self._prepare_variable_for_palette(variable, info)
        #
        # if 'standard_name' in info:
        #     variable.attrs.setdefault('standard_name', info['standard_name'])
        # variable = self._adjust_variable_for_legacy_software(variable)

        return variable

    @staticmethod
    def _mask_variable(variable):
        if '_FillValue' in variable.attrs:
            variable = variable.where(
                variable != variable.attrs['_FillValue'])
            variable.attrs['_FillValue'] = np.nan
        if 'valid_range' in variable.attrs:
            variable = variable.where(
                variable <= variable.attrs['valid_range'][1])
            variable = variable.where(
                variable >= variable.attrs['valid_range'][0])
        if 'valid_max' in variable.attrs:
            variable = variable.where(
                variable <= variable.attrs['valid_max'])
        if 'valid_min' in variable.attrs:
            variable = variable.where(
                variable >= variable.attrs['valid_min'])
        return variable

    @property
    def start_time(self):
        """Return the start time of the object."""
        try:
            return datetime.strptime(self.nc.attrs['start_sensing_data_time'], '%Y%m%d%H%M%SZ')
        except ValueError:
            return None

    @property
    def end_time(self):
        """Return the end time of the object."""
        try:
            return datetime.strptime(self.nc.attrs['end_sensing_data_time'], '%Y%m%d%H%M%SZ')
            #print(np.min(self.nc['record_start_time']))
            #return datetime.strptime(np.max(self.nc['record_end_time']),'%Y%m%dT%H%M%S%fZ')
        except ValueError:
            return None