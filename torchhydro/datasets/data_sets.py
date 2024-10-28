"""
Author: Wenyu Ouyang
Date: 2024-04-08 18:16:53
LastEditTime: 2024-08-10 15:10:27
LastEditors: Wenyu Ouyang
Description: A pytorch dataset class; references to https://github.com/neuralhydrology/neuralhydrology
FilePath: \torchhydro\torchhydro\datasets\data_sets.py
Copyright (c) 2024-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import math
import os.path
import re
import sys
from datetime import datetime
from datetime import timedelta
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
import xarray as xr
from black.mode import Deprecated
from dateutil.parser import parse
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from hydrodatasource.utils.utils import streamflow_unit_conv
from torch.utils.data import Dataset
from tqdm import tqdm

from torchhydro.configs.config import DATE_FORMATS
from torchhydro.datasets.data_scalers import ScalerHub
from torchhydro.datasets.data_sources import data_sources_dict
from torchhydro.datasets.data_utils import (
    warn_if_nan,
    wrap_t_s_dict,
)

LOGGER = logging.getLogger(__name__)


def _fill_gaps_da(da: xr.DataArray, fill_nan: Optional[str] = None) -> xr.DataArray:
    """Fill gaps in a DataArray"""
    if fill_nan is None or da is None:
        return da
    assert isinstance(da, xr.DataArray), "Expect da to be DataArray (not dataset)"
    # fill gaps
    if fill_nan == "et_ssm_ignore":
        all_non_nan_idx = []
        for i in range(da.shape[0]):
            non_nan_idx_tmp = np.where(~np.isnan(da[i].values))
            all_non_nan_idx = all_non_nan_idx + non_nan_idx_tmp[0].tolist()
        # some NaN data appear in different dates in different basins
        non_nan_idx = np.unique(all_non_nan_idx).tolist()
        for i in range(da.shape[0]):
            targ_i = da[i][non_nan_idx]
            da[i][non_nan_idx] = targ_i.interpolate_na(
                dim="time", fill_value="extrapolate"
            )
    elif fill_nan == "mean":
        # fill with mean
        for var in da["variable"].values:
            var_data = da.sel(variable=var)  # select the data for the current variable
            mean_val = var_data.mean(
                dim="basin"
            )  # calculate the mean across all basins
            if warn_if_nan(mean_val):
                # when all value are NaN, mean_val will be NaN, we set mean_val to -1
                mean_val = -1
            filled_data = var_data.fillna(
                mean_val
            )  # fill NaN values with the calculated mean
            da.loc[dict(variable=var)] = (
                filled_data  # update the original dataarray with the filled data
            )
    elif fill_nan == "interpolate":
        # fill interpolation
        for i in range(da.shape[0]):
            da[i] = da[i].interpolate_na(dim="time", fill_value="extrapolate")
    else:
        raise NotImplementedError(f"fill_nan {fill_nan} not implemented")
    return da


def detect_date_format(date_str):
    for date_format in DATE_FORMATS:
        try:
            datetime.strptime(date_str, date_format)
            return date_format
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date_str}")


class BaseDataset(Dataset):
    """Base data set class to load and preprocess data (batch-first) using PyTorch's Dataset"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        data_cfgs
            parameters for reading source data
        is_tra_val_te
            train, vaild or test
        """
        super(BaseDataset, self).__init__()
        self.data_cfgs = data_cfgs
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        # load and preprocess data
        self._load_data()

    @property
    def data_source(self):
        source_name = self.data_cfgs["source_cfgs"]["source_name"]
        source_path = self.data_cfgs["source_cfgs"]["source_path"]
        other_settings = self.data_cfgs["source_cfgs"].get("other_settings", {})
        return data_sources_dict[source_name](source_path, **other_settings)

    @property
    def streamflow_name(self):
        return self.data_cfgs["target_cols"][0]

    @property
    def precipitation_name(self):
        return self.data_cfgs["relevant_cols"][0]

    @property
    def ngrid(self):
        """How many basins/grids in the dataset

        Returns
        -------
        int
            number of basins/grids
        """
        return len(self.basins)

    @property
    def nt(self):
        """length of longest time series in all basins

        Returns
        -------
        int
            number of longest time steps
        """
        if isinstance(self.t_s_dict["t_final_range"][0], tuple):
            trange_type_num = len(self.t_s_dict["t_final_range"])
            if trange_type_num not in [self.ngrid, 1]:
                raise ValueError(
                    "The number of time ranges should be equal to the number of basins "
                    "if you choose different time ranges for different basins"
                )
            earliest_date = None
            latest_date = None
            for start_date_str, end_date_str in self.t_s_dict["t_final_range"]:
                date_format = detect_date_format(start_date_str)

                start_date = datetime.strptime(start_date_str, date_format)
                end_date = datetime.strptime(end_date_str, date_format)

                if earliest_date is None or start_date < earliest_date:
                    earliest_date = start_date
                if latest_date is None or end_date > latest_date:
                    latest_date = end_date
            earliest_date = earliest_date.strftime(date_format)
            latest_date = latest_date.strftime(date_format)
        else:
            # trange_type_num = 1
            earliest_date = self.t_s_dict["t_final_range"][0]
            latest_date = self.t_s_dict["t_final_range"][1]
        min_time_unit = self.data_cfgs["min_time_unit"]
        min_time_interval = self.data_cfgs["min_time_interval"]
        time_step = f"{min_time_interval}{min_time_unit}"
        s_date = pd.to_datetime(earliest_date)
        e_date = pd.to_datetime(latest_date)
        time_series = pd.date_range(start=s_date, end=e_date, freq=time_step)
        return len(time_series)

    @property
    def basins(self):
        """Return the basins of the dataset"""
        return self.t_s_dict["sites_id"]

    @property
    def times(self):
        """Return the times of all basins

        TODO: Although we support get different time ranges for different basins,
        we didn't implement the reading function for this case in _read_xyc method.
        Hence, it's better to choose unified time range for all basins
        """
        min_time_unit = self.data_cfgs["min_time_unit"]
        min_time_interval = self.data_cfgs["min_time_interval"]
        time_step = f"{min_time_interval}{min_time_unit}"
        if isinstance(self.t_s_dict["t_final_range"][0], tuple):
            times_ = []
            trange_type_num = len(self.t_s_dict["t_final_range"])
            if trange_type_num not in [self.ngrid, 1]:
                raise ValueError(
                    "The number of time ranges should be equal to the number of basins "
                    "if you choose different time ranges for different basins"
                )
            detect_date_format(self.t_s_dict["t_final_range"][0][0])
            for start_date_str, end_date_str in self.t_s_dict["t_final_range"]:
                s_date = pd.to_datetime(start_date_str)
                e_date = pd.to_datetime(end_date_str)
                time_series = pd.date_range(start=s_date, end=e_date, freq=time_step)
                times_.append(time_series)
        else:
            detect_date_format(self.t_s_dict["t_final_range"][0])
            trange_type_num = 1
            s_date = pd.to_datetime(self.t_s_dict["t_final_range"][0])
            e_date = pd.to_datetime(self.t_s_dict["t_final_range"][1])
            times_ = pd.date_range(start=s_date, end=e_date, freq=time_step)
        return times_

    def __len__(self):
        return self.num_samples if self.train_mode else self.ngrid

    def __getitem__(self, item: int):
        if not self.train_mode:
            x = self.x[item, :, :]
            y = self.y[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c[item, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length: idx + self.rho + self.horizon, :]
        y = self.y[basin, idx: idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def _pre_load_data(self):
        self.train_mode = self.is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = self.data_cfgs["forecast_history"]
        self.warmup_length = self.data_cfgs["warmup_length"]
        self.horizon = self.data_cfgs["forecast_length"]

    def _load_data(self):
        self._pre_load_data()
        self._read_xyc()
        # normalization
        norm_x, norm_y, norm_c = self._normalize()
        self.x, self.y, self.c = self._kill_nan(norm_x, norm_y, norm_c)
        self._trans2nparr()
        self._create_lookup_table()

    def _trans2nparr(self):
        """To make __getitem__ more efficient,
        we transform x, y, c to numpy array with shape (nsample, nt, nvar)
        """
        self.x = self.x.transpose("basin", "time", "variable").to_numpy()
        self.y = self.y.transpose("basin", "time", "variable").to_numpy()
        if self.c is not None and self.c.shape[-1] > 0:
            self.c = self.c.transpose("basin", "variable").to_numpy()
            self.c_origin = self.c_origin.transpose("basin", "variable").to_numpy()
        self.x_origin = self.x_origin.transpose("basin", "time", "variable").to_numpy()
        self.y_origin = self.y_origin.transpose("basin", "time", "variable").to_numpy()

    def _normalize(self):
        scaler_hub = ScalerHub(
            self.y_origin,
            self.x_origin,
            self.c_origin,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.x, scaler_hub.y, scaler_hub.c

    def _to_dataarray_with_unit(self, data_forcing_ds, data_output_ds, data_attr_ds):
        # trans to dataarray to better use xbatch
        if data_output_ds is not None:
            data_output = self._trans2da_and_setunits(data_output_ds)
        else:
            data_output = None
        if data_forcing_ds is not None:
            data_forcing = self._trans2da_and_setunits(data_forcing_ds)
        else:
            data_forcing = None
        if data_attr_ds is not None:
            # firstly, we should transform some str type data to float type
            data_attr = self._trans2da_and_setunits(data_attr_ds)
        else:
            data_attr = None
        return data_forcing, data_output, data_attr

    def _check_ts_xrds_unit(self, data_forcing_ds, data_output_ds):
        """Check timeseries xarray dataset unit and convert if necessary

        Parameters
        ----------
        data_forcing_ds : _type_
            _description_
        data_output_ds : _type_
            _description_
        """

        def standardize_unit(unit):
            unit = unit.lower()  # convert to lower case
            unit = re.sub(r"day", "d", unit)
            unit = re.sub(r"hour", "h", unit)
            return unit

        streamflow_unit = data_output_ds[self.streamflow_name].attrs["units"]
        prcp_unit = data_forcing_ds[self.precipitation_name].attrs["units"]

        standardized_streamflow_unit = standardize_unit(streamflow_unit)
        standardized_prcp_unit = standardize_unit(prcp_unit)
        if standardized_streamflow_unit != standardized_prcp_unit:
            data_output_ds = streamflow_unit_conv(
                data_output_ds,
                self.data_source.read_area(self.t_s_dict["sites_id"]),
                target_unit=prcp_unit,
            )
        return data_forcing_ds, data_output_ds

    def _read_xyc(self):
        """Read x, y, c data from data source

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset, xr.Dataset]
            x, y, c data
        """
        # x
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            self.t_s_dict["t_final_range"],
            self.data_cfgs["target_cols"],
        )
        if isinstance(data_output_ds_, dict) or isinstance(data_forcing_ds_, dict):
            # this means the data source return a dict with key as time_unit
            # in this BaseDataset, we only support unified time range for all basins, so we chose the first key
            # TODO: maybe this could be refactored better
            data_forcing_ds_ = data_forcing_ds_[list(data_forcing_ds_.keys())[0]]
            data_output_ds_ = data_output_ds_[list(data_output_ds_.keys())[0]]
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )
        # c
        data_attr_ds = self.data_source.read_attr_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
            all_number=True,
        )
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            data_forcing_ds, data_output_ds, data_attr_ds
        )

    def _trans2da_and_setunits(self, ds):
        """Set units for dataarray transfromed from dataset"""
        result = ds.to_array(dim="variable")
        units_dict = {
            var: ds[var].attrs["units"]
            for var in ds.variables
            if "units" in ds[var].attrs
        }
        result.attrs["units"] = units_dict
        return result

    def _kill_nan(self, x, y, c):
        data_cfgs = self.data_cfgs
        y_rm_nan = data_cfgs["target_rm_nan"]
        x_rm_nan = data_cfgs["relevant_rm_nan"]
        c_rm_nan = data_cfgs["constant_rm_nan"]
        if x_rm_nan:
            # As input, we cannot have NaN values
            _fill_gaps_da(x, fill_nan="interpolate")
            warn_if_nan(x)
        if y_rm_nan:
            _fill_gaps_da(y, fill_nan="interpolate")
            warn_if_nan(y)
        if c_rm_nan:
            _fill_gaps_da(c, fill_nan="mean")
            warn_if_nan(c)
        warn_if_nan(x, nan_mode="all")
        warn_if_nan(y, nan_mode="all")
        warn_if_nan(c, nan_mode="all")
        return x, y, c

    def _create_lookup_table(self):
        lookup = []
        # list to collect basins ids of basins without a single training sample
        basin_coordinates = len(self.t_s_dict["sites_id"])
        rho = self.rho
        warmup_length = self.warmup_length
        horizon = self.horizon
        max_time_length = self.nt
        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            if self.is_tra_val_te != "train":
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                )
            else:
                # some dataloader load data with warmup period, so leave some periods for it
                # [warmup_len] -> time_start -> [rho] -> [horizon]
                nan_array = np.isnan(self.y[basin, :, :])
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                    if not np.all(nan_array[f + rho: f + rho + horizon])
                )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)


class BasinSingleFlowDataset(BaseDataset):
    """one time length output for each grid in a batch"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(BasinSingleFlowDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, index):
        xc, ys = super(BasinSingleFlowDataset, self).__getitem__(index)
        y = ys[-1, :]
        return xc, y

    def __len__(self):
        return self.num_samples


class DplDataset(BaseDataset):
    """pytorch dataset for Differential parameter learning"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        data_cfgs
            configs for reading source data
        is_tra_val_te
            train, vaild or test
        """
        super(DplDataset, self).__init__(data_cfgs, is_tra_val_te)
        # we don't use y_un_norm as its name because in the main function we will use "y"
        # For physical hydrological models, we need warmup, hence the target values should exclude data in warmup period
        self.warmup_length = data_cfgs["warmup_length"]
        self.target_as_input = data_cfgs["target_as_input"]
        self.constant_only = data_cfgs["constant_only"]
        if self.target_as_input and (not self.train_mode):
            # if the target is used as input and train_mode is False,
            # we need to get the target data in training period to generate pbm params
            self.train_dataset = DplDataset(data_cfgs, is_tra_val_te="train")

    def __getitem__(self, item):
        """
        Get one mini-batch for dPL (differential parameter learning) model

        TODO: not check target_as_input and constant_only cases yet

        Parameters
        ----------
        item
            index

        Returns
        -------
        tuple
            a mini-batch data;
            x_train (not normalized forcing), z_train (normalized data for DL model), y_train (not normalized output)
        """
        warmup = self.warmup_length
        rho = self.rho
        horizon = self.horizon
        if self.train_mode:
            xc_norm, _ = super(DplDataset, self).__getitem__(item)
            basin, time = self.lookup_table[item]
            if self.target_as_input:
                # y_morn and xc_norm are concatenated and used for DL model
                y_norm = torch.from_numpy(
                    self.y[basin, time - warmup: time + rho + horizon, :]
                ).float()
                # the order of xc_norm and y_norm matters, please be careful!
                z_train = torch.cat((xc_norm, y_norm), -1)
            elif self.constant_only:
                # only use attributes data for DL model
                z_train = torch.from_numpy(self.c[basin, :]).float()
            else:
                z_train = xc_norm.float()
            x_train = self.x_origin[basin, time - warmup: time + rho + horizon, :]
            y_train = self.y_origin[basin, time: time + rho + horizon, :]
        else:
            x_norm = self.x[item, :, :]
            if self.target_as_input:
                # when target_as_input is True,
                # we need to use training data to generate pbm params
                x_norm = self.train_dataset.x[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                xc_norm = torch.from_numpy(x_norm).float()
            else:
                c_norm = self.c[item, :]
                c_norm = (
                    np.repeat(c_norm, x_norm.shape[0], axis=0)
                    .reshape(c_norm.shape[0], -1)
                    .T
                )
                xc_norm = torch.from_numpy(
                    np.concatenate((x_norm, c_norm), axis=1)
                ).float()
            if self.target_as_input:
                # when target_as_input is True,
                # we need to use training data to generate pbm params
                # when used as input, warmup_length not included for y
                y_norm = torch.from_numpy(self.train_dataset.y[item, :, :]).float()
                # the order of xc_norm and y_norm matters, please be careful!
                z_train = torch.cat((xc_norm, y_norm), -1)
            elif self.constant_only:
                # only use attributes data for DL model
                z_train = torch.from_numpy(self.c[item, :]).float()
            else:
                z_train = xc_norm
            x_train = self.x_origin[item, :, :]
            y_train = self.y_origin[item, warmup:, :]
        return (
            torch.from_numpy(x_train).float(),
            z_train,
        ), torch.from_numpy(y_train).float()

    def __len__(self):
        return self.num_samples if self.train_mode else len(self.t_s_dict["sites_id"])


class FlexibleDataset(BaseDataset):
    """A dataset whose datasources are from multiple sources according to the configuration"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(FlexibleDataset, self).__init__(data_cfgs, is_tra_val_te)

    @property
    def data_source(self):
        source_cfgs = self.data_cfgs["source_cfgs"]
        return {
            name: data_sources_dict[name](path)
            for name, path in zip(
                source_cfgs["source_names"], source_cfgs["source_paths"]
            )
        }

    def _read_xyc(self):
        var_to_source_map = self.data_cfgs["var_to_source_map"]
        x_datasets, y_datasets, c_datasets = [], [], []
        gage_ids = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]

        for var_name in var_to_source_map:
            source_name = var_to_source_map[var_name]
            data_source_ = self.data_source[source_name]
            if var_name in self.data_cfgs["relevant_cols"]:
                x_datasets.append(
                    data_source_.read_ts_xrdataset(gage_ids, t_range, [var_name])
                )
            elif var_name in self.data_cfgs["target_cols"]:
                y_datasets.append(
                    data_source_.read_ts_xrdataset(gage_ids, t_range, [var_name])
                )
            elif var_name in self.data_cfgs["constant_cols"]:
                c_datasets.append(
                    data_source_.read_attr_xrdataset(gage_ids, [var_name])
                )

        # 合并所有x, y, c类型的数据集
        x = xr.merge(x_datasets) if x_datasets else xr.Dataset()
        y = xr.merge(y_datasets) if y_datasets else xr.Dataset()
        c = xr.merge(c_datasets) if c_datasets else xr.Dataset()
        if "streamflow" in y:
            area = data_source_.camels.read_area(self.t_s_dict["sites_id"])
            y.update(streamflow_unit_conv(y[["streamflow"]], area))
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            x, y, c
        )

    def _normalize(self):
        var_to_source_map = self.data_cfgs["var_to_source_map"]
        for var_name in var_to_source_map:
            source_name = var_to_source_map[var_name]
            data_source_ = self.data_source[source_name]
            break
        scaler_hub = ScalerHub(
            self.y_origin,
            self.x_origin,
            self.c_origin,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=data_source_.camels,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.x, scaler_hub.y, scaler_hub.c


class HydroMeanDataset(BaseDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(HydroMeanDataset, self).__init__(data_cfgs, is_tra_val_te)

    @property
    def data_source(self):
        time_unit = (
            str(self.data_cfgs["min_time_interval"]) + self.data_cfgs["min_time_unit"]
        )
        return SelfMadeHydroDataset(
            self.data_cfgs["source_cfgs"]["source_path"],
            time_unit=[time_unit],
        )

    def _normalize(self):
        x, y, c = super()._normalize()
        return x.compute(), y.compute(), c.compute()

    def _read_xyc(self):
        data_target_ds = self._prepare_target()
        if data_target_ds is not None:
            y_origin = self._trans2da_and_setunits(data_target_ds)
        else:
            y_origin = None

        data_forcing_ds = self._prepare_forcing()
        if data_forcing_ds is not None:
            x_origin = self._trans2da_and_setunits(data_forcing_ds)
        else:
            x_origin = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.data_source.read_BA_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
                self.data_cfgs["source_cfgs"]["source_path"]["attributes"],
            )
            c_orgin = self._trans2da_and_setunits(data_attr_ds)
        else:
            c_orgin = None
        self.x_origin, self.y_origin, self.c_origin = x_origin, y_origin, c_orgin

    def __len__(self):
        return self.num_samples

    def _prepare_forcing(self):
        return self._read_from_minio(self.data_cfgs["relevant_cols"])

    def _prepare_target(self):
        return self._read_from_minio(self.data_cfgs["target_cols"])

    def _read_from_minio(self, var_lst):
        gage_id_lst = self.t_s_dict["sites_id"]
        t_range = self.t_s_dict["t_final_range"]
        interval = self.data_cfgs["min_time_interval"]
        time_unit = (
            str(self.data_cfgs["min_time_interval"]) + self.data_cfgs["min_time_unit"]
        )

        subset_list = []
        for start_date, end_date in t_range:
            adjusted_end_date = (
                datetime.strptime(end_date, "%Y-%m-%d-%H") + timedelta(hours=interval)
            ).strftime("%Y-%m-%d-%H")
            subset = self.data_source.read_ts_xrdataset(
                gage_id_lst,
                t_range=[start_date, adjusted_end_date],
                var_lst=var_lst,
                time_units=[time_unit],
            )
            subset_list.append(subset[time_unit])
        return xr.concat(subset_list, dim="time")


class Seq2SeqDataset(HydroMeanDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(Seq2SeqDataset, self).__init__(data_cfgs, is_tra_val_te)

    def _read_xyc(self):
        data_target_ds = self._prepare_target()
        if data_target_ds is not None:
            y_origin = self._trans2da_and_setunits(data_target_ds)
        else:
            y_origin = None

        data_forcing_ds = self._prepare_forcing()
        if data_forcing_ds is not None:
            x_origin = self._trans2da_and_setunits(data_forcing_ds)
            x_origin = xr.where(x_origin < 0, float("nan"), x_origin)
        else:
            x_origin = None

        if self.data_cfgs["constant_cols"]:
            data_attr_ds = self.data_source.read_attr_xrdataset(
                self.t_s_dict["sites_id"],
                self.data_cfgs["constant_cols"],
            )
            c_orgin = self._trans2da_and_setunits(data_attr_ds)
        else:
            c_orgin = None
        self.x_origin, self.y_origin, self.c_origin = x_origin, y_origin, c_orgin

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon
        prec = self.data_cfgs["prec_window"]

        p = self.x[basin, idx + 1: idx + rho + horizon + 1, 0].reshape(-1, 1)
        s = self.x[basin, idx: idx + rho, 1:]
        x = np.concatenate((p[:rho], s), axis=1)

        c = self.c[basin, :]
        c = np.tile(c, (rho + horizon, 1))
        x = np.concatenate((x, c[:rho]), axis=1)

        x_h = np.concatenate((p[rho:], c[rho:]), axis=1)
        y = self.y[basin, idx + rho - prec + 1: idx + rho + horizon + 1, :]

        if self.is_tra_val_te == "train":
            return [
                torch.from_numpy(x).float(),
                torch.from_numpy(x_h).float(),
                torch.from_numpy(y).float(),
            ], torch.from_numpy(y).float()
        return [
            torch.from_numpy(x).float(),
            torch.from_numpy(x_h).float(),
        ], torch.from_numpy(y).float()


class TransformerDataset(Seq2SeqDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        super(TransformerDataset, self).__init__(data_cfgs, is_tra_val_te)

    def __getitem__(self, item: int):
        basin, idx = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon

        p = self.x[basin, idx + 1: idx + rho + horizon + 1, 0]
        s = self.x[basin, idx: idx + rho, 1]
        x = np.stack((p[:rho], s), axis=1)

        c = self.c[basin, :]
        c = np.tile(c, (rho + horizon, 1))
        x = np.concatenate((x, c[:rho]), axis=1)

        x_h = np.concatenate((p[rho:].reshape(-1, 1), c[rho:]), axis=1)
        y = self.y[basin, idx + rho + 1: idx + rho + horizon + 1, :]

        return [
            torch.from_numpy(x).float(),
            torch.from_numpy(x_h).float(),
        ], torch.from_numpy(y).float()


class GNNDataset(Seq2SeqDataset):
    def __init__(self, data_cfgs: dict, is_tra_val_te: str, graph_tuple):
        super(GNNDataset, self).__init__(data_cfgs, is_tra_val_te)
        self.graph_tuple = graph_tuple
        upstream_ds = self.get_upstream_ds()
        upstream_ds['basin_id'] = upstream_ds['basin_id'].astype(str).str.zfill(8)
        self.x_up = upstream_ds

    def __getitem__(self, item: int):
        # 从lookup_table中获取的idx和basin是整数，但是total_df的basin是字符串，所以需要转换一下
        basin, idx = self.lookup_table[item]
        rho = self.rho
        horizon = self.horizon
        prec = self.data_cfgs["prec_window"]
        basin_id = self.basins[basin].split('_')[-1]
        str_lev_array = self.x_up.sel(basin_id=basin_id).to_array()
        # 在这里p和s的间隔应该是1吗?
        stream_up_p = str_lev_array[0][idx + 1: idx + rho + horizon + 1]
        stream_up_s = str_lev_array[0][idx: idx + rho]
        x_ps_up = np.stack((stream_up_p[:rho], stream_up_s), axis=1)
        c = self.c[basin, :]
        c = np.tile(c, (rho + horizon, 1))
        x = np.concatenate((self.x[basin, :rho, :], x_ps_up, c[:rho]), axis=1)
        x_h = np.concatenate((np.expand_dims(stream_up_p[rho:], -1), c[rho:]), axis=1)
        y = self.y[basin, idx + rho - prec + 1: idx + rho + horizon + 1, :]
        if y.shape[0] < horizon + prec:
           y = np.pad(y, ((0, horizon + prec - y.shape[0]), (0, 0)), 'edge')
        result = (([torch.from_numpy(x).float(), torch.from_numpy(x_h).float(), torch.from_numpy(y).float()],
                   torch.from_numpy(y).float())
                  if self.is_tra_val_te == "train" else ([torch.from_numpy(x).float(), torch.from_numpy(x_h).float()],
                                                         torch.from_numpy(y).float()))
        return result

    def __len__(self):
        # 15118/train, 2626/test
        '''
        if (self.is_tra_val_te == "train") | (self.is_tra_val_te == "valid"):
            return self.num_samples // self.data_cfgs['batch_size']
        else:
            return self.num_samples
        '''
        return self.num_samples

    @property
    def basins(self):
        """Return the basins of the dataset"""
        return ([site.split('_')[-1] for site in self.t_s_dict["sites_id"] if len(site.split('_')) == 2] +
                [site for site in self.t_s_dict["sites_id"] if len(site.split('_')) == 3])

    def get_upstream_ds(self):
        # GNNMultiTaskHydro.get_upstream_graph方法，如果遇到iowa的站必须除以流域面积做平均，否则误差会极大
        res_dir = self.data_cfgs["test_path"]
        tra_val_te_ds = os.path.join(res_dir, f'upstream_ds_{self.is_tra_val_te}.nc')
        if os.path.exists(tra_val_te_ds):
            total_ds = xr.open_dataset(tra_val_te_ds)
        else:
            nodes_df = self.graph_tuple[1]
            nodes_df['basin_id'] = nodes_df['basin_id'].astype(str).str.zfill(8)
            nodes_df['node_id'] = nodes_df['node_id'].astype(str)
            max_app_cols = int(nodes_df['upstream_len'].max()) - 1
            total_df = pd.DataFrame()
            for node_name in self.basins:
                station_df = pd.DataFrame()
                # basin_id = nodes_df[nodes_df['station_id'] == node_name]['basin_id'].to_list()[0]
                node_id = nodes_df['node_id'][nodes_df['station_id'] == node_name].to_list()[0]
                up_set = nx.ancestors(self.graph_tuple[0], node_id)
                if len(up_set) > 0:
                    up_node_names = nodes_df['station_id'][nodes_df['node_id'].isin(up_set)]
                    # read_data_with_id放在if下方减少读盘次数
                    streamflow_dfs = [self.read_data_with_id(up_node_name) for up_node_name in up_node_names]
                    for date_tuple in self.data_cfgs[f"t_range_{self.is_tra_val_te}"]:
                        date_times = pd.date_range(date_tuple[0], date_tuple[1], freq='1h')
                        up_col_dict = {f'streamflow_up_{i}': np.repeat(np.nan, len(date_times)) for i in range(max_app_cols)}
                        up_str_df = pd.DataFrame(up_col_dict)
                        for i in range(len(streamflow_dfs)):
                            data_table = streamflow_dfs[i]
                            # 有的time列是object，和datetime64[ns]不等，所以这里先转成datetime再比较
                            data_table['time'] = pd.to_datetime(data_table['time'])
                            up_str_col = data_table['streamflow'][data_table['time'].isin(date_times)]
                            up_str_df[f'streamflow_up_{i}'] = up_str_col.fillna(0)
                        up_str_df = up_str_df.fillna(0)
                        station_df['basin_id'] = np.repeat(node_name, len(date_times))
                        station_df['time'] = date_times
                        station_df = pd.concat([station_df, up_str_df], axis=1)
                        total_df = pd.concat([total_df, station_df], axis=0)
                else:
                    for date_tuple in self.data_cfgs[f"t_range_{self.is_tra_val_te}"]:
                        date_times = pd.date_range(date_tuple[0], date_tuple[1], freq='1h')
                        up_col_dict = {f'streamflow_up_{i}': np.repeat(np.nan, len(date_times)) for i in range(max_app_cols)}
                        up_str_df = pd.DataFrame(up_col_dict).fillna(0)
                        station_df['basin_id'] = np.repeat(node_name, len(date_times))
                        station_df['time'] = date_times
                        station_df = pd.concat([station_df, up_str_df], axis=1)
                        total_df = pd.concat([total_df, station_df], axis=0)
            total_df = total_df.set_index(['basin_id', 'time'])
            total_ds = xr.Dataset.from_dataframe(total_df[~total_df.index.duplicated()])
            total_ds.to_netcdf(tra_val_te_ds)
        return total_ds

    def read_data_with_id(self, node_name: str):
        import geopandas as gpd
        # node_name: IOWA, WY_DCP_XXXXX
        # iowa流量站有653个，但是数据足够多的只有222个被整编到nc文件中
        if ('_' in node_name) & (len(node_name.split('_'))==3):
            iowa_stream_ds = xr.open_dataset("/ftproot/iowa_streamflow_stas.nc")
            if node_name in iowa_stream_ds.station.values:
                node_df = iowa_stream_ds.sel(station=node_name).to_dataframe().reset_index()
                node_df = node_df.rename(columns={'utc_valid': 'time'})
                node_df = node_df[['time', 'streamflow']]
                sta_basin_df = self.graph_tuple[1]
                sta_basin_df['basin_id'] = sta_basin_df['basin_id'].astype(str).str.zfill(8)
                basin_id = sta_basin_df[sta_basin_df['station_id'] == node_name]['basin_id'].to_list()[0]
                area_gdf = gpd.read_file(self.data_cfgs['basins_shp'])
                area = area_gdf[area_gdf['BASIN_ID'].str.contains(basin_id)]['AREA'].to_list()[0]
                # iowa流量站单位是KCFS(1000 ft3/s)，这里除以流域面积，并变成m3/s
                node_df['streamflow'] = node_df['streamflow'] / (35.31 * area) * 3600
            else:
                node_df = pd.DataFrame()
        # node_name: songliao_21401550
        elif ('_' in node_name) & (len(node_name.split('_'))==2):
            node_df = pd.read_csv(f'/ftproot/basins-interim/timeseries/1h/{node_name}.csv', engine='c')[['time', 'streamflow']]
        # node_name: str(21401550)
        elif '_' not in node_name:
            csv_path = f'/ftproot/basins-interim/timeseries/1h/camels_{node_name}.csv'
            node_csv_path = csv_path if os.path.exists(csv_path) else csv_path.replace('camels', 'songliao')
            node_df = pd.read_csv(node_csv_path, engine='c')[['time', 'streamflow']]
        else:
            node_df = pd.DataFrame()
        return node_df
