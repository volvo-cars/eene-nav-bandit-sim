# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

import pandas as pd
import numpy as np


def create_charging_station_df():
    data = [{"station_id": "1", "node_id": "1", "node_y": 0.0, "node_x": 0.0, "power": 10000.0},
            {"station_id": "2", "node_id": "2", "node_y": 0.0, "node_x": 1.0, "power": 20000.0},
            {"station_id": "3", "node_id": "3", "node_y": 1.0, "node_x": 0.0, "power": 30000.0},
            {"station_id": "4", "node_id": "4", "node_y": 1.0, "node_x": 1.0, "power": 40000.0}]
    return pd.DataFrame(data=data)


def create_charging_graph_ids_df():
    data = [{"node_id": "1", "station_id": "1", "matrix_id": 0},
            {"node_id": "2", "station_id": "2", "matrix_id": 1},
            {"node_id": "3", "station_id": "3", "matrix_id": 2}]
    return pd.DataFrame(data=data)


def create_charging_graph_consumption_ndarray():
    array = np.array([[0., 36000000., 72000000.],
                      [36000000., 0., 36000000.],
                      [72000000., 36000000., 0.]])
    return array


def create_charging_graph_time_ndarray():
    array = np.array([[0., 3600., 7200.],
                      [3600., 0., 3600.],
                      [7200., 3600., 0.]])
    return array
