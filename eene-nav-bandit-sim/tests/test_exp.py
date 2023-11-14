# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

from functools import partial

import numpy as np

from core.exp import BanditExperiment
from navigation.env_impl import NavigationBanditEnvironment
from navigation.alg_impl import GreedyNavigationBanditAlgorithm
from tests.utils_for_tests import create_charging_station_df
from tests.utils_for_tests import create_charging_graph_ids_df
from tests.utils_for_tests import create_charging_graph_consumption_ndarray
from tests.utils_for_tests import create_charging_graph_time_ndarray


FIXED_RANDOM_SEED = 1


def create_algorithm_environment_constructor(rng):
    env_constructor = partial(NavigationBanditEnvironment,
                              charging_station_df=create_charging_station_df(),
                              charging_graph_ids_df=create_charging_graph_ids_df(),
                              charging_graph_consumption_ndarray=create_charging_graph_consumption_ndarray(),
                              charging_graph_time_ndarray=create_charging_graph_time_ndarray(),
                              battery_capacity=54000000.,
                              start_node="1",
                              end_node="3",
                              rng=rng)
    return env_constructor


def create_bandit_algorithm_constructor(rng):
    env_constructor = create_algorithm_environment_constructor(rng)
    algorithm_constructor = partial(GreedyNavigationBanditAlgorithm,
                                    env_constructor=env_constructor,
                                    rng=rng)
    return algorithm_constructor


def create_bandit_environment_constructor(rng):
    return partial(NavigationBanditEnvironment,
                   charging_station_df=create_charging_station_df(),
                   charging_graph_ids_df=create_charging_graph_ids_df(),
                   charging_graph_consumption_ndarray=create_charging_graph_consumption_ndarray(),
                   charging_graph_time_ndarray=create_charging_graph_time_ndarray(),
                   battery_capacity=54000000.,
                   start_node="1",
                   end_node="3",
                   rng=rng)


def test_run_experiment():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    experiment = BanditExperiment(experiment_name="test_experiment",
                                  experiment_id=0,
                                  experiment_horizon=1,
                                  bandit_environment_constructor=create_bandit_environment_constructor(rng),
                                  bandit_algorithm_constructor=create_bandit_algorithm_constructor(rng))
    results = experiment.run_experiment()
    assert 1 == len(results)


def test_teardown():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    experiment = BanditExperiment(experiment_name="test_experiment",
                                  experiment_id=0,
                                  experiment_horizon=1,
                                  bandit_environment_constructor=create_bandit_environment_constructor(rng),
                                  bandit_algorithm_constructor=create_bandit_algorithm_constructor(rng))
    experiment.run_experiment()
    experiment.teardown()
    assert not hasattr(experiment, "bandit_environment")
    assert not hasattr(experiment, "bandit_algorithm")
