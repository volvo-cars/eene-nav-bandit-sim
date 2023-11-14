# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

import numpy as np

from navigation.env_impl import NavigationBanditEnvironment
from tests.utils_for_tests import create_charging_station_df
from tests.utils_for_tests import create_charging_graph_ids_df
from tests.utils_for_tests import create_charging_graph_consumption_ndarray
from tests.utils_for_tests import create_charging_graph_time_ndarray


FIXED_RANDOM_SEED = 1


def create_environment(rng):
    return NavigationBanditEnvironment(
        charging_station_df=create_charging_station_df(),
        charging_graph_ids_df=create_charging_graph_ids_df(),
        charging_graph_consumption_ndarray=create_charging_graph_consumption_ndarray(),
        charging_graph_time_ndarray=create_charging_graph_time_ndarray(),
        battery_capacity=54000000.,
        start_node="1",
        end_node="3",
        rng=rng)


def test_receive_feedback_for_action():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    environment = create_environment(rng)
    environment.update_environment()
    action = (["1", "1_2_charge", "2_queue", "2", "3"], ["2"])
    feedback = environment.receive_feedback_for_action(1, action)
    assert 4 == len(feedback[0])
    assert 1 == len(feedback[1])


def test_expected_reward_for_action():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    environment = create_environment(rng)
    environment.update_environment()
    action = (["1", "1_2_charge", "2_queue", "2", "3"], ["2"])
    expected_reward = environment.expected_reward_for_action(1, action)
    assert 0 >= expected_reward


def test_find_best_action():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    environment = create_environment(rng)
    environment.update_environment()
    action = environment.find_best_action(1)
    assert 5 == len(action[0])
    assert 1 == len(action[1])


def test_find_random_action():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    environment = create_environment(rng)
    environment.update_environment()
    action = environment.find_best_action(1)
    assert 5 <= len(action[0])
    assert 1 <= len(action[1])


def test_update_environment():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    environment = create_environment(rng)
    environment.update_environment()
    action = (["1", "1_2_charge", "2_queue", "2", "3"], ["2"])
    first_feedback = environment.receive_feedback_for_action(1, action)
    environment.update_environment()
    second_feedback = environment.receive_feedback_for_action(2, action)
    assert 1 <= len(first_feedback[1])
    assert 1 <= len(second_feedback[1])
    assert first_feedback[1] != second_feedback[1]
