# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

from functools import partial

import numpy as np

from navigation.alg_impl import AbstractNavigationBanditAlgorithm
from navigation.alg_impl import EpsilonGreedyNavigationBanditAlgorithm
from navigation.alg_impl import GreedyNavigationBanditAlgorithm
from navigation.alg_impl import ThompsonSamplingNavigationBanditAlgorithm
from navigation.alg_impl import BayesUcbNavigationBanditAlgorithm
from navigation.env_impl import NavigationBanditEnvironment
from navigation.env_impl import QueuePrior
from navigation.env_impl import ChargePrior
from tests.utils_for_tests import create_charging_station_df
from tests.utils_for_tests import create_charging_graph_ids_df
from tests.utils_for_tests import create_charging_graph_consumption_ndarray
from tests.utils_for_tests import create_charging_graph_time_ndarray


FIXED_RANDOM_SEED = 1


def create_internal_environment_constructor(rng):
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


def test_update_algorithm_with_feedback():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    env_constructor = create_internal_environment_constructor(rng)
    algorithm = AbstractNavigationBanditAlgorithm(env_constructor=env_constructor, rng=rng)
    queue_prior = QueuePrior(a=2., b=2400.)
    charge_prior = ChargePrior(ln_p=13.5, q=300., r=3., s=3.)
    action = (["1", "1_2_charge", "2_queue", "2", "3"], ["2"])
    feedback = ({"1": {"1_2_charge": -3600.},
                 "1_2_charge": {"2_queue": -1000.},
                 "2_queue": {"2": -300.},
                 "2": {"3": -3600.}},
                [[-300., -1000.]])
    algorithm.update_with_feedback(1, action, feedback)
    ((alpha1, beta1), (ln_p1, q1, r1, s1)) = algorithm.posterior_parameters["2"]
    assert queue_prior.a != alpha1
    assert queue_prior.b != beta1
    assert charge_prior.ln_p != ln_p1
    assert charge_prior.q != q1
    assert charge_prior.r != r1
    assert charge_prior.s != s1


def test_select_action_epsilon_greedy():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    env_constructor = create_internal_environment_constructor(rng)
    algorithm = EpsilonGreedyNavigationBanditAlgorithm(env_constructor=env_constructor,
                                                       epsilon_function=lambda t: 1. / np.sqrt(t),
                                                       rng=rng)
    action = algorithm.select_action(1)
    assert 2 == len(action)
    assert 5 <= len(action[0])
    assert 1 <= len(action[1])


def test_select_action_greedy():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    env_constructor = create_internal_environment_constructor(rng)
    algorithm = GreedyNavigationBanditAlgorithm(env_constructor=env_constructor,
                                                rng=rng)
    action = algorithm.select_action(1)
    assert 2 == len(action)
    assert 5 <= len(action[0])
    assert 1 <= len(action[1])


def test_select_action_thompson_sampling():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    env_constructor = create_internal_environment_constructor(rng)
    algorithm = ThompsonSamplingNavigationBanditAlgorithm(env_constructor=env_constructor,
                                                          rng=rng)
    action = algorithm.select_action(1)
    assert 2 == len(action)
    assert 5 <= len(action[0])
    assert 1 <= len(action[1])


def test_select_action_bayes_ucb():
    rng = np.random.default_rng(FIXED_RANDOM_SEED)
    env_constructor = create_internal_environment_constructor(rng)
    algorithm = BayesUcbNavigationBanditAlgorithm(env_constructor=env_constructor,
                                                  rng=rng)
    action = algorithm.select_action(1)
    assert 2 == len(action)
    assert 5 <= len(action[0])
    assert 1 <= len(action[1])
