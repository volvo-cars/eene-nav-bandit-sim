# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

import logging
from typing import Callable
from collections import defaultdict

import numpy as np
import scipy.stats as st
from scipy.stats.sampling import TransformedDensityRejection

from core.alg import BanditAlgorithm
from navigation.env_impl import NavigationBanditEnvironment, QueuePrior, ChargePrior, ChargingEdgeType
from util.utils_gamma import gamma_prior_alpha_mode, GammaConjugatePriorAlphaDist
from util.common import mc_estimate_of_mean_charge_power_reciprocal, MillerDummyModeSampler


class NavigationBanditAlgorithm(BanditAlgorithm):

    def __init__(self,
                 env_constructor: Callable[[], NavigationBanditEnvironment],
                 queue_prior=QueuePrior(a=2., b=2400.),
                 charge_prior=ChargePrior(ln_p=13.5, q=300., r=3., s=3.),
                 min_power_factor=0.5,
                 power_scale=300.,
                 rng=np.random.default_rng()):
        self.queue_prior = queue_prior
        self.charge_prior = charge_prior
        self.min_power_factor = min_power_factor
        self.power_scale = power_scale
        self.rng = rng

        logging.debug('NavigationBanditAlgorithm: Building internal environment')
        self.bandit_algorithm_environment = env_constructor()
        logging.debug('NavigationBanditAlgorithm: Finished internal environment')

        parameter_dicts = self._create_posterior_parameter_dictionaries(self.bandit_algorithm_environment,
                                                                        self.queue_prior,
                                                                        self.charge_prior)
        self.posterior_parameters = parameter_dicts[0]
        self.posterior_alpha_modes = parameter_dicts[1]

        BanditAlgorithm.__init__(self)

    def _create_posterior_parameter_dictionaries(self, bandit_env, queue_prior, charge_prior):
        posterior_parameter_dict = dict()
        posterior_alpha_mode_dict = dict()
        for station_id in bandit_env.charging_stations:
            alpha0 = queue_prior.a
            beta0 = queue_prior.b
            ln_p0 = charge_prior.ln_p
            q0 = charge_prior.q
            r0 = charge_prior.r
            s0 = charge_prior.s
            posterior_parameter_dict[station_id] = ((alpha0, beta0), (ln_p0, q0, r0, s0))
            alpha_hat = gamma_prior_alpha_mode(ln_p0, q0, r0, s0)
            posterior_alpha_mode_dict[station_id] = alpha_hat
        return posterior_parameter_dict, posterior_alpha_mode_dict

    def _feedback_transformation(self, edge, charging_station_dict, feedback_tuple):
        queue_time = -feedback_tuple[0]
        charging_time = -feedback_tuple[1]
        (_, mean_consumption) = edge.parameters
        if mean_consumption <= 0. or charging_time <= 0.:
            # Assume that the power is equal to the maximum if the consumption is zero
            charging_power = charging_station_dict['power']
        else:
            charging_power = mean_consumption / charging_time
        max_power = charging_station_dict['power']
        gamma_sample = (max_power - charging_power) / self.power_scale
        return -queue_time, -gamma_sample

    def _expectation_transformation(self, charging_station_dict, parameter_tuple):
        mean_queue_time = parameter_tuple[0]
        charge_alpha = parameter_tuple[2]
        charge_beta = parameter_tuple[3]
        max_power = charging_station_dict['power']
        min_power = self.min_power_factor * max_power
        charging_power_reciprocal = mc_estimate_of_mean_charge_power_reciprocal(charge_alpha,
                                                                                charge_beta,
                                                                                min_power,
                                                                                max_power)
        return mean_queue_time, 1./charging_power_reciprocal

    def update_with_feedback(self, iteration, action, feedback):
        station_ids = action[1]
        station_feedbacks = feedback[1]

        # Find trip edges before each charging station
        station_trip_edges = dict()
        for i in range(len(action[0]) - 1):
            from_node = action[0][i]
            to_node = action[0][i+1]
            edge = self.bandit_algorithm_environment.charging_graph[from_node].get_edge(to_node)
            if edge.edge_type == ChargingEdgeType.CHARGE:
                trip_from_node = action[0][i-1]
                station_trip_edges[edge.station_id] = self.bandit_algorithm_environment.charging_graph[trip_from_node]\
                    .get_edge(from_node)

        # Update posterior parameters for the queue time and charging power distribution parameters
        for idx, station_feedback in enumerate(station_feedbacks):

            # Feedback transformation
            trip_edge = station_trip_edges[station_ids[idx]]
            charging_station = self.bandit_algorithm_environment.charging_stations[station_ids[idx]]
            transformed_feedback = self._feedback_transformation(trip_edge, charging_station, station_feedback)
            queue_time = -1. * transformed_feedback[0]
            gamma_sample = -1. * transformed_feedback[1]

            # Update posterior parameters of the queue time distribution
            alpha0, beta0 = self.posterior_parameters[station_ids[idx]][0]
            alpha = float(alpha0 + 1.)
            beta = float(beta0 + queue_time)

            # Update posterior parameters of the charging power distribution (assumed here to be gamma)
            ln_p0, q0, r0, s0 = self.posterior_parameters[station_ids[idx]][1]
            ln_p = float(ln_p0 + np.log(gamma_sample))
            q = float(q0 + gamma_sample)
            r = float(r0 + 1.)
            s = float(s0 + 1.)

            # Calculate alpha mode
            alpha_hat = gamma_prior_alpha_mode(ln_p, q, r, s)

            # Store updated parameters
            self.posterior_parameters[station_ids[idx]] = ((alpha, beta), (ln_p, q, r, s))
            self.posterior_alpha_modes[station_ids[idx]] = alpha_hat

    def select_action(self, iteration):
        raise ValueError("For NavigationBanditAlgorithm, select_action() is not implemented!")


class EpsilonGreedyNavigationBanditAlgorithm(NavigationBanditAlgorithm):

    def __init__(self,
                 env_constructor: Callable[[], NavigationBanditEnvironment],
                 epsilon_function: Callable[[int], float],
                 queue_prior=QueuePrior(a=2., b=2400.),
                 charge_prior=ChargePrior(ln_p=13.5, q=300., r=3., s=3.),
                 min_power_factor=0.5,
                 power_scale=300.,
                 rng=np.random.default_rng()):
        self.epsilon_function = epsilon_function
        NavigationBanditAlgorithm.__init__(self,
                                           env_constructor=env_constructor,
                                           queue_prior=queue_prior,
                                           charge_prior=charge_prior,
                                           min_power_factor=min_power_factor,
                                           power_scale=power_scale,
                                           rng=rng)

    def _compute_posterior_modes(self):
        posterior_modes = dict()

        for station_id in self.posterior_parameters:
            # Mode of the posterior distribution over the queue time distribution parameters
            alpha, beta = self.posterior_parameters[station_id][0]
            if alpha >= 1:
                lambda_hat = (alpha - 1.) / beta
            else:
                # This shouldn't happen with a properly selected prior distribution.
                raise ValueError("Prior alpha parameter should be greater than or equal to 1!")
            expected_queue_time = 1. / lambda_hat

            # Mode of the posterior distribution over the transformed charge power distribution parameters
            ln_p, q, r, s = self.posterior_parameters[station_id][1]

            # Mode of the posterior distribution over alpha
            alpha_hat = self.posterior_alpha_modes[station_id]

            # Mode of the posterior distribution over beta
            alpha0 = s * alpha_hat
            beta0 = q
            if alpha0 >= 1.:
                beta_hat = (alpha0 - 1.) / beta0
            else:
                raise ValueError("This should not happen with a properly chosen prior distribution!")

            # MAP estimate of the expected value of the gamma weight distribution
            gamma_mean_estimate = alpha_hat / beta_hat

            # Expected weight
            parameters = (expected_queue_time, gamma_mean_estimate, alpha_hat, beta_hat)
            charging_station = self.bandit_algorithm_environment.charging_stations[station_id]
            (actual_queue_time_mean, actual_charge_power_mean) = self._expectation_transformation(
                charging_station,
                parameters)

            # Charging station parameters
            (_, _, _, max_power, min_power, _) = charging_station['charge_time_parameters']
            clamped_charging_power = np.minimum(max_power, np.maximum(min_power, actual_charge_power_mean))
            posterior_modes[station_id] = (actual_queue_time_mean,
                                           clamped_charging_power,
                                           (lambda_hat, alpha_hat, beta_hat))

        return posterior_modes

    def select_action(self, iteration):
        epsilon = self.epsilon_function(iteration)

        if epsilon > 0.0 and self.rng.random() < epsilon:
            (path, station_ids) = self.bandit_algorithm_environment.find_random_action(iteration)
        else:
            posterior_modes = self._compute_posterior_modes()
            self.bandit_algorithm_environment.replace_action_parameters(None, posterior_modes)
            (path, station_ids) = self.bandit_algorithm_environment.find_best_action(iteration)
        return path, station_ids


class GreedyNavigationBanditAlgorithm(EpsilonGreedyNavigationBanditAlgorithm):

    def __init__(self,
                 env_constructor: Callable[[], NavigationBanditEnvironment],
                 queue_prior=QueuePrior(a=2., b=2400.),
                 charge_prior=ChargePrior(ln_p=13.5, q=300., r=3., s=3.),
                 min_power_factor=0.5,
                 power_scale=300.,
                 rng=np.random.default_rng()):
        EpsilonGreedyNavigationBanditAlgorithm.__init__(self,
                                                        env_constructor=env_constructor,
                                                        epsilon_function=lambda t: 0.0,
                                                        queue_prior=queue_prior,
                                                        charge_prior=charge_prior,
                                                        min_power_factor=min_power_factor,
                                                        power_scale=power_scale,
                                                        rng=rng)


class ThompsonSamplingNavigationBanditAlgorithm(NavigationBanditAlgorithm):

    def __init__(self,
                 env_constructor: Callable[[], NavigationBanditEnvironment],
                 number_of_cached_samples=100,
                 queue_prior=QueuePrior(a=2., b=2400.),
                 charge_prior=ChargePrior(ln_p=13.5, q=300., r=3., s=3.),
                 min_power_factor=0.5,
                 power_scale=300.,
                 rng=np.random.default_rng()):
        self.use_mode_sampler = defaultdict(lambda: False)
        self.cached_samples = defaultdict(list)
        self.cached_samples_indices = defaultdict(lambda: 0)
        self.number_of_cached_samples = number_of_cached_samples
        NavigationBanditAlgorithm.__init__(self,
                                           env_constructor=env_constructor,
                                           queue_prior=queue_prior,
                                           charge_prior=charge_prior,
                                           min_power_factor=min_power_factor,
                                           power_scale=power_scale,
                                           rng=rng)

    def _create_sampler(self, ln_p, q, r, s, use_mode_sampler=False, mode=None):
        miller_dist = GammaConjugatePriorAlphaDist(ln_p, q, r, s)
        created_sampler = False
        used_mode_sampler = False
        current_sampler_impl = 0
        if mode is None:
            mode = gamma_prior_alpha_mode(ln_p, q, r, s)
        while not created_sampler:
            if current_sampler_impl == 0 and not use_mode_sampler:
                try:
                    sampler = TransformedDensityRejection(miller_dist, c=0., center=mode)
                    created_sampler = True
                except:
                    logging.error("Could not create TransformedDensityRejection sampler (log-concave), ln_p=" +
                                  str(ln_p) + ", q=" + str(q) + ", r=" + str(r) + ", s=" + str(s))
                    current_sampler_impl += 1
            else:
                sampler = MillerDummyModeSampler(mode)
                created_sampler = True
                used_mode_sampler = True
        return sampler, used_mode_sampler

    def _generate_posterior_samples(self, iteration):
        posterior_samples = dict()
        for station_id in self.posterior_parameters:

            # Mode of the posterior distribution over the queue time distribution parameters
            alpha, beta = self.posterior_parameters[station_id][0]
            # Sample reward distribution parameter
            lambda_star = self.rng.gamma(alpha, 1. / beta)
            expected_queue_time = 1. / lambda_star

            # Mode of the posterior distribution over the transformed charge power distribution parameters
            ln_p, q, r, s = self.posterior_parameters[station_id][1]

            # Sample alpha parameter from posterior distribution (and cache a number of samples for efficiency)
            cached_samples = self.cached_samples[station_id]
            cached_samples_index = self.cached_samples_indices[station_id]
            if cached_samples_index >= len(cached_samples):
                alpha_hat = self.posterior_alpha_modes[station_id]
                if not self.use_mode_sampler[station_id]:
                    sampler, used_mode_sampler = self._create_sampler(ln_p, q, r, s, mode=alpha_hat)
                    self.use_mode_sampler[station_id] = used_mode_sampler
                else:
                    sampler, _ = self._create_sampler(ln_p, q, r, s, use_mode_sampler=True, mode=alpha_hat)
                cached_samples = sampler.rvs(self.number_of_cached_samples)
                cached_samples_index = 0
            alpha_star = cached_samples[cached_samples_index]
            self.cached_samples[station_id] = cached_samples
            self.cached_samples_indices[station_id] = cached_samples_index + 1

            # Sample beta parameter from posterior_distribution
            beta_star = self.rng.gamma(s * alpha_star, 1. / q)

            # Expected weight given alpha_star and beta_star
            sampled_gamma_mean = alpha_star / beta_star

            parameters = (expected_queue_time, sampled_gamma_mean, alpha_star, beta_star)
            charging_station = self.bandit_algorithm_environment.charging_stations[station_id]
            (actual_queue_time_mean, actual_charge_power_mean) = self._expectation_transformation(charging_station,
                                                                                                  parameters)

            # Charging station parameters
            (_, _, _, max_power, min_power, _) = charging_station['charge_time_parameters']
            clamped_charging_power = np.minimum(max_power, np.maximum(min_power, actual_charge_power_mean))
            posterior_samples[station_id] = (actual_queue_time_mean,
                                             clamped_charging_power,
                                             (lambda_star, alpha_star, beta_star))
        return posterior_samples

    def update_with_feedback(self, iteration, action, feedback):
        NavigationBanditAlgorithm.update_with_feedback(self, iteration, action, feedback)
        station_ids = action[1]
        for station_id in station_ids:
            # Reset sample cache
            self.cached_samples_indices[station_id] = self.number_of_cached_samples

    def select_action(self, iteration):
        posterior_samples = self._generate_posterior_samples(iteration)
        self.bandit_algorithm_environment.replace_action_parameters(None, posterior_samples)
        (path, station_ids) = self.bandit_algorithm_environment.find_best_action(iteration)
        return path, station_ids


class BayesUcbNavigationBanditAlgorithm(NavigationBanditAlgorithm):

    def _compute_posterior_ucbs(self, iteration):
        posterior_parameter_list = [(station_id, self.posterior_parameters[station_id])
                                    for station_id in self.posterior_parameters]
        posterior_queue_ucb_list = st.gamma.ppf(1. - 1. / (iteration + 1.),
                                                a=[alpha for (_, ((alpha, _), _)) in posterior_parameter_list],
                                                scale=[1. / beta for (_, ((_, beta), _)) in posterior_parameter_list])

        posterior_charge_alpha_estimate = np.array([self.posterior_alpha_modes[station_id] for (station_id, _)
                                                    in posterior_parameter_list]).flatten()

        posterior_charge_beta_ucb = st.gamma.ppf(1. - 1. / (iteration + 1.),
                                                 a=[alpha_hat * s for (alpha_hat, (_, (_, (_, _, _, s))))
                                                    in zip(posterior_charge_alpha_estimate,
                                                           posterior_parameter_list)],
                                                 scale=[1 / q for (_, (_, (_, q, _, _))) in posterior_parameter_list])
        posterior_charge_gamma_ucb = np.array([alpha_hat / beta_star for (alpha_hat, beta_star)
                                               in zip(posterior_charge_alpha_estimate,
                                                      posterior_charge_beta_ucb)]).flatten()

        posterior_ucb = {
            station_id: self._expectation_transformation(self.bandit_algorithm_environment.charging_stations[station_id],
                                                         (1. / queue_time_ucb,
                                                          np.array([gamma_ucb]),
                                                          alpha_est,
                                                          beta_ucb))
            for ((station_id, _), queue_time_ucb, gamma_ucb, alpha_est, beta_ucb)
            in zip(posterior_parameter_list, posterior_queue_ucb_list,
                   posterior_charge_gamma_ucb, posterior_charge_alpha_estimate,
                   posterior_charge_beta_ucb)}

        for (station_id, _) in posterior_parameter_list:
            # Clamp charging station parameters
            charging_station = self.bandit_algorithm_environment.charging_stations[station_id]
            (_, _, _, max_power, min_power, _) = charging_station['charge_time_parameters']
            (actual_queue_time_mean, actual_charge_power_mean) = posterior_ucb[station_id]
            clamped_charging_power = np.minimum(max_power, np.maximum(min_power, actual_charge_power_mean))
            posterior_ucb[station_id] = (actual_queue_time_mean, clamped_charging_power)

        return posterior_ucb

    def select_action(self, iteration):
        posterior_ucbs = self._compute_posterior_ucbs(iteration)
        self.bandit_algorithm_environment.replace_action_parameters(None, posterior_ucbs)
        (path, station_ids) = self.bandit_algorithm_environment.find_best_action(iteration)
        return path, station_ids
