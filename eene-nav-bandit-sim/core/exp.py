# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

import logging


class BanditExperiment(object):
    """ Bandit experiment, handling sequential interaction between a bandit environment and a bandit algorithm
    """

    def __init__(self,
                 experiment_name,
                 experiment_id,
                 experiment_horizon,
                 bandit_environment_constructor,
                 bandit_algorithm_constructor):
        """ Constructor for the bandit experiment

        :param experiment_name: Name of the experiment
        :param experiment_id: ID of the experiment
        :param experiment_horizon: Horizon of the experiment (total number of iterations)
        :param bandit_environment_constructor: Zero-argument function returning a new bandit environment
        :param bandit_algorithm_constructor: Zero-argument function returning a new bandit algorithm
        """
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.experiment_horizon = experiment_horizon
        self.bandit_environment_constructor = bandit_environment_constructor
        self.bandit_algorithm_constructor = bandit_algorithm_constructor
        self.bandit_environment = None
        self.bandit_algorithm = None

    def _run_iteration(self, iteration, bandit_environment, bandit_algorithm, optimal_expected_reward):
        action = bandit_algorithm.select_action(iteration)
        feedback = bandit_environment.receive_feedback_for_action(iteration, action)
        bandit_algorithm.update_with_feedback(iteration, action, feedback)
        action_expected_reward = bandit_environment.expected_reward_for_action(iteration, action)
        instant_regret = optimal_expected_reward - action_expected_reward
        bandit_environment.update_environment()

        return {'experiment_id': self.experiment_id,
                'experiment_name': self.experiment_name,
                'iteration': iteration,
                'instant_regret': instant_regret
                }

    def run_experiment(self):
        """ Run the experiment for the specified number of iterations and return the results

        :return: Experiment results (Pandas DataFrame with regret per algorithm and iteration)
        """
        logging.debug("BanditExperiment: Starting experiment, name: " + str(self.experiment_name))
        self.bandit_environment = self.bandit_environment_constructor()
        self.bandit_algorithm = self.bandit_algorithm_constructor()

        optimal_action = self.bandit_environment.find_best_action(1)
        optimal_expected_reward = self.bandit_environment.expected_reward_for_action(1, optimal_action)
        self.bandit_environment.update_environment()

        experiment_results = []
        for iteration in range(1, self.experiment_horizon+1):
            logging.debug("BanditExperiment: Experiment iteration " + str(iteration) +
                          ", name: " + str(self.experiment_name))
            iteration_results = self._run_iteration(iteration,
                                                   self.bandit_environment,
                                                   self.bandit_algorithm,
                                                   optimal_expected_reward)
            experiment_results.append(iteration_results)

        return experiment_results

    def teardown(self):
        """ Teardown of the experiment (delete algorithm and environment objects)
        """
        del self.bandit_algorithm
        del self.bandit_environment

