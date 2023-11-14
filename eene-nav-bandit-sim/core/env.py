# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

class BanditEnvironment(object):
    """ Interface for a (generic) bandit environment
    """

    def receive_feedback_for_action(self, iteration, action):
        """ Given the current iteration and an action selected by a bandit algorithm to be performed,
        the bandit environment should provide the algorithm with feedback for the performed action.

        :param iteration: The current iteration (i.e., time step).
        :param action: An action (to be performed).
        :return: Feedback for the action.
        """
        pass

    def expected_reward_for_action(self, iteration, action):
        """ Given the current iteration and an action, this method should return the expected reward
        of the action.

        :param iteration: The current iteration (i.e., time step).
        :param action: An action.
        :return: Expected reward for the action.
        """
        pass

    def find_best_action(self, iteration):
        """ Given the current iteration, this method should return the best action (i.e., the action
        with the highest expected reward).

        :param iteration: The current iteration (i.e., time step).
        :return: The best action.
        """
        pass

    def find_random_action(self, iteration):
        """ Given the current iteration, this method should return a (close to uniformly) random action.

        :param iteration: The current iteration (i.e., time step).
        :return: A random action.
        """
        pass

    def update_environment(self):
        """ Update the bandit environment at the end of an iteration (if necessary).
        """
        pass
