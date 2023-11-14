# Copyright 2023 Volvo Car Corporation
# Licensed under Apache 2.0.

class BanditAlgorithm(object):
    """ Interface for a (generic) bandit algorithm
    """

    def select_action(self, iteration):
        """ Given the current iteration, the bandit algorithm should provide the environment
        with an action.

        :param iteration: The current iteration (i.e., time step).
        :return: The action which will be passed to the environment the current iteration.
        """
        pass

    def update_with_feedback(self, iteration, action, feedback):
        """ Given the current iteration, an (already performed) action and feedback for the
        action (provided by the environment), the algorithm should update its knowledge
        about the environment.

        :param iteration: The current iteration (i.e., time step).
        :param action: An (already performed) action.
        :param feedback: Feedback for the action.
        """
        pass
