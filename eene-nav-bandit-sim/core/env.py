
class BanditEnvironment(object):

    def receive_feedback_for_action(self, iteration, action):
        pass

    def expected_reward_for_action(self, iteration, action):
        pass

    def find_best_action(self, iteration):
        pass

    def find_random_action(self, iteration):
        pass

    def update_environment(self):
        pass

    def replace_action_parameters(self, edge_parameters, charging_station_parameters):
        pass
