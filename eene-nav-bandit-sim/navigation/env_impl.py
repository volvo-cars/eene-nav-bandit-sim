import logging

from enum import Enum
from collections import defaultdict, namedtuple
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats.sampling import TransformedDensityRejection

from core.env import BanditEnvironment
from util.road_graph import RoadGraph, Edge, VertexOutgoingEdges
from util.utils_gamma import GammaConjugatePriorAlphaDist
from util.common import mc_estimate_of_mean_charge_power_reciprocal


class ChargingEdgeType(Enum):
    TRIP = 0
    CHARGE = 1
    QUEUE = 2


QueuePrior = namedtuple('QueuePrior', 'a b')


ChargePrior = namedtuple('ChargePrior', 'ln_p q r s')


class NavigationBanditEnvironment(BanditEnvironment):

    def __init__(self,
                 charging_station_df: pd.DataFrame,
                 charging_graph_ids_df: pd.DataFrame,
                 charging_graph_consumption_ndarray: np.ndarray,
                 charging_graph_time_ndarray: np.ndarray,
                 battery_capacity: str,
                 start_node: str,
                 end_node: str,
                 queue_prior=QueuePrior(a=2., b=2400.),
                 charge_prior=ChargePrior(ln_p=13.5, q=300., r=3., s=3.),
                 min_power_factor=0.5,
                 power_scale=300.,
                 rng=np.random.default_rng()):
        self.charging_graph_consumption_ndarray = charging_graph_consumption_ndarray
        self.charging_graph_time_ndarray = charging_graph_time_ndarray
        self.charging_graph_power_ndarray = np.zeros(charging_graph_consumption_ndarray.shape[0])
        self.battery_capacity = battery_capacity
        self.start_node = start_node
        self.end_node = end_node
        self.queue_prior = queue_prior
        self.charge_prior = charge_prior
        self.min_power_factor = min_power_factor
        self.power_scale = power_scale
        self.rng = rng

        logging.debug('NavigationBanditEnvironment: Building charging station dictionary')
        self.charging_stations = self._create_charging_station_dictionary(charging_station_df, charging_graph_ids_df)

        logging.debug('NavigationBanditEnvironment: Building charging network graph')
        charging_graph_tuple = self._create_charging_graph(charging_graph_ids_df,
                                                           self.charging_graph_consumption_ndarray,
                                                           self.charging_graph_time_ndarray,
                                                           self.charging_graph_power_ndarray,
                                                           self.charging_stations)
        self.charging_graph = charging_graph_tuple[0]
        self.nx_charging_graph = charging_graph_tuple[1]
        self.node_idx_to_matrix_idx = charging_graph_tuple[2]
        self.node_idx_to_charge_origin_idx = charging_graph_tuple[3]
        logging.debug('NavigationBanditEnvironment: Finished building base data structures')

    def _create_charging_station_dictionary(self, charging_station_df, charging_graph_ids_df):

        filter_dict = {str(sid): str(nid) for (sid, nid) in zip(charging_graph_ids_df['station_id'],
                                                                charging_graph_ids_df['node_id'])}

        charging_stations = dict()
        for df_row in charging_station_df.to_dict('records'):
            station_id = str(df_row['station_id'])
            node_id = str(df_row['node_id'])
            if station_id in filter_dict:
                if node_id != filter_dict[station_id]:
                    raise ValueError('Invalid station ID & node ID pair in charging graph ID DataFrame!')

                charging_station = dict()
                _, charging_station_parameters = self._generate_base_action_parameters(None, df_row)

                # Charging station base attributes

                charging_station['station_id'] = station_id
                charging_station['node_id'] = node_id
                charging_station['node_y'] = df_row['node_y']
                charging_station['node_x'] = df_row['node_x']
                charging_station['power'] = df_row['power']

                # Charging station changeable attributes
                charging_station['queue_time_parameters'] = charging_station_parameters[0]
                charging_station['charge_time_parameters'] = charging_station_parameters[1]

                charging_stations[station_id] = charging_station
        return charging_stations

    def _create_trip_edge(self, from_station, to_station, parameters, final_station=False):
        trip_edge = Edge(0.)
        trip_edge.parameters = parameters
        trip_edge.edge_type = ChargingEdgeType.TRIP
        trip_edge.from_lat = from_station['node_y']
        trip_edge.from_lon = from_station['node_x']
        trip_edge.from_node = str(from_station['node_id'])
        trip_edge.to_lat = to_station['node_y']
        trip_edge.to_lon = to_station['node_x']
        if final_station:
            trip_edge.to_node = str(to_station['node_id'])
        else:
            trip_edge.to_node = trip_edge.from_node + '_' + str(to_station['node_id']) + '_charge'
        return trip_edge

    def _create_charge_edge(self, from_station, to_station, parameters, station_id):
        charge_edge = Edge(0.)
        charge_edge.edge_type = ChargingEdgeType.CHARGE
        charge_edge.from_node = str(from_station['node_id']) + '_' + str(to_station['node_id']) + '_charge'
        charge_edge.from_lat = to_station['node_y']
        charge_edge.from_lon = to_station['node_x']
        charge_edge.to_node = str(to_station['node_id']) + '_queue'
        charge_edge.to_lat = to_station['node_y']
        charge_edge.to_lon = to_station['node_x']
        charge_edge.station_id = station_id
        charge_edge.parameters = parameters
        return charge_edge

    def _create_queue_edge(self, from_station, to_station, parameters, station_id):
        queue_edge = Edge(0.)
        queue_edge.edge_type = ChargingEdgeType.QUEUE
        queue_edge.from_node = str(to_station['node_id']) + '_queue'
        queue_edge.from_lat = to_station['node_y']
        queue_edge.from_lon = to_station['node_x']
        queue_edge.to_node = str(to_station['node_id'])
        queue_edge.to_lat = to_station['node_y']
        queue_edge.to_lon = to_station['node_x']
        queue_edge.station_id = station_id
        queue_edge.parameters = parameters
        return queue_edge

    def _create_charging_graph(self,
                               charging_graph_ids_df,
                               charging_graph_consumption,
                               charging_graph_time,
                               charging_graph_power,
                               charging_stations):
        charging_graph = RoadGraph()
        record_list = charging_graph_ids_df.to_dict('records')
        matrix_idx_to_stations = [None for i in range(len(record_list))]
        node_idx_to_matrix_idx = dict()
        node_idx_to_charge_origin_idx = dict()
        for record in record_list:
            matrix_idx_to_stations[record['matrix_id']] = record['station_id']
        for i in range(charging_graph_time.shape[0]):
            for j in range(charging_graph_time.shape[1]):
                if charging_graph_consumption[i, j] <= self.battery_capacity:
                    from_station = charging_stations[matrix_idx_to_stations[i]]
                    to_station = charging_stations[matrix_idx_to_stations[j]]
                    trip_edge_parameters, _ = self._generate_base_action_parameters(
                        {'mean_time': charging_graph_time[i, j],
                         'mean_consumption': charging_graph_consumption[i, j]},
                        None)
                    charging_station_parameters = (to_station['queue_time_parameters'],
                                                   to_station['charge_time_parameters'])

                    # Check that the second station in the pair is not the end node
                    if self.end_node != str(to_station['node_id']):
                        # Trip edge
                        trip_edge = self._create_trip_edge(from_station, to_station,
                                                           parameters=trip_edge_parameters,
                                                           final_station=False)
                        node_idx_to_matrix_idx[trip_edge.from_node] = i
                        node_idx_to_matrix_idx[trip_edge.to_node] = j

                        # Trip edge expected weight (time)
                        trip_edge.expected_weight = self._base_action_expected_feedback(trip_edge, None)

                        # Set trip edge in forward direction
                        if trip_edge.from_node not in charging_graph:
                            charging_graph[trip_edge.from_node] = VertexOutgoingEdges(charging_graph)
                        charging_graph[trip_edge.from_node].set_edge(trip_edge.to_node, trip_edge)

                        # Charging edge
                        charge_edge = self._create_charge_edge(from_station, to_station,
                                                               parameters=(trip_edge_parameters,
                                                                           charging_station_parameters),
                                                               station_id=matrix_idx_to_stations[j])
                        node_idx_to_matrix_idx[charge_edge.from_node] = j
                        node_idx_to_matrix_idx[charge_edge.to_node] = j
                        node_idx_to_charge_origin_idx[charge_edge.from_node] = i

                        # Charging edge expected weight (time)
                        charging_graph_power[j] = charging_station_parameters[1][0]
                        charge_edge.expected_weight = self._base_action_expected_feedback(charge_edge, to_station)

                        # Set charging edge in forward direction
                        if charge_edge.from_node not in charging_graph:
                            charging_graph[charge_edge.from_node] = VertexOutgoingEdges(charging_graph)
                        charging_graph[charge_edge.from_node].set_edge(charge_edge.to_node, charge_edge)

                        # Queue edge
                        queue_edge = self._create_queue_edge(from_station, to_station,
                                                             parameters=charging_station_parameters,
                                                             station_id=matrix_idx_to_stations[j])
                        node_idx_to_matrix_idx[queue_edge.from_node] = j
                        node_idx_to_matrix_idx[queue_edge.to_node] = j

                        # Queue edge expected weight (time)
                        queue_edge.expected_weight = self._base_action_expected_feedback(queue_edge, to_station)

                        # Set queue edge in forward direction
                        if queue_edge.from_node not in charging_graph:
                            charging_graph[queue_edge.from_node] = VertexOutgoingEdges(charging_graph)
                        charging_graph[queue_edge.from_node].set_edge(queue_edge.to_node, queue_edge)
                    else:
                        # The target node of this trip is the end node, so don't place a charging station

                        # Trip edge
                        trip_edge = self._create_trip_edge(from_station, to_station, trip_edge_parameters,
                                                           final_station=True)
                        node_idx_to_matrix_idx[trip_edge.from_node] = i
                        node_idx_to_matrix_idx[trip_edge.to_node] = j

                        # Trip edge expected weight
                        trip_edge.expected_weight = self._base_action_expected_feedback(trip_edge, None)

                        # Set trip edge in forward direction
                        if trip_edge.from_node not in charging_graph:
                            charging_graph[trip_edge.from_node] = VertexOutgoingEdges(charging_graph)
                        charging_graph[trip_edge.from_node].set_edge(trip_edge.to_node, trip_edge)

        nx_charging_graph = nx.DiGraph()
        for from_node in charging_graph:
            for to_node in charging_graph[from_node]:
                edge = charging_graph[from_node].get_edge(to_node)
                nx_charging_graph.add_edge(from_node, to_node, weight=edge.expected_weight)

        return charging_graph, nx_charging_graph, node_idx_to_matrix_idx, node_idx_to_charge_origin_idx

    def _generate_base_action_parameters(self, edge_dict, charging_station_dict):
        edge_parameters = None
        charging_station_parameters = None
        if edge_dict is not None:
            edge_parameters = (edge_dict['mean_time'], edge_dict['mean_consumption'])
        if charging_station_dict is not None:

            # Queue parameters
            queue_lambda = self.rng.gamma(self.queue_prior.a, 1. / self.queue_prior.b)
            expected_queue_time = 1. / queue_lambda
            queue_parameters = (expected_queue_time, queue_lambda)

            # Charge parameters
            max_power = charging_station_dict['power']
            min_power = self.min_power_factor * max_power
            sampler = TransformedDensityRejection(GammaConjugatePriorAlphaDist(ln_p=self.charge_prior.ln_p,
                                                                               q=self.charge_prior.q,
                                                                               r=self.charge_prior.r,
                                                                               s=self.charge_prior.s),
                                                  c=0.)
            charge_alpha = sampler.rvs(1)
            charge_beta = self.rng.gamma(charge_alpha * self.charge_prior.s, 1. / self.charge_prior.q)
            charging_power_reciprocal = mc_estimate_of_mean_charge_power_reciprocal(charge_alpha,
                                                                                    charge_beta,
                                                                                    min_power,
                                                                                    max_power,
                                                                                    scale=self.power_scale)
            charging_power = 1. / charging_power_reciprocal
            charge_parameters = (charging_power, charge_alpha, charge_beta, max_power, min_power, self.power_scale)
            charging_station_parameters = (queue_parameters, charge_parameters)
        return edge_parameters, charging_station_parameters

    def _base_action_expected_feedback(self, edge, charging_station):
        if charging_station is None and edge.edge_type != ChargingEdgeType.TRIP:
            raise ValueError('Charging station should be paired with charging edge')
        if edge.edge_type == ChargingEdgeType.QUEUE:
            ((_, queue_lambda), _) = edge.parameters
            edge_expected_feedback = 1. / queue_lambda
        elif edge.edge_type == ChargingEdgeType.CHARGE:
            ((_, mean_consumption), (_, (charging_power, _, _, _, _, _))) = edge.parameters
            charging_power_reciprocal = 1. / charging_power
            edge_expected_feedback = mean_consumption * charging_power_reciprocal
        else:  # ChargingEdgeType.TRIP
            (mean_time, _) = edge.parameters
            edge_expected_feedback = mean_time
        return edge_expected_feedback

    def _base_action_stochastic_feedback(self, edge, charging_station):
        if charging_station is None and edge.edge_type != ChargingEdgeType.TRIP:
            raise ValueError('Charging station should be paired with charging edge')
        if edge.edge_type == ChargingEdgeType.QUEUE:
            edge_stochastic_feedback = charging_station['queue_time_sample']
        elif edge.edge_type == ChargingEdgeType.CHARGE:
            ((_, mean_consumption), _) = edge.parameters
            edge_stochastic_feedback = mean_consumption / charging_station['charging_power_sample']
        else:
            edge_stochastic_feedback = edge.expected_weight
        return edge_stochastic_feedback

    def _nx_weight_function(self, u, v, d):
        i = self.node_idx_to_matrix_idx[u]
        j = self.node_idx_to_matrix_idx[v]
        if i == j and u.endswith('_charge'):
            h = self.node_idx_to_charge_origin_idx[u]
            charging_time = self.charging_graph_consumption_ndarray[h, i] / self.charging_graph_power_ndarray[j]
            if not np.isnan(charging_time) and np.isfinite(charging_time):
                return charging_time
            else:
                raise ValueError("Invalid charging time in networkx graph: " + charging_time)
        else:
            return d.get("weight", 1)

    def _nx_astar_heuristic(self, u, v):
        # NOTE: This A* heuristic function currently depends on the travel times
        #       (except charging and queueing), being fixed.
        i = self.node_idx_to_matrix_idx[u]
        j = self.node_idx_to_matrix_idx[v]
        if i == j:
            return 0.
        else:
            return self.charging_graph_time_ndarray[i, j]

    def _charging_station_update(self, charging_station):
        # Queue feedback
        (_, queue_lambda) = charging_station['queue_time_parameters']
        charging_station['queue_time_sample'] = self.rng.exponential(1. / queue_lambda)
        # Charge feedback
        (_, charge_alpha, charge_beta, max_power, min_power, scale) = charging_station['charge_time_parameters']
        gamma_sample = self.rng.gamma(charge_alpha, 1. / charge_beta)
        charging_power = max(min_power, max_power - (gamma_sample * scale))
        charging_station['charging_power_sample'] = charging_power

    def receive_feedback_for_action(self, iteration, action):
        (path, station_ids) = action
        station_reward_dict = dict()
        path_reward_dict = defaultdict(dict)
        for from_node, to_node in zip(path, path[1:]):
            edge = self.charging_graph[from_node].get_edge(to_node)
            if edge.edge_type == ChargingEdgeType.CHARGE:
                station_id = edge.station_id
                station = self.charging_stations[station_id]
                charging_time = self._base_action_stochastic_feedback(edge, station)
                if station_id not in station_reward_dict:
                    station_reward_dict[station_id] = [None, None]
                station_reward_dict[station_id][1] = -charging_time
                path_reward_dict[from_node][to_node] = -charging_time
            elif edge.edge_type == ChargingEdgeType.QUEUE:
                station_id = edge.station_id
                station = self.charging_stations[station_id]
                queue_time = self._base_action_stochastic_feedback(edge, station)
                if station_id not in station_reward_dict:
                    station_reward_dict[station_id] = [None, None]
                station_reward_dict[station_id][0] = -queue_time
                path_reward_dict[from_node][to_node] = -queue_time
            else:
                trip_time = self._base_action_stochastic_feedback(edge, None)
                path_reward_dict[from_node][to_node] = -trip_time

        return path_reward_dict, [station_reward_dict[idx] for idx in station_ids]

    def expected_reward_for_action(self, iteration, action):
        (path, _) = action
        expected_cost = 0.
        for from_node, to_node in zip(path, path[1:]):
            edge = self.charging_graph[from_node].get_edge(to_node)
            if edge.edge_type == ChargingEdgeType.TRIP:
                charging_station = None
            else:
                charging_station = self.charging_stations[edge.station_id]
            expected_cost += self._base_action_expected_feedback(edge, charging_station)
        return -expected_cost

    def find_best_action(self, iteration):
        path = nx.astar_path(self.nx_charging_graph,
                             self.start_node,
                             self.end_node,
                             heuristic=self._nx_astar_heuristic,
                             weight=self._nx_weight_function)
        station_ids = []
        for from_node, to_node in zip(path, path[1:]):
            edge = self.charging_graph[from_node].get_edge(to_node)
            if edge.edge_type == ChargingEdgeType.QUEUE:
                station_ids.append(edge.station_id)
        return path, station_ids

    def find_random_action(self, iteration):
        start = self.start_node
        end = self.end_node
        filtered_station_ids = [station_id for station_id in self.charging_stations
                                if station_id != self.start_node and station_id != self.end_node]

        found_station = False
        while not found_station:
            station_id = self.rng.choice(filtered_station_ids)
            node_id = self.charging_stations[station_id]['node_id']
            try:
                first_part = nx.astar_path(self.nx_charging_graph,
                                           start,
                                           node_id,
                                           heuristic=self._nx_astar_heuristic,
                                           weight=self._nx_weight_function)
                second_part = nx.astar_path(self.nx_charging_graph,
                                            node_id,
                                            end,
                                            heuristic=self._nx_astar_heuristic,
                                            weight=self._nx_weight_function)
                path = first_part + second_part[1:]
                found_station = True
            except:
                logging.debug('Warning: Unconnected station!')

        station_ids = []
        for from_node, to_node in zip(path, path[1:]):
            edge = self.charging_graph[from_node].get_edge(to_node)
            if edge.edge_type == ChargingEdgeType.QUEUE:
                station_ids.append(edge.station_id)
        return path, station_ids

    def update_environment(self):
        for station_id in self.charging_stations:
            self._charging_station_update(self.charging_stations[station_id])

    def replace_action_parameters(self, edge_parameters, charging_station_parameters):
        if edge_parameters is not None:
            for from_node in edge_parameters:
                for to_node in edge_parameters[from_node]:
                    self.nx_charging_graph[from_node][to_node]['weight'] = edge_parameters[from_node][to_node]
        if charging_station_parameters is not None:
            for station_id in charging_station_parameters:
                node_id = self.charging_stations[station_id]['node_id']
                if self.end_node != str(node_id):
                    station_from_node = str(node_id) + '_queue'
                    station_to_node = str(node_id)
                    station_parameters = charging_station_parameters[station_id]
                    if station_parameters[0] is not None:
                        self.nx_charging_graph[station_from_node][station_to_node]['weight'] = station_parameters[0]
                    if station_parameters[1] is not None:
                        matrix_node_id = self.node_idx_to_matrix_idx[str(node_id)]
                        self.charging_graph_power_ndarray[matrix_node_id] = station_parameters[1]
