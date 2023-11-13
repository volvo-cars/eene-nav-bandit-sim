import logging
import argparse
from collections import OrderedDict, deque, defaultdict

import numpy as np
import pandas as pd
import networkx as nx

from util.road_graph import RoadGraph, Edge, VertexOutgoingEdges


def main():
    parser = argparse.ArgumentParser(description='Run charging graph preprocessing script')
    parser.add_argument('--road-graph-csv-path', help='road_graph_csv_path', type=str)
    parser.add_argument('--charging-station-csv-path', help='charging_station_csv_path', type=str)
    parser.add_argument('--output-dir', help='output_dir', type=str, default='../output/')
    parser.add_argument('--logging-level', help='logging_level', type=str, default='info')
    args, unknown_args = parser.parse_known_args()

    # Configure logging
    configure_logging(args)
    logging.info("Starting charging graph preprocessing script!")

    logging.info('Starting charging station converter')
    graph_df = pd.read_csv(args.road_graph_csv_path)
    logging.info('Loaded source network DataFrame, size: ' + str(graph_df.shape[0]))

    charging_station_df = pd.read_csv(args.charging_station_csv_path)
    logging.info('Loaded source charging stations DataFrame, size: ' + str(charging_station_df.shape[0]))

    filtered_charging_station_df = filter_charging_stations(charging_station_df)
    logging.info('Filtered source charging stations DataFrame, size: ' + str(filtered_charging_station_df.shape[0]))

    source_graph = create_graph(graph_df)
    nx_graph = nx.DiGraph()
    for from_node in source_graph:
        for to_node in source_graph[from_node]:
            nx_graph.add_edge(from_node, to_node, weight=source_graph[from_node][to_node][0])
    logging.info('Created source network graph')

    source_charging_stations = create_charging_station_dictionary(filtered_charging_station_df)
    logging.info('Created charging station dictionary')

    logging.info('Starting charging station preprocessing')
    finished_time_arrays = list()
    finished_consumption_arrays = list()
    for from_idx, from_station_id in enumerate(source_charging_stations):
        from_station = source_charging_stations[from_station_id]
        logging.info('Station no. ' + str(from_idx) + ': Starting shortest path')

        predecessors, _ = nx.dijkstra_predecessor_and_distance(nx_graph, from_station['node_id'])
        distances = add_consumption_to_distances(predecessors, source_graph, from_station['node_id'])
        logging.info('Station no. ' + str(from_idx) + ': Finished shortest path')

        time_array = np.inf * np.ones((1, len(source_charging_stations)))
        consumption_array = np.inf * np.ones((1, len(source_charging_stations)))

        for to_idx, to_station_id in enumerate(source_charging_stations):
            to_node_id = source_charging_stations[to_station_id]['node_id']
            if to_node_id in distances:
                time, consumption = distances[to_node_id]
                time_array[0, to_idx] = time
                consumption_array[0, to_idx] = consumption
        logging.info('Station no. ' + str(from_idx) + ': Finished charging station lookup')

        finished_time_arrays.append(time_array)
        finished_consumption_arrays.append(consumption_array)

    logging.info('Finished all charging stations')
    complete_time_array = np.vstack(finished_time_arrays)
    complete_consumption_array = np.vstack(finished_consumption_arrays)
    logging.info('Stacked all arrays')

    np.save(args.output_dir + 'complete_time_array.npy', complete_time_array)
    np.save(args.output_dir + 'complete_consumption_array.npy', complete_consumption_array)
    logging.info('Saved time and consumption arrays')

    station_ids = []
    node_ids = []
    matrix_ids = []
    for matrix_id, station_id in enumerate(source_charging_stations):
        station_ids.append(station_id)
        node_ids.append(source_charging_stations[station_id]['node_id'])
        matrix_ids.append(matrix_id)
    charging_id_df = pd.DataFrame({'station_id': station_ids,
                                   'node_id': node_ids,
                                   'matrix_id': matrix_ids})
    charging_id_df.to_csv(args.output_dir + 'charging_graph_ids.csv')
    logging.info('Stored charging graph IDs')
    logging.info('Finished preprocessing charging station graph, closing.')


def create_graph(graph_df):
    graph = RoadGraph()
    for df_row in graph_df.to_dict('records'):
        expected_weight = np.array([df_row['mean_time'], df_row['mean_consumption']])

        edge = Edge(expected_weight)

        # From-node ID, lat, lon
        edge.from_node = str(df_row['from_id'])
        edge.from_lat = df_row['from_y']
        edge.from_lon = df_row['from_x']

        # To-node ID, lat, lon
        edge.to_node = str(df_row['to_id'])
        edge.to_lat = df_row['to_y']
        edge.to_lon = df_row['to_x']

        # Expected weight
        edge.expected_weight = expected_weight

        # Set edge in forward direction
        if edge.from_node not in graph:
            graph[edge.from_node] = VertexOutgoingEdges(graph)
        graph[edge.from_node].set_edge(edge.to_node, edge)
    return graph


def create_charging_station_dictionary(charging_station_df):
    charging_stations = OrderedDict()
    for df_row in charging_station_df.to_dict('records'):
        charging_station = dict()

        # Charging station base attributes
        station_id = str(df_row['station_id'])
        charging_station['station_id'] = station_id
        charging_station['node_id'] = str(df_row['node_id'])
        charging_station['node_y'] = df_row['node_y']
        charging_station['node_x'] = df_row['node_x']
        charging_station['power'] = df_row['power']

        charging_stations[station_id] = charging_station
    return charging_stations


def add_consumption_to_distances(predecessors, source_graph, start_node):
    successors = defaultdict(list)
    for successor, predecessor in predecessors.items():
        if len(predecessor) > 0:
            successors[predecessor[0]].append(successor)
    consumption_distances = dict()
    node_queue = deque()
    node_queue.append(start_node)
    consumption_distances[start_node] = np.array([0., 0.])
    while len(node_queue) > 0:
        node = node_queue.pop()
        for successor in successors[node]:
            consumption_distance = source_graph[node][successor]
            consumption_distances[successor] = consumption_distances[node] + consumption_distance
            node_queue.append(successor)
    return consumption_distances


def filter_charging_stations(charging_station_df):
    return charging_station_df.loc[charging_station_df.groupby(['node_id'])['power'].idxmax()]


def configure_logging(args):
    if args.logging_level.lower() == 'debug':
        logging_level = logging.DEBUG
    elif args.logging_level.lower() == 'info':
        logging_level = logging.INFO
    elif args.logging_level.lower() == 'warning':
        logging_level = logging.WARNING
    elif args.logging_level.lower() == 'error':
        logging_level = logging.ERROR
    elif args.logging_level.lower() == 'critical':
        logging_level = logging.CRITICAL
    else:
        logging_level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging_level)


if __name__ == '__main__':
    main()
