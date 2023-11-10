import logging
import argparse
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.exp import BanditExperiment
from navigation.env_impl import ChargePrior
from navigation.env_impl import QueuePrior
from navigation.env_impl import NavigationBanditEnvironment
from navigation.alg_impl import GreedyNavigationBanditAlgorithm
from navigation.alg_impl import EpsilonGreedyNavigationBanditAlgorithm
from navigation.alg_impl import ThompsonSamplingNavigationBanditAlgorithm
from navigation.alg_impl import BayesUcbNavigationBanditAlgorithm


def main():
    parser = argparse.ArgumentParser(description='Run bandit navigation simulation')
    parser.add_argument('--charging-station-csv-path', help='charging_station_csv_path', type=str)
    parser.add_argument('--charging-graph-ids-csv-path', help='charging_graph_ids_csv_path', type=str)
    parser.add_argument('--charging-consumption-npy-path', help='charging_consumption_npy_path', type=str)
    parser.add_argument('--charging-time-npy-path', help='charging_time_npy_path', type=str)
    parser.add_argument('--start-node-id', help='start_node_id', type=str)
    parser.add_argument('--end-node-id', help='end_node_id', type=str)
    parser.add_argument('--output-dir', help='output_dir', type=str, default='../output/')
    parser.add_argument('--charge-prior-param-ln-p0', help='charge_prior_param_ln_p0', type=float, default=13.5)
    parser.add_argument('--charge-prior-param-q0', help='charge_prior_param_q0', type=float, default=300.)
    parser.add_argument('--charge-prior-param-r0', help='charge_prior_param_r0', type=float, default=3.)
    parser.add_argument('--charge-prior-param-s0', help='charge_prior_param_s0', type=float, default=3.)
    parser.add_argument('--queue-prior-param-alpha0', help='queue_prior_param_alpha0', type=float, default=2.)
    parser.add_argument('--queue-prior-param-beta0', help='queue_prior_param_beta0', type=float, default=2400.)
    parser.add_argument('--battery-capacity', help='battery_capacity', type=float, default=70.*1000.*3600.)
    parser.add_argument('--min-power-factor', help='min_power_factor', type=float, default=0.5)
    parser.add_argument('--power-scale', help='power_scale', type=float, default=300.)
    parser.add_argument('--number-of-experiment-runs', help='number_of_experiment_runs', type=int, default=5)
    parser.add_argument('--experiment-horizon', help='experiment_horizon', type=int, default=100)
    parser.add_argument('--start-seed', help='start_seed', type=int, default=0)
    parser.add_argument('--logging-level', help='logging_level', type=str, default='info')
    args, unknown_args = parser.parse_known_args()

    # Configure logging
    configure_logging(args)
    logging.info("Starting bandit navigation simulation!")

    # Configure priors
    queue_prior = QueuePrior(a=args.queue_prior_param_alpha0,
                             b=args.queue_prior_param_beta0)
    charge_prior = ChargePrior(ln_p=args.charge_prior_param_ln_p0,
                               q=args.charge_prior_param_q0,
                               r=args.charge_prior_param_r0,
                               s=args.charge_prior_param_s0)

    # Load Charging Graph DataFrames
    charging_station_df = pd.read_csv(args.charging_station_csv_path)
    charging_graph_ids_df = pd.read_csv(args.charging_graph_ids_csv_path)

    # Load pre-computed time and consumption matrices
    charging_graph_consumption_ndarray = np.load(args.charging_consumption_npy_path)
    charging_graph_time_ndarray = np.load(args.charging_time_npy_path)

    # Create experiments
    experiments = create_experiments(number_of_experiment_runs=args.number_of_experiment_runs,
                                     experiment_horizon=args.experiment_horizon,
                                     charging_station_df=charging_station_df,
                                     charging_graph_ids_df=charging_graph_ids_df,
                                     charging_graph_consumption_ndarray=charging_graph_consumption_ndarray,
                                     charging_graph_time_ndarray=charging_graph_time_ndarray,
                                     battery_capacity=args.battery_capacity,
                                     start_node=args.start_node_id,
                                     end_node=args.end_node_id,
                                     min_power_factor=args.min_power_factor,
                                     power_scale=args.power_scale,
                                     charge_prior=charge_prior,
                                     queue_prior=queue_prior,
                                     start_seed=args.start_seed)

    # Run experiments
    results = []
    for experiment in experiments:
        experiment_run = int(experiment.experiment_id / (len(experiments) / args.number_of_experiment_runs))
        logging.info("Starting experiment run: " + str(experiment_run) +
                     ", id: " + str(experiment.experiment_id) +
                     ", name: " + str(experiment.experiment_name) +
                     ", horizon: " + str(args.experiment_horizon))
        experiment_results = experiment.run_experiment()
        cumulative_regret = 0.
        for iteration_results in experiment_results:
            iteration_results['cumulative_regret'] = cumulative_regret + iteration_results['instant_regret']
            cumulative_regret = iteration_results['cumulative_regret']
        results = results + experiment_results
        experiment.teardown()
        logging.info("Finished experiment run: " + str(experiment_run) +
                     ", id: " + str(experiment.experiment_id) +
                     ", name: " + str(experiment.experiment_name) +
                     ", horizon: " + str(args.experiment_horizon))

    # Summarize results
    results_df = pd.DataFrame(data=results)
    logging.info("Finished bandit navigation simulation!")

    results_df.to_csv(args.output_dir + 'results.csv')
    logging.info("Saved results CSV to: " + args.output_dir + 'results.csv')

    fig = create_regret_plot(results_df)
    fig.savefig(args.output_dir + 'regret_plot.png', bbox_inches='tight', pad_inches=0, dpi=300)
    logging.info("Saved regret plot to: " + args.output_dir + 'regret_plot.png')


def create_experiments(number_of_experiment_runs,
                       experiment_horizon,
                       charging_station_df,
                       charging_graph_ids_df,
                       charging_graph_consumption_ndarray,
                       charging_graph_time_ndarray,
                       battery_capacity,
                       start_node,
                       end_node,
                       min_power_factor,
                       power_scale,
                       charge_prior,
                       queue_prior,
                       start_seed):
    experiments = []
    for experiment_run in range(number_of_experiment_runs):
        env_seed = start_seed + experiment_run
        agent_seed = start_seed + number_of_experiment_runs + experiment_run

        env_constructor = partial(NavigationBanditEnvironment,
                                  charging_station_df=charging_station_df,
                                  charging_graph_ids_df=charging_graph_ids_df,
                                  charging_graph_consumption_ndarray=charging_graph_consumption_ndarray,
                                  charging_graph_time_ndarray=charging_graph_time_ndarray,
                                  battery_capacity=battery_capacity,
                                  start_node=start_node,
                                  end_node=end_node,
                                  queue_prior=queue_prior,
                                  charge_prior=charge_prior,
                                  min_power_factor=min_power_factor,
                                  power_scale=power_scale)

        # Greedy navigation bandit experiment
        greedy_alg_constructor = partial(GreedyNavigationBanditAlgorithm,
                                         env_constructor=partial(env_constructor,
                                                                 rng=np.random.default_rng(agent_seed)),
                                         queue_prior=queue_prior,
                                         charge_prior=charge_prior,
                                         min_power_factor=min_power_factor,
                                         power_scale=power_scale,
                                         rng=np.random.default_rng(agent_seed))
        experiments.append(BanditExperiment(experiment_name='Greedy',
                                            experiment_id=experiment_run*4,
                                            experiment_horizon=experiment_horizon,
                                            bandit_algorithm_constructor=greedy_alg_constructor,
                                            bandit_environment_constructor=partial(env_constructor,
                                                                                   rng=np.random.default_rng(env_seed))))

        # Epsilon-Greedy navigation bandit experiment
        eps_greedy_alg_constructor = partial(EpsilonGreedyNavigationBanditAlgorithm,
                                             env_constructor=partial(env_constructor,
                                                                     rng=np.random.default_rng(agent_seed)),
                                             epsilon_function=lambda t: 1. / np.sqrt(t),
                                             queue_prior=queue_prior,
                                             charge_prior=charge_prior,
                                             min_power_factor=min_power_factor,
                                             power_scale=power_scale,
                                             rng=np.random.default_rng(agent_seed))
        experiments.append(BanditExperiment(experiment_name='Epsilon-Greedy',
                                            experiment_id=experiment_run*4 + 1,
                                            experiment_horizon=experiment_horizon,
                                            bandit_algorithm_constructor=eps_greedy_alg_constructor,
                                            bandit_environment_constructor=partial(env_constructor,
                                                                                   rng=np.random.default_rng(env_seed))))

        # Thompson Sampling navigation bandit experiment
        ts_alg_constructor = partial(ThompsonSamplingNavigationBanditAlgorithm,
                                     env_constructor=partial(env_constructor,
                                                             rng=np.random.default_rng(agent_seed)),
                                     number_of_cached_samples=100,
                                     queue_prior=queue_prior,
                                     charge_prior=charge_prior,
                                     min_power_factor=min_power_factor,
                                     power_scale=power_scale,
                                     rng=np.random.default_rng(agent_seed))
        experiments.append(BanditExperiment(experiment_name='Thompson Sampling',
                                            experiment_id=experiment_run*4 + 2,
                                            experiment_horizon=experiment_horizon,
                                            bandit_algorithm_constructor=ts_alg_constructor,
                                            bandit_environment_constructor=partial(env_constructor,
                                                                                   rng=np.random.default_rng(env_seed))))

        # BayesUCB navigation bandit experiment
        ucb_alg_constructor = partial(BayesUcbNavigationBanditAlgorithm,
                                      env_constructor=partial(env_constructor,
                                                              rng=np.random.default_rng(agent_seed)),
                                      queue_prior=queue_prior,
                                      charge_prior=charge_prior,
                                      min_power_factor=min_power_factor,
                                      power_scale=power_scale,
                                      rng=np.random.default_rng(agent_seed))
        experiments.append(BanditExperiment(experiment_name='BayesUCB',
                                            experiment_id=experiment_run*4 + 3,
                                            experiment_horizon=experiment_horizon,
                                            bandit_algorithm_constructor=ucb_alg_constructor,
                                            bandit_environment_constructor=partial(env_constructor,
                                                                                   rng=np.random.default_rng(env_seed))))
    return experiments


def create_regret_plot(results_df):
    results_df_copy = results_df.copy()
    results_df_copy['cumulative_regret_avg'] = results_df_copy['cumulative_regret']
    results_df_copy['cumulative_regret_std'] = results_df_copy['cumulative_regret']
    agg_df = results_df_copy.groupby(['experiment_name', 'iteration']).agg(
        {'cumulative_regret_avg': np.mean,
         'cumulative_regret_std': np.std}).reset_index()
    fig, ax = plt.subplots()
    for name, exp in agg_df.groupby('experiment_name'):
        line, = ax.plot(exp.iteration, exp.cumulative_regret_avg)
        line.set_label(name)
        ax.fill_between(exp.iteration,
                        exp.cumulative_regret_avg - exp.cumulative_regret_std,
                        exp.cumulative_regret_avg + exp.cumulative_regret_std,
                        alpha=0.4)
    ax.set(title="Cumulative Regret",
           xlabel="Time Step",
           ylabel="Regret")
    ax.legend()
    return fig


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
