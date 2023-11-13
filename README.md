# EENE Navigation Bandit Simulator

This repository contains code for replicating the results of the 
following [paper](https://openreview.net/forum?id=ndw90pkNM9):

`Niklas Åkerblom & Morteza Haghir Chehreghani (2023). A Combinatorial Semi-Bandit Approach to Charging Station 
Selection for Electric Vehicles. Transactions on Machine Learning Research (TMLR).`

## Prerequisites 

The simulation framework requires the user to provide a road network graph and a list of charging stations. Both need
to be provided as CSV files. Each row in the road network graph CSV file represents an edge in the road network graph,
in the following format:

| from_id | from_y | from_x | to_id | to_y | to_x | mean_consumption | mean_time |
|---------|--------|--------|-------|------|------|------------------|-----------|
| ...     | ...    | ...    | ...   | ...  | ...  | ...              | ...       |

The columns `from_id` and `to_id` contain the `string` source and target vertex (node) IDs of each edge, respectively. 
The columns `from_y` and `from_x` contain the `float` Y (latitude) and X (longitude) coordinates of the source vertex,
with `to_y` and `to_x` containing the corresponding coordinate for the target vertex. The columns `mean_consumption` 
and `mean_time` contain the `float` mean energy consumption (in `joules`) and travel time (in `seconds`) of the edge, 
respectively.

Each row in the charging station list CSV file represents a distinct charging station, in the following format:

| station_id | node_id | node_y | node_x | power |
|------------|---------|--------|--------|-------|
| ...        | ...     | ...    | ...    | ...   |

The column `station_id` should contain distinct `string` IDs for all charging stations. The columns `node_id`, `node_y`
and `node_x` should contain the `string` vertex ID and the `float` Y and `float` X coordinates of the closest road
graph vertex to each charging station. The `power` column should contain the `float` power (in `watts`) provided by
each charging station.

While these files are not provided in this repository, the public data sources used in the paper are 
[OpenStreetMap](https://www.openstreetmap.org/) and [Open Charge Map](https://openchargemap.org/) for the road graph
and charging station list, respectively.

## Graph Preprocessing

Before the simulation can be started, it is necessary to preprocess the road graph and charging station list, using
the script `charging_graph_preprocesser.py` in the `eene-nav-bandit-sim` folder, with the following command line 
arguments:

`python charging_graph_preprocesser.py --road-graph-csv-path [road graph CSV path] 
--charging-station-csv-path [charging station CSV path]`

This will produce three files, `charging_graph_ids.csv`, `complete_consumption_array.npy` and `complete_time_array.npy`,
which will be saved to the `output` folder (by default). The complete list of possible arguments to
`charging_graph_preprocesser.py` is given below:

| Argument                    |  Type  | Default Value |                                                                                Description |
|:----------------------------|:------:|:-------------:|-------------------------------------------------------------------------------------------:|
| --road-graph-csv-path       | String |      N/A      |                                                                        Road graph CSV path |
| --charging-station-csv-path | String |      N/A      |                                                                  Charging station CSV path |
| --output-dir                | String | "../output/"  |                                                                           Output directory |
| --logging-level             | String |    "info"     | Logging level (possible values: `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"`) |

**NOTE**: Depending on the size of the road network graph and the number of charging stations, the preprocessing script 
may take several hours to complete.

## Running Simulation

To start the simulation, run the Python file `sim_runner.py`  in the `eene-nav-bandit-sim` folder with the minimum 
required set of command line arguments as follows:

`python sim_runner.py --charging-station-csv-path [charging station CSV path] --start-node-id [start vertex ID]
--end-node-id [end vertex ID]`

Completing the simulation will produce a CSV file (`results.csv`) containing instant and cumulative regret per 
simulation iteration for each of the implemented exploration methods (`Greedy`, `Epsilon-Greedy`, `Thompson Sampling` 
and `BayesUCB`). In addition, a plot showing the cumulative regret per simulation iteration will be saved as
`regret_plot.png`. The default output folder is `output`. The complete list of possible arguments to `sim_runner.py`
is given below:

| Argument                        |  Type  |               Default Value                |                                                                                  Description |
|:--------------------------------|:------:|:------------------------------------------:|---------------------------------------------------------------------------------------------:|
| --charging-station-csv-path     | String |                    N/A                     |                                                                    Charging station CSV path |
| --start-node-id                 | String |                    N/A                     |                     Start vertex ID (**must** correspond to `node_id` of a charging station) |
| --end-node-id                   | String |                    N/A                     |                       End vertex ID (**must** correspond to `node_id` of a charging station) |
| --charging-graph-ids-csv-path   | String |     "../output/charging_graph_ids.csv"     |                              Path to output `charging_graph_ids.csv` of preprocessing script |
| --charging-consumption-npy-path | String | "../output/complete_consumption_array.npy" |                      Path to output `complete_consumption_array.npy` of preprocessing script |
| --charging-time-npy-path        | String |    "../output/complete_time_array.npy"     |                             Path to output `complete_time_array.npy` of preprocessing script |
| --output-dir                    | String |                "../output/"                |                                                                             Output directory |
| --number-of-experiment-runs     |  Int   |                     5                      |                                          Number of times each exploration method will be run |
| --experiment-horizon            |  Int   |                    100                     |                                           Number of simulation iterations in each experiment |
| --start-seed                    |  Int   |                     0                      |                   Start random seed of simulation (incremented by 1 for each experiment run) |
| --battery-capacity              | Float  |                252000000.0                 |                                         Battery capacity of simulation vehicle (in `joules`) |
| --min-power-factor              | Float  |                    0.5                     |            Minimum power provided by each charging station, as factor of the specified level |
| --power-scale                   | Float  |                   300.0                    |               Scaling factor for the power probability distribution of each charging station |
| --charge-prior-param-ln-p0      | Float  |                    13.5                    |               Charging power prior distribution parameter `ln_p0` (`log(pi_0)` in the paper) |
| --charge-prior-param-q0         | Float  |                   300.0                    |                    Charging power prior distribution parameter `q0` (`gamma_0` in the paper) |
| --charge-prior-param-r0         | Float  |                    3.0                     |                       Charging power prior distribution parameter `r0` (`xi_0` in the paper) |
| --charge-prior-param-s0         | Float  |                    3.0                     |                       Charging power prior distribution parameter `s0` (`xi_0` in the paper) |
| --queue-prior-param-alpha0      | Float  |                    2.0                     |                    Queue time prior distribution parameter `alpha0` (`alpha_0` in the paper) |
| --queue-prior-param-beta0       | Float  |                   2400.0                   |                      Queue time prior distribution parameter `beta0` (`beta_0` in the paper) |
| --logging-level                 | String |                   "info"                   |   Logging level (possible values: `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"`) |

**NOTE**: Charging power prior distribution parameters `r0` and `s0` are merged to a single parameter `xi_0` in the 
paper.

## License

The code is licensed under an Apache 2.0 license. The code is provided "as is".

## Acknowledgements

This work is funded by the Strategic Vehicle Research and Innovation Programme (FFI) of Sweden, 
through the project EENE (reference number: 2018-01937). 