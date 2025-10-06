from colorama import Fore, Style
import pickle
import socket
import time
import csv
import os
import subprocess

from ax import optimize
from ax.core import ParameterType, RangeParameter, SearchSpace, MultiObjective, Objective, ObjectiveThreshold
from ax.core.experiment import Experiment
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)
from ax.runners.synthetic import SyntheticRunner
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from botorch.utils.sampling import draw_sobol_samples

import numpy
import pandas as pd
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)
from ax.core import ChoiceParameter
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume

# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.plot.contour import plot_contour
from ax.plot.diagnostic import tile_cross_validation
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.report_utils import exp_to_df

# Plotting imports and initialization
from ax.utils.notebook.plotting import init_notebook_plotting, render
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from botorch.utils.multi_objective.pareto import is_non_dominated
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.core.parameter_constraint import ParameterConstraint
from ax import SumConstraint


# ----------------------------------------------------------------------------------------------------------------------

def run_script():
    subprocess.run(["C:/Users/ritun_uci/.conda/envs/fleetpy_backup/python", "run_examples_new.py"])
    time.sleep(1)


def update_csv_with_train_x(values):
    a, b, c, d, e, h = values
    a1 = float(a)  # transit fare ($)
    b1 = float(b)  # Micro distance based fare ($/mile)
    c1 = float(c)  # Micro start fare ($)
    d1 = int(d)  # Fleet size
    e1 = float(e)  # Peak fare factor
    h1 = float(h)  # Micro to fixed factor
    input_csv_path = "D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/Input_parameter/input_parameter.csv"
    with open(input_csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Update the required values
    rows[1][5] = str(a1)  # "transit_fare ($)" in .csv file
    rows[1][6] = str(b1)  # "microtransit_dist_based_rate ($/mile)" in .csv file
    rows[1][7] = str(c1)  # "microtransit_start_fare ($)" in .csv file
    rows[1][8] = d1  # "Fleet_size" in .csv file

    rows[1][9] = e1  # "PkFareFactor" in .csv file
    rows[1][12] = h1  # "Micro2FixedFactor" in .csv file

    # Write the updated rows back to the CSV file
    with open(input_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    time.sleep(1)


def collect_output():
    output_csv_path = "D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/lemon_grove/output_folder" \
                      "/lemon_grove_evaluation_zonal_partition_False.csv"

    # output_csv_path = "D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/lemon_grove/output_folder" \
    #                   "/downtown_sd_evaluation_zonal_partition_False.csv"

    with open(output_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

        x1 = rows[3][30]  # "tt_sub" ($) in .csv file
        y1 = rows[3][50]  # "tt_mob_lgsm_inc_with_micro" in .csv file
        a1 = [float(val) for idx, val in enumerate(rows[3]) if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 50, 72]]
        value_11 = -float(x1)
        value_21 = float(y1)

    return [value_11, value_21], a1


# Function for Exchanging Input and Output with the Simulation (Fleetpy)
def send_instruction(train_x):
    train_x_array = train_x.numpy()
    all_outputs = []
    other_metrics = []
    for values in train_x_array:
        update_csv_with_train_x(values)
        run_script()
        output, other_metric = collect_output()
        all_outputs.append(output)
        other_metrics.append(other_metric)
    all_outputs_array = numpy.array(all_outputs)
    other_metrics_array = numpy.array(other_metrics)

    received_output_list = all_outputs_array.tolist()[0]
    received_other_metrics_list = other_metrics_array.tolist()[0]

    return received_output_list, received_other_metrics_list


## Test function for debugging purpose
# def send_instruction(train_x):
#     train_x_array = train_x.numpy()
#     received_output_list = [train_x_array[0].tolist()[0], train_x_array[0].tolist()[1]]
#     received_other_metrics_list = train_x_array[0].tolist()
#     # print(f"{Fore.RED}Done{Style.RESET_ALL}")
#     return received_output_list, received_other_metrics_list

# Parameters for Ax experiment
N_BATCH = 1  # Number of iterations in optimization loop   # used 30 before
BATCH_SIZE = 1  # Number of candidate points to be generated   # used 2 before
n_samples = 1  # Initial sobol samples for training             # used 15 before
# dimension = 3   # Number of input parameters
ref_point = [-30000, 0]  # TODO: If -50000 is used then in 'metric_a': 'lower_is_better=False' to be used
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

num_samples = 256
warmup_steps = 512
# ----------------------------------------------------------------------------------------------------------------------
# TODO: Search Space Definition for Discrete variables (i.e., the variables can take only certain values) Ax also
#  supports categorical variables

# param_transit_fare = ChoiceParameter(name="Transit Fare ($)", values=[i * 0.10 for i in range(10, 51)],
#                         parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True)
# param_micro_dist_fare = ChoiceParameter(name="Micro distance based fare ($/mile)", values=[i * 0.10 for i in range(51)],
#                         parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True)
# param_micro_start_fare = ChoiceParameter(name="Micro start fare ($)", values=[i * 0.10 for i in range(31)],
#                         parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True)
param_transit_fare = RangeParameter(name="Transit Fare ($)", lower=1.0, upper=5.0, parameter_type=ParameterType.FLOAT)
param_micro_dist_fare = RangeParameter(name="Micro distance based fare ($/mile)", lower=0.0, upper=5.0,
                                       parameter_type=ParameterType.FLOAT)
param_micro_start_fare = RangeParameter(name="Micro start fare ($)", lower=0.0, upper=3.0,
                                        parameter_type=ParameterType.FLOAT)
param_fleet_size = RangeParameter(name="Fleet size", lower=2.0, upper=7.0, parameter_type=ParameterType.INT)
param_peak_fare_factor = RangeParameter(name="Peak fare factor", lower=1.0, upper=4.0,
                                        parameter_type=ParameterType.FLOAT)
param_micro_fixed_factor = RangeParameter(name="Micro to fixed factor", lower=0.0, upper=1.0,
                                          parameter_type=ParameterType.FLOAT)

# Define the constraint
constraint = ParameterConstraint(
    constraint_dict={
        "Micro distance based fare ($/mile)": -1.0,
        "Micro start fare ($)": -1.0
    },
    bound=-0.5
)

# constraint = SumConstraint(parameters=[param_micro_dist_fare, param_micro_start_fare], is_upper_bound=False, bound=0.5)

search_space = SearchSpace(
    parameters=[param_transit_fare, param_micro_dist_fare, param_micro_start_fare, param_fleet_size,
                param_peak_fare_factor, param_micro_fixed_factor], parameter_constraints=[constraint]

)
# ----------------------------------------------------------------------------------------------------------------------
# search_space = SearchSpace(
#     parameters=[
#         ChoiceParameter(name="Transit Fare ($)", values=[i * 0.10 for i in range(10, 51)],
#                         parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True),
#         ChoiceParameter(name="Micro distance based fare ($/mile)", values=[i * 0.10 for i in range(51)],
#                         parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True),
#         ChoiceParameter(name="Micro start fare ($)", values=[i * 0.10 for i in range(5, 31)],
#                         parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True),
#         RangeParameter(name="Fleet size", lower=2.0, upper=7.0, parameter_type=ParameterType.INT),
#         RangeParameter(name="Peak fare factor", lower=1.0, upper=4.0, parameter_type=ParameterType.FLOAT),
#         RangeParameter(name="Micro to fixed factor", lower=0.0, upper=1.0, parameter_type=ParameterType.FLOAT),
#     ],
#
# )
# ----------------------------------------------------------------------------------------------------------------------
# Initialize lists to collect results
all_inputs = []
all_objectives = []
all_other_metrics = []

final_non_dominated_inputs = []
final_non_dominated_objectives = []
final_non_dominated_metrics = []

num_sobol_samples = n_samples
sobol_counter_f1 = 1
sobol_counter_f2 = 1
objective_cache = {}


def evaluate_objectives(x_sorted):
    start_time = time.time()
    input_data = torch.tensor(x_sorted).unsqueeze(0)
    print(f"{Fore.RED}send_instruction is being called for input {input_data} {Style.RESET_ALL}")
    _last_output, _last_other_metrics = send_instruction(input_data)
    end_time = time.time()
    duration = end_time - start_time
    log_path = "D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/optimization_output/sub_mob_logsum_28th_May_25_more_itr" \
               "/indiv_call_timings_eval.csv"
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["evaluation_time_(sec)"])
        writer.writerow([duration])

    return _last_output, _last_other_metrics


param_names = ["Transit Fare ($)", "Micro distance based fare ($/mile)", "Micro start fare ($)", "Fleet size",
               "Peak fare factor", "Micro to fixed factor"]


def f1(x) -> float:
    x_sorted = [x[p_name] for p_name in param_names]
    input_tuple = tuple(x_sorted)
    global sobol_counter_f1
    if input_tuple not in objective_cache:
        if sobol_counter_f1 <= num_sobol_samples:
            print(f"{Fore.GREEN}f1 evaluating for sobol input: {input_tuple}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}f1 evaluating for candidate input: {input_tuple}{Style.RESET_ALL}")

        output, other_metrics = evaluate_objectives(x_sorted)
        objective_cache[input_tuple] = (output, other_metrics)

        all_inputs.append(x_sorted)
        all_objectives.append(output)
        all_other_metrics.append(other_metrics)
        print(f"{Fore.GREEN}f1 objective output:{objective_cache[input_tuple][0][0]}{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.GREEN}f1 obj. retrieving for input: {input_tuple}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}f1 objective output from cache:{objective_cache[input_tuple][0][0]}{Style.RESET_ALL}\n")

        all_inputs.append(x_sorted)
        all_objectives.append(objective_cache[input_tuple][0])
        all_other_metrics.append(objective_cache[input_tuple][1])

    sobol_counter_f1 += 1
    return float(objective_cache[input_tuple][0][0])


def f2(x) -> float:
    x_sorted = [x[p_name] for p_name in param_names]
    input_tuple = tuple(x_sorted)
    global sobol_counter_f2
    if sobol_counter_f2 <= num_sobol_samples:
        print(f"{Fore.BLUE}f2 evaluating for sobol input: {input_tuple}{Style.RESET_ALL}")
    else:
        print(f"{Fore.BLUE}f2 evaluating for candidate input: {input_tuple}{Style.RESET_ALL}")
    sobol_counter_f2 += 1
    print(f"{Fore.BLUE}f2 objective output: {objective_cache[input_tuple][0][1]}{Style.RESET_ALL}\n")
    return float(objective_cache[input_tuple][0][1])


metric_a = GenericNoisyFunctionMetric(name="Total subsidy ($)", f=f1, noise_sd=0.0, lower_is_better=False)
metric_b = GenericNoisyFunctionMetric(name="Total mob. Logsum increase with micro", f=f2, noise_sd=0.0,
                                      lower_is_better=False)

# MultiObjective setup
mo = MultiObjective(
    objectives=[Objective(metric=metric_a), Objective(metric=metric_b)],
)

objective_thresholds = [
    ObjectiveThreshold(metric=metric_a, bound=ref_point[0], relative=False),
    ObjectiveThreshold(metric=metric_b, bound=ref_point[1], relative=False),
]

optimization_config = MultiObjectiveOptimizationConfig(
    objective=mo,
    objective_thresholds=objective_thresholds,
)


# Build and initialize experiment
def build_experiment():
    experiment = Experiment(
        name="multi_obj_exp",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment


training_start_time = time.time()
experiment = build_experiment()


def initialize_experiment(experiment):
    sobol = Models.SOBOL(search_space=experiment.search_space)
    experiment.new_batch_trial(sobol.gen(n_samples)).run()
    return experiment.fetch_data()


data = initialize_experiment(experiment)

# Define the surrogate model outside the loop
# surrogate = Surrogate(
#     botorch_model_class=SaasFullyBayesianSingleTaskGP,
#     mll_options={
#         "num_samples": num_samples,  # Increasing this may result in better model fits
#         "warmup_steps": warmup_steps,  # Increasing this may result in better model fits
#     },
# )
training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"\n{Fore.GREEN}Training duration is {training_duration:.2f} seconds")

optimization_start_time = time.time()
# Run optimization using qNEHVI acquisition function
hv_list = []
for i in range(N_BATCH):
    if i == 0:
        print(f"\n\n{Fore.RED}Sobol Samples End Here{Style.RESET_ALL}")
        print(f"{Fore.RED}Optimization Loop Starts Here{Style.RESET_ALL}\n\n")

    model = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=data,
    )

    # Generate new candidates
    generator_run = model.gen(BATCH_SIZE)

    # Create and run the batch trial
    trial = experiment.new_batch_trial(generator_run=generator_run)
    trial.run()

    # Fetch data and update
    data = Data.from_multiple_data([data, trial.fetch_data()])

    # Extract input data and objectives from the trial
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[["Total subsidy ($)", "Total mob. Logsum increase with micro"]].values, **tkwargs)

    # Collect input parameters, objective values, and other metrics
    inputs = generator_run.arms  # Input parameter sets
    objectives = exp_df[["Total subsidy ($)", "Total mob. Logsum increase with micro"]].values
    # ----------------------------------------------------------------------------------------------------------------------
    # Compute hypervolume
    partitioning = DominatedPartitioning(ref_point=torch.tensor(ref_point, **tkwargs), Y=outcomes)
    try:
        hv = partitioning.compute_hypervolume().item()
    except:
        hv = 0
        print("Failed to compute hypervolume")
    hv_list.append(hv)
    print(f"{Fore.MAGENTA}Iteration {i}, Hypervolume: {hv}{Style.RESET_ALL}")

optimization_end_time = time.time()
optimization_duration = optimization_end_time - optimization_start_time
print(f"\n{Fore.GREEN}Optimization duration is {optimization_duration:.2f} seconds")
# ----------------------------------------------------------------------------------------------------------------------
# Save inputs and objectives as per new method
df = exp_to_df(experiment).sort_values(by=["trial_index"])
df.to_csv("D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/optimization_output"
          "/sub_mob_logsum_28th_May_25_more_itr/optimization_result_new_method.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# Non Dominant Data Collection (Work on this part)
all_objectives_tensor = torch.tensor(all_objectives)
all_inputs_tensor = torch.tensor(all_inputs)
all_other_metrics_tensor = torch.tensor(all_other_metrics)

sorted_indices = all_objectives_tensor[:, 0].sort()[1]

all_objectives_tensor_sorted = all_objectives_tensor[sorted_indices]
all_inputs_sorted = all_inputs_tensor[sorted_indices]
all_other_metrics_sorted = all_other_metrics_tensor[sorted_indices]

non_dominated_mask = is_non_dominated(all_objectives_tensor_sorted)

final_non_dominated_input_tensor = all_inputs_sorted[non_dominated_mask]
final_non_dominated_objective_tensor = all_objectives_tensor_sorted[non_dominated_mask]
final_non_dominated_other_metrics_tensor = all_other_metrics_sorted[non_dominated_mask]

df_non_dom_inputs = pd.DataFrame(final_non_dominated_input_tensor.numpy(),
                                 columns=["Transit Fare ($)", "Micro distance based fare ($/mile)",
                                          "Micro start fare ($)", "Fleet size", "Peak fare factor",
                                          "Micro to fixed factor"])
df_non_dom_objectives = pd.DataFrame(final_non_dominated_objective_tensor.numpy(),
                                     columns=["Total subsidy ($)", "Total mob. Logsum increase with micro"])
df_non_dom_metrics = pd.DataFrame(final_non_dominated_other_metrics_tensor.numpy(),
                                  columns=['car_users', 'car_mode_share (%)',
                                           'trsit_mode_users (W_M_F)',
                                           'transit_mode_share (%)',
                                           'pure_M_users', 'M_mode_share (%)',
                                           'M_trips', 'pure_F_users',
                                           'F_mode_share (%)', 'F_trips',
                                           'pure_walk_users',
                                           'W_mode_share (%)', 'walk_users',
                                           'M_pls_F_users',
                                           'M_pls_F_mode_share (%)',
                                           'F_oper_cost ($)',
                                           'M_oper_cost ($)', 'F_revenue ($)',
                                           'M_revenue ($)', 'Total_T_revenue ($)',
                                           'sub_per_F_trip ($)',
                                           'sub_per_F_rider ($)',
                                           'sub_per_M_trip ($)',
                                           'sub_per_M_rider ($)',
                                           'sub_per_T_trip ($)',
                                           'sub_per_T_rider ($)',
                                           'sub_per_M_pax_mile ($/mi)',
                                           'sub_per_F_pax_mile ($/mi)',
                                           'sub_per_T_pax_mile ($/mi)',
                                           'sub_per_M_VMT ($/mi)',
                                           'sub_per_F_VMT ($/mi)',
                                           'sub_per_T_VMT ($/mi)',
                                           'tt_auto_gas_cost ($)',
                                           'auto_gas_cost_per_mile ($/mi)',
                                           'avg_M_fare', 'avg_F_fare',
                                           'avg_T_fare',
                                           'avg_auto_gas_cost',
                                           'tt_o_pckt_cost ($)',
                                           'tt_gen_cost', 'tt_mode_switch',
                                           'M_avg_wait_time (s)',
                                           'F_avg_wait_time (s)',
                                           'avg_walk_time (s)',
                                           'tt_walk_time (h)', 'car_VMT (mi)',
                                           'M_VMT (mi)',
                                           'M_PMT (mi)', 'M_PMT/M_VMT',
                                           'F_VMT (mi)', 'F_PMT (mi)',
                                           'F_PMT/F_VMT', 'tt_VMT (mi)',
                                           'tt_walk_dist (mi)',
                                           'wghted_acc_emp_5_min',
                                           'wghted_acc_emp_10_min',
                                           'wghted_acc_emp_15_min',
                                           'M_util_rate (%)', 'M_veh_occ',
                                           'M_avg_speed (mph)'])

df_total_non_dom = pd.concat([df_non_dom_inputs, df_non_dom_objectives, df_non_dom_metrics], axis=1)
df_total_non_dom.to_csv("D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/optimization_output/"
                        "sub_mob_logsum_28th_May_25_more_itr/non_dominated_data_old_method.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# Work on this part
# Save the results to a CSV file
final_all_inputs = numpy.array(all_inputs)
final_all_objectives = numpy.array(all_objectives)
final_all_other_metrics = numpy.array(all_other_metrics)

df_inputs = pd.DataFrame(all_inputs,
                         columns=["Transit Fare ($)", "Micro distance based fare ($/mile)", "Micro start fare ($)",
                                  "Fleet size", "Peak fare factor", "Micro to fixed factor"])
df_objectives = pd.DataFrame(all_objectives, columns=["Total subsidy ($)", "Total mob. Logsum increase with micro"])
df_other_metrics = pd.DataFrame(all_other_metrics, columns=['car_users', 'car_mode_share (%)',
                                                            'trsit_mode_users (W_M_F)',
                                                            'transit_mode_share (%)',
                                                            'pure_M_users', 'M_mode_share (%)',
                                                            'M_trips', 'pure_F_users',
                                                            'F_mode_share (%)', 'F_trips',
                                                            'pure_walk_users',
                                                            'W_mode_share (%)', 'walk_users',
                                                            'M_pls_F_users',
                                                            'M_pls_F_mode_share (%)',
                                                            'F_oper_cost ($)',
                                                            'M_oper_cost ($)', 'F_revenue ($)',
                                                            'M_revenue ($)', 'Total_T_revenue ($)',
                                                            'sub_per_F_trip ($)',
                                                            'sub_per_F_rider ($)',
                                                            'sub_per_M_trip ($)',
                                                            'sub_per_M_rider ($)',
                                                            'sub_per_T_trip ($)',
                                                            'sub_per_T_rider ($)',
                                                            'sub_per_M_pax_mile ($/mi)',
                                                            'sub_per_F_pax_mile ($/mi)',
                                                            'sub_per_T_pax_mile ($/mi)',
                                                            'sub_per_M_VMT ($/mi)',
                                                            'sub_per_F_VMT ($/mi)',
                                                            'sub_per_T_VMT ($/mi)',
                                                            'tt_auto_gas_cost ($)',
                                                            'auto_gas_cost_per_mile ($/mi)',
                                                            'avg_M_fare', 'avg_F_fare',
                                                            'avg_T_fare',
                                                            'avg_auto_gas_cost',
                                                            'tt_o_pckt_cost ($)',
                                                            'tt_gen_cost', 'tt_mode_switch',
                                                            'M_avg_wait_time (s)',
                                                            'F_avg_wait_time (s)',
                                                            'avg_walk_time (s)',
                                                            'tt_walk_time (h)', 'car_VMT (mi)',
                                                            'M_VMT (mi)',
                                                            'M_PMT (mi)', 'M_PMT/M_VMT',
                                                            'F_VMT (mi)', 'F_PMT (mi)',
                                                            'F_PMT/F_VMT', 'tt_VMT (mi)',
                                                            'tt_walk_dist (mi)',
                                                            'wghted_acc_emp_5_min',
                                                            'wghted_acc_emp_10_min',
                                                            'wghted_acc_emp_15_min',
                                                            'M_util_rate (%)', 'M_veh_occ',
                                                            'M_avg_speed (mph)'])

# Concatenate all data and save to output file
df_total = pd.concat([df_inputs, df_objectives, df_other_metrics], axis=1)
df_total.to_csv("D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/optimization_output/"
                "sub_mob_logsum_28th_May_25_more_itr/result_old_method.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Plot 1: All Objective Points ('total_points.png')
######################################################
batch_number = torch.cat(
    [
        torch.zeros(n_samples),
        torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1),
    ]
).numpy()

fig1, ax1 = plt.subplots(figsize=(12, 8))
sc1 = ax1.scatter(final_all_objectives[:, 0],
                  final_all_objectives[:, 1],
                  c=batch_number,
                  alpha=0.8, )
ax1.set_xlabel("Total Subsidy ($)", fontsize=14)
ax1.set_ylabel("Total Mobility Logsum Increase with Microtransit", fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
norm1 = plt.Normalize(batch_number.min(), batch_number.max())
sm1 = ScalarMappable(norm=norm1, cmap='viridis')
sm1.set_array([])

cbar_ax1 = fig1.add_axes([0.96, 0.15, 0.01, 0.7])
cbar1 = fig1.colorbar(sm1, cax=cbar_ax1)
cbar1.ax.set_title("Iteration")
fig1.savefig('D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/optimization_output'
             '/sub_mob_logsum_28th_May_25_more_itr/total_points.png', format='png', dpi=300)
plt.show()

# Non-Dominant Points ('non_dominant_points.png')
######################################################
non_dominated_batch_numbers = batch_number[sorted_indices][non_dominated_mask]
fig2, ax2 = plt.subplots(figsize=(12, 8))

sc2 = ax2.scatter(final_non_dominated_objective_tensor[:, 0].cpu().numpy(),
                  final_non_dominated_objective_tensor[:, 1].cpu().numpy(),
                  c=non_dominated_batch_numbers,
                  alpha=0.8, )
ax2.set_xlabel("Total Subsidy ($)", fontsize=14)
ax2.set_ylabel("Total Mobility Logsum Increase with Microtransit", fontsize=14)
ax2.tick_params(axis='both', labelsize=14)
norm2 = plt.Normalize(non_dominated_batch_numbers.min(), non_dominated_batch_numbers.max())
sm2 = ScalarMappable(norm=norm2, cmap='viridis')
sm2.set_array([])
cbar_ax2 = fig2.add_axes([0.95, 0.15, 0.01, 0.7])
cbar2 = fig2.colorbar(sm2, cax=cbar_ax2)
cbar2.ax.set_title("Iteration")
cbar2.set_ticks(numpy.arange(non_dominated_batch_numbers.min(), non_dominated_batch_numbers.max() + 1, 1))
fig2.savefig('D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/optimization_output'
             '/sub_mob_logsum_28th_May_25_more_itr/non_dominant_points.png', format='png', dpi=300)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# HYPERVOLUME PLOTTING
# Plot the observed hypervolumes over iterations
iterations = numpy.arange(1, N_BATCH + 1)
fig3, ax3 = plt.subplots()
ax3.plot(iterations, hv_list, color='blue')
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Hypervolume")
fig3.savefig('D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/optimization_output'
             '/sub_mob_logsum_28th_May_25_more_itr/hypervolume_plot.png', format='png', dpi=300)
plt.show()
print(f"Hypervolumes: {hv_list}")
print(f"Non-dom batch numbers: {non_dominated_batch_numbers}")
# print(f"Inputs\n", all_inputs)
# print(f"Outputs\n",all_objectives)
# print(f"Other\n",all_other_metrics)
# ----------------------------------------------------------------------------------------------------------------------
# TODO: Pareto plotting new method
# frontier = compute_posterior_pareto_frontier(
#     experiment=experiment,
#     data=experiment.fetch_data(),
#     primary_objective=metric_a,
#     secondary_objective=metric_b,
#     absolute_metrics=["Total subsidy ($)", "Total mob. Logsum increase with micro"],
#     num_points=20,
# )
# pareto_plot = plot_pareto_frontier(frontier, CI_level=0.90)
# # Extract the Matplotlib figure from the AxPlotConfig object
# fig = pareto_plot.plot
# # Display the plot
# # plt.show()
# fig.savefig("D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/optimization_output/"
#                            "try/pareto_frontier_plot_new.png", dpi=300)
#
# # Extract Pareto frontier data
# pareto_observations = pareto_plot.data[0]['observations']
# # Create a DataFrame
# pareto_df = pd.DataFrame(pareto_observations)
# # Add parameter values to the DataFrame
# for i, arm in enumerate(frontier.pareto_optimal_points):
#     for param_name, param_value in arm.parameters.items():
#         pareto_df.loc[i, param_name] = param_value
# # Save the DataFrame to a CSV file
# pareto_df.to_csv("D:/Ritun/Siwei_Micro_Transit/Bayesian_Optimization/optimization_output/"
#                  "try/pareto_frontier_data_new.csv", index=False)
