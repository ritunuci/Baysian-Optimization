import numpy
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.sampling import SobolQMCNormalSampler
import socket
import pickle
import random
from pyDOE import lhs
from matplotlib import pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated
from matplotlib.cm import ScalarMappable

# Function for Exchanging Input and Output with the Simulation (Fleetpy)
def send_instruction(train_x):
    host = '127.0.0.1'  # The server's hostname or IP address
    # port = 65432        # The port used by the server
    port = 52097  # The port used by the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        train_x_array = train_x.numpy()
        # Serialize train_x
        serialized_train_x = pickle.dumps(train_x_array)

        # Send serialized data to the server
        s.sendall(serialized_train_x)

        # Receive response from the server
        data = s.recv(52097)  # Increased buffer size for larger data
        received_output = pickle.loads(data)
        # print('Received output:', received_output)

        received_output_tensor = torch.tensor(received_output, dtype=torch.double)
        # received_output_tensor = received_output_tensor.view(-1, 1)
        # print('Received output as tensor:', received_output_tensor)

        return received_output_tensor
# ----------------------------------------------------------------------------------------------------------------------
# Definition of Bounds and Initial Data Generation for Model Training
lower_bounds = torch.tensor([340.0, 25.0])
upper_bounds = torch.tensor([345.0, 27.0])
bounds = torch.stack([lower_bounds, upper_bounds])

# Initial data generation
def generate_initial_data(n_samples):
    train_x = torch.rand(n_samples, 2, dtype=torch.double)
    train_x[:, 0] = lower_bounds[0] + (upper_bounds[0] - lower_bounds[0]) * train_x[:, 0]
    train_x[:, 1] = lower_bounds[1] + (upper_bounds[1] - lower_bounds[1]) * train_x[:, 1]

    train_y = send_instruction(train_x)
    train_y = train_y.to(dtype=torch.double)
    return train_x, train_y

# NUM_RESTARTS = 10
# RAW_SAMPLES = 1024
standard_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
# MC_SAMPLES = 256
# TODO: added 7/6/2024
MC_SAMPLES = 512
NUM_RESTARTS = 20
RAW_SAMPLES = 2048
# ----------------------------------------------------------------------------------------------------------------------
# Model Initialization Function
def initialize_model(train_x, train_y):
    # train_x = normalize(train_x, bounds)
    models = []
    for i in range(train_y.shape[-1]):
        train_objective = train_y[:, i]
        models.append(
            SingleTaskGP(train_x, train_objective.unsqueeze(-1))
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model
# ----------------------------------------------------------------------------------------------------------------------
# Candiadate Generation Function
def generate_next_candidate(x, y, n_candidates=1):
    train_x = normalize(x, bounds)
    mll, model = initialize_model(train_x, y)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # train_x = normalize(x, bounds)
    with torch.no_grad():
        pred = model.posterior(train_x).mean

    acq_fun_list = []
    for _ in range(n_candidates):
        weights = sample_simplex(2).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(
                weights,
                pred
            )
        )
        acq_fun = qNoisyExpectedImprovement(
            model=model,
            objective=objective,
            sampler=sampler,
            X_baseline=train_x,
            prune_baseline=True,
        )
        acq_fun_list.append(acq_fun)

    candidates, acq_values = optimize_acqf_list(
        acq_function_list=acq_fun_list,
        bounds=standard_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={
            "batch_limit": 5,
            "maxiter": 200,
        }
    )

    candidates = unnormalize(candidates, bounds)
    return candidates, acq_values
# ----------------------------------------------------------------------------------------------------------------------
# Candidate Plotting Functions

# def plot_candidates(x, y):
#     """
#     Plot the candidates along with their objective values in a 3D space.
#     """
#     # Ensure x and y are numpy arrays
#     x_np = x.numpy()
#     y_np = y.numpy()
#
#     fig = plt.figure(figsize=(14, 6))
#
#     # Plot the first objective in 3D space
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.scatter(x_np[:, 0], x_np[:, 1], y_np[:, 0], c='blue', label='Objective 1')
#     ax1.set_xlabel('Variable 1')
#     ax1.set_ylabel('Variable 2')
#     ax1.set_zlabel('Objective 1')
#     ax1.set_title('Objective 1 in 3D Space')
#     ax1.grid(True)
#     ax1.legend()
#
#     # Plot the second objective in 3D space
#     ax2 = fig.add_subplot(122, projection='3d')
#     ax2.scatter(x_np[:, 0], x_np[:, 1], y_np[:, 1], c='orange', label='Objective 2')
#     ax2.set_xlabel('Variable 1')
#     ax2.set_ylabel('Variable 2')
#     ax2.set_zlabel('Objective 2')
#     ax2.set_title('Objective 2 in 3D Space')
#     ax2.grid(True)
#     ax2.legend()
#
#     plt.tight_layout()
#     plt.show()
# --------------------------------------------------------
def plot_candidates(x, y):
    """
    Plot the candidates along with their objective values.
    """
    # Ensure x and y are numpy arrays
    x_np = x.numpy()
    y_np = y.numpy()

    plt.figure(figsize=(10, 5))

    # Plot the input design points
    plt.subplot(1, 2, 1)
    plt.scatter(x_np[:, 0], x_np[:, 1], c='blue', label='Design Points')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title('Design Points')
    plt.grid(True)
    plt.legend()

    # Plot the objective values
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(y_np)), y_np[:, 0], c='blue', label='Objective 1')
    plt.scatter(range(len(y_np)), y_np[:, 1], c='orange', label='Objective 2')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Values')
    plt.title('Objective Values Over Iterations')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# # TODO: added 7/6/2024
# def plot_pareto_front(x, y):
#     fig, axes = plt.subplots(1, 1, figsize=(7, 7))
#     algos = ["qNEHVI"]
#     cm = plt.cm.get_cmap('viridis')
#     batch_number = torch.cat(
#         [torch.zeros(n_start), torch.arange(1, n_iter + 1).repeat(n_samples, 1).t().reshape(-1)]
#     ).numpy()
#
#     sc = axes.scatter(y[:, 0], y[:, 1], c=batch_number, alpha=0.8)
#     y_sorted = y[y[:, 0].sort()[1]]
#
#     axes.plot(
#         [_y[0] for non_dominated, _y in zip(is_non_dominated(-y_sorted), y_sorted) if non_dominated],
#         [_y[1] for non_dominated, _y in zip(is_non_dominated(-y_sorted), y_sorted) if non_dominated],
#         label="Pareto Front",
#         c="r"
#     )
#
#     axes.set_xlabel("Objective 1")
#     axes.set_ylabel("Objective 2")
#     norm = plt.Normalize(batch_number.min(), batch_number.max())
#     sm = ScalarMappable(norm=norm, cmap=cm)
#
#     sm.set_array([])
#     fig.subplots_adjust(right=0.9)
#     cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
#     cbar = fig.colorbar(sm, cax=cbar_ax)
#     cbar.ax.set_title("Iteration")
#
#     axes.legend()
#     plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Main Optimization Loop
n_iter = 2
n_start = 2   # Number of initial point to evaluate for training
n_samples = 1
x, y = generate_initial_data(n_start)

for i in range(n_iter):
    print(f"Iteration {i}")
    candidate, _ = generate_next_candidate(x, y, n_candidates=n_samples)
    print(f"Candidates: {candidate}")
    x = torch.cat([x, candidate])
    y = torch.cat([y, send_instruction(candidate)], dim=0)
    # if i % 10 == 0:  # Plot every 10 iterations
    #     plot_pareto_front(x, y)                             # TODO: added 7/6/2024
    # plot_candidates(torch.cat([x,y], dim=1))
# plot_candidates(x, y)
# ----------------------------------------------------------------------------------------------------------------------
# Plotting the Pareto Front
fig, axes = plt.subplots(1, 1, figsize=(7, 7))
algos = ["qNEHVI"]
cm = plt.cm.get_cmap('viridis')

batch_number = torch.cat(
    [torch.zeros(n_start), torch.arange(1, n_iter+1).repeat(n_samples, 1).t().reshape(-1)]
).numpy()

sc = axes.scatter(y[:, 0], y[:, 1], c=batch_number, alpha=0.8)

y_sorted = y[y[:, 0].sort()[1]]

axes.plot(
    [_y[0] for non_dominated, _y in zip(is_non_dominated(-y_sorted), y_sorted) if non_dominated],
    [_y[1] for non_dominated, _y in zip(is_non_dominated(-y_sorted), y_sorted) if non_dominated],
    label="Pareto Front",
    c="r"
)

axes.set_xlabel("Objective 1")
axes.set_ylabel("Objective 2")
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm = ScalarMappable(norm=norm, cmap=cm)

sm.set_array([])
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")

axes.legend()
plt.show()
# ----------------------------------------------------------------------------------------------------------------------

