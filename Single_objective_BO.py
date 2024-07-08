# Imports
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement
import matplotlib.pyplot as plt
import numpy
import socket
import pickle
# Main simulation function

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
        data = s.recv(4096)  # Increased buffer size for larger data
        received_output = pickle.loads(data)
        # print('Received output:', received_output)

        received_output_tensor = torch.tensor(received_output, dtype=torch.double)
        received_output_tensor = received_output_tensor.view(-1, 1)
        # print('Received output as tensor:', received_output_tensor)

        return received_output_tensor

lower_bounds = torch.tensor([340.0, 20.0])
upper_bounds = torch.tensor([360.0, 30.0])
bounds = torch.stack([lower_bounds, upper_bounds])

train_x = torch.rand(5, 2, dtype=torch.double)
train_x[:, 0] = lower_bounds[0] + (upper_bounds[0] - lower_bounds[0]) * train_x[:, 0]
train_x[:, 1] = lower_bounds[1] + (upper_bounds[1] - lower_bounds[1]) * train_x[:, 1]

# Generate initial data
train_obj = send_instruction(train_x)

# Function for fitting the model and getting the next sampling point in an iteration
def get_next_points(train_x, train_obj, bounds, n_points):
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acquisition_function = qExpectedImprovement(model=model, best_f=train_obj.max())

    candidates, _ = optimize_acqf(acq_function=acquisition_function, bounds=bounds, q=n_points,
                                  num_restarts=200, raw_samples=512, options={"batch_limit": 5, "maxiter": 200})

    return candidates

# Function to visualize the Gaussian Process regressor
def plot_gp_model(model, train_x, train_y, bounds, iteration):
    # Create a grid of points to evaluate the GP
    grid_size = 50
    x1 = torch.linspace(bounds[0, 0], bounds[1, 0], grid_size)
    x2 = torch.linspace(bounds[0, 1], bounds[1, 1], grid_size)
    X1, X2 = torch.meshgrid(x1, x2)
    X = torch.stack([X1.ravel(), X2.ravel()], dim=-1)

    # Normalize the grid points
    X_normalized = (X - bounds[0]) / (bounds[1] - bounds[0])

    # Get predictions from the GP model
    model.eval()
    with torch.no_grad():
        pred = model.posterior(X_normalized)
        mean = pred.mean.cpu().numpy().reshape(grid_size, grid_size)
        std = pred.variance.sqrt().cpu().numpy().reshape(grid_size, grid_size)

    # Plot the GP predictions
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].contourf(X1.numpy(), X2.numpy(), mean, levels=50, cmap='viridis')
    ax[0].scatter(train_x[:, 0].cpu(), train_x[:, 1].cpu(), c='red', label='Data Points')
    ax[0].set_title(f'GP Mean Prediction - Iteration {iteration}')
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].legend()

    ax[1].contourf(X1.numpy(), X2.numpy(), std, levels=50, cmap='viridis')
    ax[1].scatter(train_x[:, 0].cpu(), train_x[:, 1].cpu(), c='red', label='Data Points')
    ax[1].set_title(f'GP Standard Deviation - Iteration {iteration}')
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].legend()

    plt.show()

# Numbers of iterations in optimization
n_runs = 5

# Main optimization loop
for i in range(n_runs):
    print(f"Number of optimization run: {i}")
    new_candidates = get_next_points(train_x, train_obj, bounds, 1)
    new_results = send_instruction(new_candidates)
    print(f"New candidates are: {new_candidates}")
    train_x = torch.cat([train_x, new_candidates])
    train_obj = torch.cat([train_obj, new_results])
    print(f'Current Objectives: {train_obj}')
    print(f"Best Objective: {-train_obj.max()}")

    # Visualize the GP model after each iteration
    model = SingleTaskGP(train_x, train_obj)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    # plot_gp_model(model, train_x, train_obj, bounds, i)


