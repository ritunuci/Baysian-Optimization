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
import numpy as np
import socket

# Main simulation function
def simulation_model(parameters):
    result_tensor = torch.rand(parameters.size(0),1, dtype=torch.double)
    for i in range(parameters.size(0)):  # data_tensor.size(0) returns the size of the first dimension
        result_tensor[i] = parameters[i].sum()
    return result_tensor

bounds = torch.stack([torch.tensor([0., 0.]), torch.tensor([1.0, 1.0])])
train_x = torch.rand(10, 2, dtype=torch.double)
train_obj = simulation_model(train_x)

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

new_candidates = get_next_points(train_x, train_obj, bounds, 1)
print(new_candidates)
print(train_obj)
print(train_x)
# Numbers of iterations in optimization
n_runs = 5

# Main optimization loop
for i in range(n_runs):
    print(f"Number of optimization run: {i}")
    new_candidates = get_next_points(train_x, train_obj, bounds, 1)
    print(new_candidates, type(new_candidates))
    new_results = simulation_model(new_candidates)
    #
    print(f"New candidates are: {new_candidates}")
    train_x = torch.cat([train_x, new_candidates])
    train_obj = torch.cat([train_obj, new_results])

    print(f"Best point works this way: {train_obj.max()}")

    # Visualize the GP model after each iteration
    model = SingleTaskGP(train_x, train_obj)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    plot_gp_model(model, train_x, train_obj, bounds, i)

# ----------------------------------------------------------------------------------------------------------------------
# print(get_next_points(train_x, train_obj, bounds, 1))

# model = SingleTaskGP(train_x, train_obj)
# mll = ExactMarginalLogLikelihood(model.likelihood, model)
# fit_gpytorch_mll(mll)
# #
# acquisition_function = qExpectedImprovement(model = model, best_f=train_obj.max())
#
# candidates, _ = optimize_acqf(acq_function = acquisition_function, bounds = bounds, q = 1,
#                               num_restarts = 200, raw_samples = 512, options = {"batch_limit": 5, "maxiter": 200
# print(train_obj.max().items())
# print(train_x)
# print(train_obj)
# print(candidates)

# N_ITER = 5
# for _ in range(N_ITER):
#     candidate, acq_value = optimize_acqf(
#         acq_function=acquisition_function,
#         bounds=bounds,
#         q=1,
#         num_restarts=5,
#         raw_samples=20,
#     )
#     # candidate[:, 1] = candidate[:, 1].round()  # Maintaining the integer constraint
#     new_obj = simulation_model(candidate)
#     # new_obj = new_obj.view(-1, 1)  # Reshaping
#     train_x = torch.cat([train_x, candidate])
#     train_obj = torch.cat([train_obj, new_obj])
#
#     model.set_train_data(train_x, train_obj, strict=False)
#     fit_gpytorch_mll(mll)
#
# best_parameters = train_x[train_obj.argmax()]
# print("Best parameters found:", best_parameters)
#
# print(train_x)
# print(train_obj)

# print(bounds)
# ---------------------------------------------------------------------------------------------------------------------

# import torch
# from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_mll
# from gpytorch.mlls import ExactMarginalLogLikelihood
#
# # Double precision is highly recommended for GPs.
# # See https://github.com/pytorch/botorch/discussions/1444
# train_X = torch.rand(10, 2, dtype=torch.double)
# Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
# Y += 0.1 * torch.rand_like(Y)
# train_Y = (Y - Y.mean()) / Y.std()
#
# gp = SingleTaskGP(train_X, train_Y)
# mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
# fit_gpytorch_mll(mll)
#
# from botorch.acquisition import UpperConfidenceBound
#
# UCB = UpperConfidenceBound(gp, beta=0.1)
#
# from botorch.optim import optimize_acqf
#
# bounds = torch.stack([torch.zeros(2), torch.ones(2)])
# candidate, acq_value = optimize_acqf(
#     UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
# )