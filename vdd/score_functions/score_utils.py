import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import torch

def plot_gaussian_ellipse(mean, covariance, ax, color):
    """
    Plot the mean and covariance of a Gaussian distribution as an ellipse.

    Parameters:
    mean (np.array): Mean vector of the Gaussian distribution.
    covariance (np.array): Covariance matrix of the Gaussian distribution.
    """
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Compute the angle between the x-axis and the largest eigenvector
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Create an ellipse representing the covariance matrix
    width, height = 2 * np.sqrt(eigenvalues)  # 2 standard deviations
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, alpha=1.0, fill=False, color=color)

    # Plot the ellipse
    ax.add_patch(ellipse)
    plt.scatter(*mean, c=color)  # Plot the mean

def plot_2d_gaussians(means, chols, ax, title: str = '2D Gaussian', color: str = 'green'):
    for i in range(means.shape[0]):
        plot_gaussian_ellipse(means[i], chols[i] @ chols[i].T, ax, color)
    # plt.title(title)

def distribute_components_torch(n):
    # Calculate grid size
    grid_side = int(torch.ceil(torch.sqrt(torch.tensor(n).float())))  # Number of points along one dimension

    # Generate grid points
    linspace = torch.linspace(-0.5, 0.5, grid_side)
    grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing='ij')

    # Flatten the grid and take the first n points
    points_x = grid_x.flatten()[:n]
    points_y = grid_y.flatten()[:n]

    return points_x, points_y


def plot_distribution_torch(n):
    x, y = distribute_components_torch(n)
    plt.scatter(x.numpy(), y.numpy())  # Convert to numpy for plotting
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Uniform distribution of {n} components in PyTorch')
    plt.show()

