import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

colors = ["#471365", "#24878E", "#F7E621"]  # dark green to light yellow
color_map = LinearSegmentedColormap.from_list("custom_cmap", colors)

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
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, alpha=1.0, fill=False, color=color, linewidth=4.0)

    # Plot the ellipse
    ax.add_patch(ellipse)
    plt.scatter(*mean, c=color)  # Plot the mean

def plot_2d_gaussians(means, chols, ax, title: str = '2D Gaussian', color: str = 'green'):
    for i in range(means.shape[0]):
        plot_gaussian_ellipse(means[i], chols[i] @ chols[i].T, ax, color)
    # plt.title(title)

def plot_2d_gaussians_color_map(means, chols, ax,
                                x_range, y_range,
                                title: str = '2D Gaussian',
                                color: str = 'green'):
    cov = chols @ chols.transpose(-1, -2)
    n_components = means.shape[0]
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.means_ = means
    gmm.covariances_ = cov

    gmm.weights_ = np.ones(n_components) / n_components
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))

    # Step 4: Create a grid of points
    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.linspace(y_range[0], y_range[1], 500)
    X_, Y_ = np.meshgrid(x, y)
    XX = np.array([X_.ravel(), Y_.ravel()]).T

    # Step 5: Compute the GMM density
    log_density = gmm.score_samples(XX)
    density = np.exp(log_density)
    # Z = log_density.reshape(X_.shape)  # Use log density
    Z = density.reshape(X_.shape)  # Use density

    ax.contourf(X_, Y_, Z, levels=10, cmap=color_map)


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

