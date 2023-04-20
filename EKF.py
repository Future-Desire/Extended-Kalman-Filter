import os
import random

import matplotlib.pyplot as plt
import numpy as np

from Renderer import Renderer


def seed_everything(seed=42):
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return None


class EKF(object):
    """
    A class for implementing EKFs.

    Attributes
    ----------
    mu : numpy.ndarray
        The N-element mean vector
    Sigma : numpy.ndarray
        The N x N covariance matrix
    R : numpy.ndarray
        The 3 x 3 covariance matrix for the additive Gaussian noise
        corresponding to motion uncertainty.
    Q : numpy.ndarray
        The 2 x 2 covariance matrix for the additive Gaussian noise
        corresponding to the measurement uncertainty.
    XYT : numpy.ndarray
        An N x T array of ground-truth poses, where each column corresponds
        to the ground-truth pose at that time step.
    MU : numpy.ndarray
        An N x (T + 1) array, where each column corresponds to mean of the
        posterior distribution at that time step. Note that there are T + 1
        entries because the first entry is the mean at time 0.
    VAR : numpy.ndarray
        An N x (T + 1) array, where each column corresponds to the variances
        of the posterior distribution (i.e., the diagonal of the covariance
        matrix) at that time step. As with MU, the first entry in VAR is
        the variance of the prior (i.e., at time 0)
    renderer : Renderer
        An instance of the Renderer class

    Methods
    -------
    getVariances()
        Return the diagonal of the covariance matrix.
    prediction (u)
        Perform the EKF prediction step with control u.
    update(z)
        Perform the EKF update step with measurement z.
    """

    def __init__(self, mu, Sigma, R, Q, XYT):
        """
        Initialize the class.

        Attributes
        ----------
        mu : numpy.ndarray
            The initial N-element mean vector for the distribution
        Sigma : numpy.ndarray
            The initial N x N covariance matrix for the distribution
        R : numpy.ndarray
            The N x N covariance matrix for the additive Gaussian noise
            corresponding to motion uncertainty.
        Q : numpy.ndarray
            The M x M covariance matrix for the additive Gaussian noise
            corresponding to the measurement uncertainty.
        XYT : numpy.ndarray
            An N x T array of ground-truth poses, where each column corresponds
            to the ground-truth pose at that time step.
        """

        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q
        self.XYT = XYT

        # Initialize dimension variables
        self.dim_N = 2
        self.dim_M = 2

        # Keep track of mean and variance over time
        self.MU = mu
        self.VAR = np.diag(self.Sigma).reshape(3, 1)

        # Initialize renderer
        xLim = np.array((np.amin(XYT[0, :] - 2), np.amax(XYT[0, :] + 2)))
        yLim = np.array((np.amin(XYT[1, :] - 2), np.amax(XYT[1, :] + 2)))
        self.renderer = Renderer(xLim, yLim, 3, "red", "green")

    def _motion_model(self, x, u, v_t, add_noise=True):
        """Computes the motion model f(x) given the mean state estimate (mu) and the motion noise v_t."""
        x_t = x[0] + (u[0] + int(add_noise) * v_t[0]) * np.cos(x[2])
        y_t = x[1] + (u[0] + int(add_noise) * v_t[0]) * np.sin(x[2])
        theta_t = x[2] + u[1] + int(add_noise) * v_t[1]

        return np.array([x_t, y_t, theta_t])

    def _measurement_model(self, x, w_t, add_noise=True):
        """Computes the measurement model h(x) given the mean state estimate (mu) and the measurement noise w_t."""
        z_r_t = x[0] ** 2 + x[1] ** 2 + int(add_noise) * w_t[0]
        z_theta_t = np.arctan2(x[1], x[0]) + int(add_noise) * w_t[1]

        return np.array([z_r_t, z_theta_t])

    def _F_jacobian(self, u, v_t):
        """Computes the Jacobian F as partial f/x evaluated at the mean state estimate (mu) given the control and motion noise v_t."""
        row_1 = ([1, 0, -1 * (u[0] + v_t[0]) * np.sin(self.mu[2])],)
        row_2 = ([0, 1, (u[0] + v_t[0]) * np.cos(self.mu[2])],)
        row_3 = ([0, 0, 1],)
        return np.vstack([row_1, row_2, row_3])

    def _G_jacobian(self):
        """Computes the Jacobian G as partioal f/v evaluated at the mean state estimate (mu)."""
        row_1 = [np.cos(self.mu[2]), 0]
        row_2 = [np.sin(self.mu[2]), 0]
        row_3 = [0, 1]
        return np.vstack([row_1, row_2, row_3])

    def _H_jacobian(self):
        """Computes the Jacobian H as partioal h/x evaluated at the mean state estimate (mu)."""
        z_t = self.mu[0] ** 2 + self.mu[1] ** 2

        row_1 = [2 * self.mu[0], 2 * self.mu[1], 0]
        row_2 = [-1 * self.mu[1] / z_t, self.mu[0] / z_t, 0]
        return np.vstack([row_1, row_2])

    def _angle_wrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        while theta < -np.pi:
            theta += 2 * np.pi

        while theta > np.pi:
            theta -= 2 * np.pi

        return theta

    def _update_part_c(self, mu_bar, sigma_bar, z):
        """Update function for part C of assignment."""
        self.mu = mu_bar
        self.Sigma = sigma_bar
        return None

    def prediction(self, u):
        """
        Perform the EKF prediction step based on control u.

        Updates the variables self.mu_bar and self.sigma_bar

        Parameters
        ----------
        u : numpy.ndarray
            A 2-element vector that includes the forward distance that the
            robot traveled and its change in orientation.
        """

        # Sample motion noise
        v_t = np.random.multivariate_normal(np.zeros(self.dim_N), self.R)

        # Compute Jacobians
        F_t = self._F_jacobian(u, v_t)
        G_t = self._G_jacobian()

        # Compute mu_bar and sigma_bar
        mu_bar = self._motion_model(self.mu, u, v_t)
        sigma_bar = (F_t @ self.Sigma @ F_t.T) + (G_t @ self.R @ G_t.T)

        return mu_bar, sigma_bar

    def update(self, mu_bar, sigma_bar, z):
        """
        Perform the EKF update step based on observation z.

        Updates the variables self.mu and self.Sigma

        Parameters
        ----------
        mu_bar : numpy.ndarray
            The predicted mean vector

        sigma_bar : numpy.ndarray
            The predicted covariance matrix

        z : numpy.ndarray
            A 2-element vector that includes the range and bearing to the
            landmark.
        """

        # Sample measurement noise
        w_t = np.random.multivariate_normal(np.zeros(self.dim_M), self.Q)

        # Compute Jacobian
        H_t = self._H_jacobian()

        # Compute Kalman gain
        K_t = sigma_bar @ H_t.T @ np.linalg.inv((H_t @ sigma_bar @ H_t.T) + self.Q)

        # Compute error term
        error = z - self._measurement_model(mu_bar, w_t)
        error[1] = self._angle_wrap(error[1])  # Wrap angle for theta error

        # Update mu and sigma
        self.mu = mu_bar + (K_t @ error)
        self.Sigma = (np.identity(3) - (K_t @ H_t)) @ sigma_bar

        return None

    def run(self, U, Z, seed=42, show_animation=False, part_c=False):
        """
        Main EKF loop that iterates over control and measurement data.

        Parameters
        ----------
        U : numpy.ndarray
            A 2 x T array, where each column provides the control input
            at that time step.
        Z : numpy.ndarray
            A 2 x T array, where each column provides the measurement
            at that time step
        seed : int
            The random seed to use for reproducibility
        show_animation : bool
            Whether or not to show the animation of the EKF
        part_c : bool
            Which update function to use (part C of assignment)
        """
        seed_everything(seed=seed)

        for t in range(np.size(U, 1)):
            mu_bar, sigma_bar = self.prediction(U[:, t])
            if not part_c:
                self.update(mu_bar, sigma_bar, Z[:, t])
            else:
                self._update_part_c(mu_bar, sigma_bar, Z[:, t])

            self.MU = np.column_stack((self.MU, self.mu))
            self.VAR = np.column_stack((self.VAR, np.diag(self.Sigma)))

            if show_animation:
                self.renderer.render(self.mu, self.Sigma, self.XYT[:, t])

        self.renderer.drawTrajectory(self.MU[0:2, :], self.XYT[0:2, :])
        self.renderer.plotError(self.MU, self.XYT, self.VAR)

        plt.ioff()
        plt.show()
