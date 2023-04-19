from pdb import set_trace

import matplotlib.pyplot as plt
import numpy as np

from Renderer import Renderer


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

        # Keep track of mean and variance over time
        self.MU = mu
        self.VAR = np.diag(self.Sigma).reshape(3, 1)

        xLim = np.array((np.amin(XYT[0, :] - 2), np.amax(XYT[0, :] + 2)))
        yLim = np.array((np.amin(XYT[1, :] - 2), np.amax(XYT[1, :] + 2)))

        self.renderer = Renderer(xLim, yLim, 3, "red", "green")

    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        while theta < -np.pi:
            theta = theta + 2 * np.pi

        while theta > np.pi:
            theta = theta - 2 * np.pi

        return theta

    def motion_noise(self):
        """
        Sample from a Gaussian distribution.

        Returns
        -------
        numpy.ndarray
            A sample from the Gaussian distribution
        """
        return np.random.multivariate_normal(mean=np.zeros(2), cov=self.R)

    def measurement_noise(self):
        """
        Sample from a Gaussian distribution.

        Returns
        -------
        numpy.ndarray
            A sample from the Gaussian distribution
        """
        return np.random.multivariate_normal(mean=np.zeros(2), cov=self.Q)

    def F_Jacobian(self, mu_theta, d_t, v_d):
        return np.array(
            [
                [1, 0, -(d_t + v_d) * np.sin(mu_theta)],
                [0, 1, (d_t + v_d) * np.cos(mu_theta)],
                [0, 0, 1],
            ]
        )

    def G_Jacobian(self, mu_theta):
        return np.array([[np.cos(mu_theta), 0], [np.sin(mu_theta), 0], [0, 1]])

    def H_Jacobian(self, mu_x, mu_y, z_r_hat):
        return np.array(
            [[2 * (mu_x), 2 * (mu_y), 0], [-(mu_y) / z_r_hat, (mu_x) / z_r_hat, 0]]
        )

    def prediction(self, u):
        """
        Perform the EKF prediction step based on control u.

        Parameters
        ----------
        u : numpy.ndarray
            A 2-element vector that includes the forward distance that the
            robot traveled and its change in orientation.
        """
        d_t, delta_theta_t = u
        mu_x, mu_y, mu_theta = self.mu
        v_d, v_theta = self.motion_noise()

        # Update the mu
        mu_x_new = mu_x + (d_t + v_d) * np.cos(mu_theta)
        mu_y_new = mu_y + (d_t + v_d) * np.sin(mu_theta)
        mu_theta_new = self.angleWrap(mu_theta + delta_theta_t + v_theta)
        mu_new = np.array([mu_x_new, mu_y_new, mu_theta_new])

        # Compute the Jacobians F and G
        F = self.F_Jacobian(mu_theta, d_t, v_d)
        G = self.G_Jacobian(mu_theta)

        # Update the covariance
        Sigma_new = F @ self.Sigma @ F.T + G @ self.R @ G.T

        # Update the class attributes
        self.mu = mu_new
        self.Sigma = Sigma_new

    def update(self, z):
        """
        Perform the EKF update step based on observation z.

        Parameters
        ----------
        z : numpy.ndarray
            A 2-element vector that includes the squared distance between
            the robot and the sensor, and the robot's heading.
        """
        z_r, z_theta = z
        mu_x, mu_y, mu_theta = self.mu
        w_r, w_theta = self.measurement_noise()

        # Compute the squared distance and angle
        z_r_hat = (mu_x) ** 2 + (mu_y) ** 2
        z_theta_hat = self.angleWrap(np.arctan2(mu_y, mu_x) - mu_theta)

        # Compute the Jacobian H
        H = self.H_Jacobian(mu_x, mu_y, z_r_hat)

        # Compute Kalman gain
        K = self.Sigma @ H.T @ np.linalg.inv(H @ self.Sigma @ H.T + self.Q)

        # Update the mean
        mu_new = self.mu + K @ (
            np.array([z_r, z_theta]) - np.array([z_r_hat, z_theta_hat])
        )

        # Update the covariance
        Sigma_new = (np.eye(3) - K @ H) @ self.Sigma

        # Update the class attributes
        self.mu = mu_new
        self.Sigma = Sigma_new

    def run(self, U, Z):
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
        """
        for t in range(np.size(U, 1)):
            self.prediction(U[:, t])
            self.update(Z[:, t])

            self.MU = np.column_stack((self.MU, self.mu))
            self.VAR = np.column_stack((self.VAR, np.diag(self.Sigma)))

            self.renderer.render(self.mu, self.Sigma, self.XYT[:, t])

        self.renderer.drawTrajectory(self.MU[0:2, :], self.XYT[0:2, :])
        self.renderer.plotError(self.MU, self.XYT, self.VAR)
        plt.ioff()
        plt.show()
