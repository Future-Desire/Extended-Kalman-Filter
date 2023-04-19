import numpy as np
import matplotlib.pyplot as plt
from Renderer import Renderer


class EKF(object):
    """A class for implementing EKFs.

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
        """Initialize the class.

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
        
        self.N = self.mu.shape[0]
        self.M = self.Q.shape[0]

        # Keep track of mean and variance over time
        self.MU = mu
        self.VAR = np.diag(self.Sigma).reshape(3, 1)

        xLim = np.array((np.amin(XYT[0, :] - 2), np.amax(XYT[0, :] + 2)))
        yLim = np.array((np.amin(XYT[1, :] - 2), np.amax(XYT[1, :] + 2)))

        self.renderer = Renderer(xLim, yLim, 3, 'red', 'green')
        
        
        # Noise term : 2 element vector that describes nosie in angular motion
        self.v_t = np.random.multivariate_normal(np.zeros(self.N), self.R)
        self.w_t = np.random.multivariate_normal(np.zeros(self.M), self.Q)

    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        while theta < -np.pi:
            theta = theta + 2*np.pi

        while theta > np.pi:
            theta = theta - 2*np.pi

        return theta

    def prediction(self, u):
        """Perform the EKF prediction step based on control u.
        
        Updates the variables self.mu_bar and self.sigma_bar

        Parameters
        ----------
        u : numpy.ndarray
            A 2-element vector that includes the forward distance that the
            robot traveled and its change in orientation.
        """
        
        # Redraw noise term s
        self.v_t = np.random.multivariate_normal(np.zeros(self.N), self.R)
        self.w_t = np.random.multivariate_normal(np.zeros(self.M), self.Q)
        
        
            
        
        # F,G = computeFandG 
        
        # mu_bar = f(self.mu, u)
        # sigma_bar = F @ self.Sigma @ F.T +  G @ R @ G.T
        
        
        
        # Your code goes here
        pass

    def update(self, z):
        """Perform the EKF update step based on observation z.
        
        Updates the variables self.mu and self.Sigma

        Parameters
        ----------
        z : numpy.ndarray
            A 2-element vector that includes the squared distance between
            the robot and the sensor, and the robot's heading.
        """
        # Your code goes here
        pass

    def run(self, U, Z):
        """Main EKF loop that iterates over control and measurement data.

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
        
    def motionModelFunction(self, x, u):
        """Function that computes the motion model dynamics
        
        Parameters
        ----------
        x : numpy.ndarray
            A 3-element vector [x_t, y_t, theta_t] with (x_t, y_t) describing
            position and theta_t describing orientation
        u : numpy.ndarray
            A 2-element vector [d_t, delta_sigma_t] where d_t represents the
            body relative forward distance moved and delta_sigma_t represents
            the change in orientation
            
        Returns 
        ----------
        out : numpy.ndarray
            A 3-element vector [x_new, y_new, theta_new] of the position and 
            orientation 
        """
        
        x_new = x[0] + (u[0] + self.v_t[0]) * np.cos(x[2])
        y_new = x[1] + (u[0] + self.v_t[0]) * np.sin(x[2])
        theta_new = x[2] + u[1] + self.v_t[1]
        
        out = np.array([x_new, y_new, theta_new])
        
        return out
    
    def measurementModelFunction(self, x):
        """Function that computes the motion model dynamics
        
        Parameters
        ----------
        x : numpy.ndarray
            A 3-element vector [x_t, y_t, theta_t] with (x_t, y_t) describing
            position and theta_t describing orientation
            
        Returns 
        ----------
        out : numpy.ndarray
            A 2-element vector [x_new, y_new, theta_new] of the position and 
            orientation 
            
        """
        
        
        
        pass  
    
    def F_Jacobian(self, u):
        """Computes the Jacobian F as partial f/x evaluated at the mean state estimate (mu)
            given the control
        
        Parameters
        ----------
        u : numpy.ndarray
            A 2-element vector [d_t, delta_sigma_t] where d_t represents the
            body relative forward distance moved and delta_sigma_t represents
            the change in orientation
            
        Returns 
        ----------
        out : numpy.ndarray
            A 3 x 3 array of the Jacobian of motionModelFunction evaluated at 
            x = self.mu and input u
        """
        
        out = np.array([
            [1, 0 , -1*(u[0] + self.v_t[0]) * np.sin(self.mu[2])],
            [0, 1, (u[0] + self.v_t[0]) * np.cos(self.mu[2])],
            [0, 0, 1]
        ])
        
        return out 
    
    def G_Jacobian(self, u):
        """Computes the Jacobian G as partioal f/v evaluated at the mean state estimate (mu)
            given the control
        
        Parameters
        ----------
        u : numpy.ndarray
            A 2-element vector [d_t, delta_sigma_t] where d_t represents the
            body relative forward distance moved and delta_sigma_t represents
            the change in orientation
            
        Returns 
        ----------
        out : numpy.ndarray
            A 3 x 2 array of the Jacobian of motionModelFunction evaluated at 
            x = self.mu and input u
        """
        
        out = np.array([
            [np.cos(self.mu[2]), 0],
            [np.sin(self.mu[2]), 0],
            [0, 1]
        ])
        
        return out 
    
    def H_Jacobian(self, u):
        """Computes the Jacobian G as partioal f/v evaluated at the mean state estimate (mu)
            given the control
        
        Parameters
        ----------
        u : numpy.ndarray
            A 2-element vector [d_t, delta_sigma_t] where d_t represents the
            body relative forward distance moved and delta_sigma_t represents
            the change in orientation
            
        Returns 
        ----------
        out : numpy.ndarray
            A 2 x 3 array of the Jacobian of motionModelFunction evaluated at 
            x = self.mu and input u
        """
        
        out = np.array([
            [np.cos(self.mu[2]), 0],
            [np.sin(self.mu[2]), 0],
            [0, 1]
        ])
        
        return out
    
    