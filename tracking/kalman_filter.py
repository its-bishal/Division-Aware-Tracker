

import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class KalmanFilter:
    """
    A simple kalman filter for tracking bounding boxes in image space REF. DeepSORT.

    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    
    contains the bounding box xenter position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (Linear observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1

        # create kalman filter model matrices
        self._motion_matrix = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_matrix[1, ndim + i] = dt
        self._update_matrix = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainity are chosen relative to the current
        # state estimate. These weights control the amount of uncertainity in the model.
        # This is a bit hacky
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    

    def _bbox2pos(self, bbox):
        """
        : param bbox: ndarray Bounding box coordinates (x_min, y_min, x_max, y_max)
        : return: Bounding box coordinates (x, y, a, h) with center position (x, y),
        aspect ratio a, and height h.
        """
        ndim_recover =  False
        if bbox.ndim == 1:
            ndim_recover = True
            bbox = bbox[None, :]
        x_cen = (bbox[:, 0] + bbox[:, 2]) / 2.0
        y_cen = (bbox[:, 1] + bbox[:, 3]) / 2.0
        aspect_ratio = (bbox[:, 2] - bbox[:, 0]) / (bbox[:, 3] - bbox[:, 1]) # width/height
        height = bbox[:, 3] - bbox[:, 1]
        if ndim_recover:
            return np.squeeze(np.stack([x_cen, y_cen, aspect_ratio, height], axis=1).astype(np.float32))
        return np.stack([x_cen, y_cen, aspect_ratio, height], axis=1).astype(np.float32)
    
    def predict_bbox(self, mean):
        x_cen, y_cen, asp_ratio, height = mean[0:4]
        height = max(0.0, height)
        width = asp_ratio * height
        width = max(0.0, width)
        x_min = x_cen - width / 2
        y_min = y_cen - height / 2
        x_max = x_min + width + 1
        y_max = y_min + height + 1

        return np.maximum(np.array([x_min, y_min, x_max, y_max]), 0.)
    
    def initiate(self, measurement):
        """
        Create track from unassociated measurement.
        
        : param measurement: ndarray bounding box coordinates (x_min, y_min, x_max, y_max)
        : return (ndarray, ndarray): Returns the mean vector (8 dimensional) and covariance matrix
            (8x8 dimensional) of the new track. Unobserved velocities are initiated to 0 mean.
        """

        measurement = self._bbox2pos(measurement)
        mean_pos = measurement
        mean_velocity = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_velocity]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]

        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """
        Run kalman filter prediction step.
        
        : param mean (ndarray): the 8 dimensional mean vector of the object state at the previous time steo.
                covariance (ndarray) : the 8x8 dimensional covariance matrix of the object state at the precious time step.
        
        : return (ndarray, ndarray): Returns the mean vector and covariance matrix of the predicted state.
                Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_matrix, mean)
        covariance = np.linalg.multi_dot((
            self._motion_matrix, covariance, self._motion_matrix.T
        )) + motion_cov

        return mean, covariance
    
    def _project(self, mean, covariance):
        """
        Project state distribution to measurement space.
        
        :params mean (ndarray) : The state's mean vector (8 dimensional array).
                covariance (ndarray): The state's covariance matrix (8x8 dimensional).
        
        : return (ndarray, ndarray): Returns the projected mean and covariance matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_matrix, mean)
        covariance = np.linalg.multi_dot((
            self._update_matrix, covariance, self._update_matrix.T
        ))
        return mean, covariance + innovation_cov
    
    def update(self, mean, covariance, measurement):
        """
        Run kalman filter correction step.
        
        :params mean: the predicted state's mean vector (8 dimensional).
                covariance(ndarray): the state's covariance matrix (8x8 dimensional).
                measurement (ndarray): The 4 dimensional measurement vector: bounding box coordinates (x_min, y_min, x_max, y_max)
        
        : returns (ndarray, ndarray): Returns the measurement-corrected state distribution.
        """
        measurement = self._bbox2pos(measurement)
        projected_mean, projected_cov = self._project(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_matrix.T).T,
            check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance -  np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Computing gating distance between state distribution and measurements.
        
        A suitable distance threshold can be obtained from 'chi2inv95'.
        If 'only_position' is False, the chi-square distribution has 4 degrees of freedom, otherwise 2.
        
        : params
            mean (ndarray): mean vector over the state distribution (8 dimensional).
            covariance (ndarray): covariance of the state distribution (8x8 dimensional).
            measurements (ndarray): An Nx4 dimensional matrix of N measurements, each in format (x, y, a, h)
                where (x, y) is the bounding box center position, a the aspect ratio and h the height.
            only_position (Optional[bool]): If true, distance computation is done with respect to the bounding box center position only.
        
        : returns
            ndarray : Retruns an array of length N, where the i-th element contains the squared Manhalanobis distance between (mean, covariance) and measurements[i].
        """

        measurements = self._bbox2pos(measurements)
        mean, covariance = self._project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return np.array(squared_maha)
    