
import numpy as np
import scipy.linalg


"""
Unscented Kalman Filter (UKF) for tracking bounding boxes in image space.

Implements the same public API as KalmanFilter (initiate, predict, update,
gating_distance, predict_bbox) so it is a drop-in replacement.

State space (8-dimensional):
    x, y, a, h, vx, vy, va, vh

where (x, y) is the bounding box center, a is the aspect ratio (w/h), h is
the height, and the latter four are their respective velocities.

The motion model is the same constant-velocity model as the standard KF.
The observation function h(x) = x[0:4] is linear, so the UKF's gain is the
same as the EKF / KF for this particular problem — but the UKF framework
generalises cleanly to any non-linear extension (e.g. turning-rate models,
non-linear aspect-ratio dynamics) without re-deriving Jacobians.
"""


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) for cell bounding-box tracking.

    Sigma-point parameters follow the standard Wan & van der Merwe (2000) choice:
        alpha = 1e-3  (spread of sigma points around mean)
        beta  = 2     (Gaussian prior assumption; optimal for Gaussian noise)
        kappa = 0     (secondary scaling; 0 is common for state-estimation)

    These can be overridden via the constructor for experimentation.
    """

    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.n = 8           # state dimension
        self.m = 4           # measurement dimension
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Derived sigma-point weights
        lam = alpha ** 2 * (self.n + kappa) - self.n
        self._lambda = lam

        # Mean weights  Wm
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + lam)))
        self.Wm[0] = lam / (self.n + lam)

        # Covariance weights  Wc
        self.Wc = self.Wm.copy()
        self.Wc[0] = self.Wm[0] + (1 - alpha ** 2 + beta)

        # State-transition matrix (constant-velocity, dt=1)
        ndim, dt = 4, 1
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt          # position += velocity * dt

        # Observation matrix  H (extract position state only)
        self._H = np.eye(ndim, 2 * ndim)       # z = H x

        # Noise scale factors (identical to KalmanFilter for fair comparison)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate 2n+1 sigma points from the current state distribution."""
        n = self.n
        scale = np.sqrt((n + self._lambda) * np.ones(n))

        try:
            S = np.linalg.cholesky((n + self._lambda) * cov)
        except np.linalg.LinAlgError:
            # Regularise if near-singular
            cov_reg = cov + np.eye(n) * 1e-6
            S = np.linalg.cholesky((n + self._lambda) * cov_reg)

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = mean
        for i in range(n):
            sigmas[i + 1]     = mean + S[i]
            sigmas[n + i + 1] = mean - S[i]
        return sigmas

    def _process_noise(self, mean: np.ndarray) -> np.ndarray:
        """Build the 8×8 process-noise covariance matrix Q."""
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        return np.diag(np.square(np.r_[std_pos, std_vel]))

    def _measurement_noise(self, mean: np.ndarray) -> np.ndarray:
        """Build the 4×4 measurement-noise covariance matrix R."""
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        return np.diag(np.square(std))

    # ------------------------------------------------------------------
    # Public API  (mirrors KalmanFilter exactly)
    # ------------------------------------------------------------------

    def _bbox2pos(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x_min, y_min, x_max, y_max] → [cx, cy, a, h]."""
        ndim_recover = False
        if bbox.ndim == 1:
            ndim_recover = True
            bbox = bbox[None, :]
        x_cen = (bbox[:, 0] + bbox[:, 2]) / 2.0
        y_cen = (bbox[:, 1] + bbox[:, 3]) / 2.0
        asp   = (bbox[:, 2] - bbox[:, 0]) / (bbox[:, 3] - bbox[:, 1])
        h     = bbox[:, 3] - bbox[:, 1]
        out   = np.stack([x_cen, y_cen, asp, h], axis=1).astype(np.float32)
        return np.squeeze(out) if ndim_recover else out

    def predict_bbox(self, mean: np.ndarray) -> np.ndarray:
        """Convert state mean back to [x_min, y_min, x_max, y_max]."""
        x_cen, y_cen, asp, height = mean[0:4]
        height = max(0.0, float(height))
        width  = max(0.0, float(asp) * height)
        x_min  = x_cen - width  / 2
        y_min  = y_cen - height / 2
        return np.maximum(np.array([x_min, y_min, x_min + width + 1, y_min + height + 1]), 0.0)

    def initiate(self, measurement: np.ndarray):
        """
        Create a new track from an unassociated detection.

        Parameters
        ----------
        measurement : ndarray  shape (4,)
            Bounding box [x_min, y_min, x_max, y_max].

        Returns
        -------
        mean       : ndarray  shape (8,)
        covariance : ndarray  shape (8, 8)
        """
        pos  = self._bbox2pos(measurement)
        mean = np.r_[pos, np.zeros(4, dtype=np.float32)]

        h = pos[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """
        UKF prediction step.

        Propagates each sigma point through the (linear) state-transition
        function, then recompose the predicted mean and covariance.

        Parameters
        ----------
        mean       : ndarray  shape (8,)
        covariance : ndarray  shape (8, 8)

        Returns
        -------
        mean_pred : ndarray  shape (8,)
        cov_pred  : ndarray  shape (8, 8)
        """
        Q = self._process_noise(mean)
        sigmas = self._sigma_points(mean, covariance)

        # Propagate sigma points through f(x) = F @ x
        sigmas_pred = (self._F @ sigmas.T).T  # shape (2n+1, n)

        # Recover predicted mean
        mean_pred = np.einsum('i,ij->j', self.Wm, sigmas_pred)

        # Recover predicted covariance
        diff = sigmas_pred - mean_pred
        cov_pred = np.einsum('i,ij,ik->jk', self.Wc, diff, diff) + Q

        return mean_pred, cov_pred

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """
        UKF correction (measurement update) step.

        Parameters
        ----------
        mean        : ndarray  shape (8,)   Predicted state mean.
        covariance  : ndarray  shape (8, 8) Predicted state covariance.
        measurement : ndarray  shape (4,)   Observed bounding box [x_min, y_min, x_max, y_max].

        Returns
        -------
        new_mean       : ndarray  shape (8,)
        new_covariance : ndarray  shape (8, 8)
        """
        z = self._bbox2pos(measurement)      # shape (4,)
        R = self._measurement_noise(mean)    # shape (4, 4)

        sigmas = self._sigma_points(mean, covariance)  # (2n+1, 8)

        # Project sigma points into measurement space via H (linear here)
        sigmas_z = (self._H @ sigmas.T).T   # (2n+1, 4)

        # Predicted measurement mean
        z_pred = np.einsum('i,ij->j', self.Wm, sigmas_z)  # (4,)

        # Innovation covariance  S
        dz = sigmas_z - z_pred
        S = np.einsum('i,ij,ik->jk', self.Wc, dz, dz) + R

        # Cross covariance  Pxz
        dx = sigmas - mean
        Pxz = np.einsum('i,ij,ik->jk', self.Wc, dx, dz)

        # Kalman gain  K = Pxz @ S^{-1}
        try:
            chol, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
            K = scipy.linalg.cho_solve((chol, lower), Pxz.T, check_finite=False).T
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(S)

        # State update
        innovation  = z - z_pred
        new_mean    = mean + K @ innovation
        new_cov     = covariance - K @ S @ K.T

        return new_mean, new_cov

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                        measurements: np.ndarray, only_position: bool = False) -> np.ndarray:
        """
        Compute squared Mahalanobis distance between predicted state and measurements.

        Compatible with chi2inv95 thresholds (same as KalmanFilter).

        Parameters
        ----------
        mean         : ndarray  shape (8,)
        covariance   : ndarray  shape (8, 8)
        measurements : ndarray  shape (N, 4)  each row [x_min, y_min, x_max, y_max]
        only_position: bool     if True, use only (x, y) — 2 DOF

        Returns
        -------
        ndarray  shape (N,)  squared Mahalanobis distances
        """
        measurements = self._bbox2pos(measurements)

        # Project mean/cov into measurement space
        mean_z = self._H @ mean
        S      = self._H @ covariance @ self._H.T + self._measurement_noise(mean)

        if only_position:
            mean_z = mean_z[:2]
            S      = S[:2, :2]
            measurements = measurements[:, :2]

        chol = np.linalg.cholesky(S)
        d    = measurements - mean_z
        z    = scipy.linalg.solve_triangular(chol, d.T, lower=True, check_finite=False)
        return np.sum(z * z, axis=0)
