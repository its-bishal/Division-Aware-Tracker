

class BaseTrackerHyperParams:
    """Hyperparameters shared by every tracking scenario (general tracking
    behavior, not related to mitosis/division detection)."""

    max_record_frame = 500      # Max frames of tracking info stored (memory cap).
    iou_base = 0.1               # Min IoU required between predicted and detected cell for a match.
    max_track_node = 500         # Max observations (nodes) allowed within a single track.
    max_objects = 1000           # Max cells allowed per frame (memory cap).

    distance_threshold = 1000.0  # Max allowable distance between predicted position and detected centroid.
    area_threshold = 1           # Min acceptable area similarity / min segmented object area.
    edge_pixel = 0                # Border pixels ignored during tracking (objects leaving FOV).
    min_merge_threshold = 0.5     # Threshold to decide whether two detections should be merged.
    minimal_tracklet_length = 2   # Min consecutive detections before a trajectory is accepted.
    roi_verify_threshold = 2      # Frames used to verify a region of interest before confirming tracking.
    roi_verify_punish_rate = 0.6  # Confidence penalty applied when ROI verification fails.
    max_frame_gap = 5             # Max consecutive missed frames a cell can have and keep its track ID.

    # Filter selection: 'kalman' | 'ukf'
    kalman_filter_type = 'kalman'


class BaselineKalmanOnlyHyperParams(BaseTrackerHyperParams):
    """Baseline: Kalman Filter only, no LDDMM division matching.

    Uses a standard IoU threshold for easy divisions and disables LDDMM
    candidate search entirely, relying on strict Kalman gating instead.
    """

    mitosis_threshold = 0.15        # Standard IoU threshold for easy divisions.
    lddmm_distance_threshold = 0.0  # 0 disables LDDMM candidate-pair search.
    using_kfgating = True           # Strict Kalman gating rejects impossible jumps.
    max_missed_frame = 5            # Max missed detections before terminating a track.
    kalman_filter_type = 'kalman'   # Standard linear Kalman filter.


class ModelALDDMMMSEHyperParams(BaseTrackerHyperParams):
    """Model A: Kalman + LDDMM using MSE loss for division matching."""

    mitosis_threshold = 0.25         # Raised so LDDMM handles complex divisions.
    lddmm_distance_threshold = 50.0  # Search radius (px) for potential daughter cells.
    lddmm_loss_type = 'mse'
    lddmm_iterations = 50            # Standard convergence steps.
    lddmm_sigma = 10.0               # Smoothness of the deformation field.
    lddmm_step_size = 0.01
    lddmm_energy_threshold = 0.5     # Tighter threshold since MSE energy is typically lower.
    kalman_filter_type = 'kalman'    # Standard linear Kalman filter.


class ModelBLDDMMSinkhornHyperParams(BaseTrackerHyperParams):
    """Model B: Kalman + LDDMM using Sinkhorn (unbalanced OT) loss for division matching."""

    mitosis_threshold = 0.25         # Same as Model A for a fair comparison.
    lddmm_distance_threshold = 50.0  # Same search radius as Model A.
    lddmm_loss_type = 'sinkhorn'
    lddmm_iterations = 50
    lddmm_sigma = 10.0
    lddmm_step_size = 0.01
    lddmm_energy_threshold = 2.0     # Relaxed since Sinkhorn values run larger than MSE.
    lddmm_sinkhorn_blur = 0.05       # Entropic regularization; sweet spot for crisp cell boundaries.
    lddmm_sinkhorn_reach = 0.5       # Unbalanced-mass penalty; lets a parent split gracefully.
    kalman_filter_type = 'kalman'    # Standard linear Kalman filter.


class BaselineUKFOnlyHyperParams(BaseTrackerHyperParams):
    """Baseline (UKF): Unscented Kalman Filter only, no LDDMM division matching.

    Drop-in replacement for BaselineKalmanOnlyHyperParams.
    Use this scenario to compare tracking accuracy of KF vs UKF on the same dataset.
    """

    mitosis_threshold = 0.15
    lddmm_distance_threshold = 0.0
    using_kfgating = True
    max_missed_frame = 5
    kalman_filter_type = 'ukf'       # Unscented Kalman filter.


class ModelBUKFSinkhornHyperParams(BaseTrackerHyperParams):
    """Model B (UKF variant): UKF + LDDMM using Sinkhorn loss.

    Allows a 4-way comparison:
        baseline_kf  vs  baseline_ukf
        model_b_kf   vs  model_b_ukf
    """

    mitosis_threshold = 0.25
    lddmm_distance_threshold = 50.0
    lddmm_loss_type = 'sinkhorn'
    lddmm_iterations = 50
    lddmm_sigma = 10.0
    lddmm_step_size = 0.01
    lddmm_energy_threshold = 2.0
    lddmm_sinkhorn_blur = 0.05
    lddmm_sinkhorn_reach = 0.5
    kalman_filter_type = 'ukf'       # Unscented Kalman filter.


_SCENARIOS = {
    'baseline':     BaselineKalmanOnlyHyperParams,
    'model_a':      ModelALDDMMMSEHyperParams,
    'model_b':      ModelBLDDMMSinkhornHyperParams,
    'baseline_ukf': BaselineUKFOnlyHyperParams,
    'model_b_ukf':  ModelBUKFSinkhornHyperParams,
}


def TrackerHyperParams(scenario: str) -> BaseTrackerHyperParams:
    """Factory returning the hyperparameter class for a given scenario.

    Args:
        scenario: One of 'baseline', 'model_a', 'model_b'.

    Returns:
        The corresponding hyperparameter class (not an instance).

    Raises:
        ValueError: If the scenario name is not recognized.
    """
    try:
        return _SCENARIOS[scenario]
    except KeyError:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Choose from: {list(_SCENARIOS)}"
        )

