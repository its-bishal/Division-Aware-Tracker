

class TrackerHyperParams:
    max_record_frame = 500 # Maximum number of frames whose tracking information is stored. Prevents unlimited memory growth.
    iou_base = 0.1 # Minimum Intersection-over-Union required between predicted and detected cell for a possible match.
    max_track_node = 500 # Maximum number of observations (nodes) allowed within a single track. Limits trajectory length.
    max_objects = 1000 # Maximum number of cells allowed per frame. Prevents excessive memory allocation.
    using_kfgrating = False # Enables/disables Kalman Filter gating. Gating rejects detections too far from Kalman prediction.
    mitosis_threshold = 0.15 # Threshold for detecting cell division (mitosis). Helps decide whether one parent becomes two daughter tracks.
    distance_threshold = 1000.0 # Maximum allowable distance between predicted position and detected centroid.
    area_threshold = 1 # Minimum acceptable area similarity or minimum segmented object area. Helps reject tiny noisy regions.
    edge_pixel = 0 # Pixels near image border ignored during tracking. Used because boundary objects often leave the field of view.
    min_merge_threshold = 0.5 # Threshold used to decide whether two detections should be merged or considered one object.
    minimal_tracklet_length = 2 # Minimum number of consecutive detections required before a trajectory is accepted as a valid cell track.
    roi_verify_threshold = 2 # Number of frames used to verify whether a region of interest (ROI) is a valid object before confirming tracking.
    roi_verify_punish_rate = 0.6 # Penalty applied when ROI verification fails. Lowers confidence of unreliable tracks.
    max_frame_gap = 5 # Maximum number of consecutive frames where a cell can disappear and still keep the same track ID.
    max_missed_frame = 5 # Maximum number of missed detections before terminating a track. Often used together with Kalman prediction.

    # LDDMM matching hyperparameters
    lddmm_distance_threshold = 50.0 # Maximum distance between parent and potential daughter centroids to compute LDDMM cost
    lddmm_energy_threshold = 2.0 # Maximum acceptable LDDMM deformation energy to consider it a valid division
    lddmm_iterations = 50 # Number of iterations for diffeomorphic matching optimization
    lddmm_sigma = 10.0 # Smoothing parameter for LDDMM vector field
    lddmm_step_size = 0.01 # Step size for LDDMM optimization
