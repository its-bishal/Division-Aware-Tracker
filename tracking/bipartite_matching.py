
import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracks import Tracks
from .tracks import Region
from .recorder import Recorder
from .kalman_filter import KalmanFilter
from utils.configs import TrackerHyperParams
from .lddmm import compute_lddmm_cost


class BiTracker:
    def __init__(self):

        self.minimal_tracklet_len = TrackerHyperParams.minimal_tracklet_length
        self.roi_verify_max_iteration = TrackerHyperParams.roi_verify_threshold
        self.roi_verify_punish_rate = TrackerHyperParams.roi_verify_punish_rate
        self.using_kfgating = TrackerHyperParams.using_kfgrating
        self.mitosis_th = TrackerHyperParams.mitosis_threshold
        self.dis_th = TrackerHyperParams.distance_threshold
        self.area_th = TrackerHyperParams.area_threshold
        self.edge_pixel = TrackerHyperParams.edge_pixel
        
        self.kalman_filter = KalmanFilter()
        self.tracks = Tracks()
        self.recorder = Recorder()

        self.frame_index = 0
        self.predict_detections = []

        self.img_sz = None

    def get_all_tracks(self):
        return self.tracks.get_all_tracks(minimal_len=self.minimal_tracklet_len)

    def _assignment_mitosis(self, seg_ious, assigned_pairs_id, unassigned_tracks_id, 
                            unassigned_detections_id, ids, detections_id=None, mitosis_th=0.15):
        track_num = seg_ious.shape[0]
        if track_num > 0:
            box_num = seg_ious.shape[1]
        else:
            box_num = 0
        if detections_id is None:
            detections_id = list(range(box_num))
        
        continue_pairs_id, mitosis_pair_id, children_det_id = [], [], []
        assigned_pairs_id_continue = []
        for (track_id, det_id) in assigned_pairs_id:
            t_index = ids.index(track_id)
            iou = seg_ious[t_index, :]
            
            if len(iou) < 2:
                assigned_pairs_id_continue.append((track_id, det_id))
                continue
                
            topk_index = (-iou).argsort()[:2]
            candidate_id = [detections_id[topk_index[0]], detections_id[topk_index[1]]]
            if iou[topk_index[1]] >= mitosis_th:
                if candidate_id[0] not in children_det_id and \
                    candidate_id[1] not in children_det_id:
                    children_det_id.extend(candidate_id)
                    mitosis_pair_id.append((track_id, candidate_id[0], candidate_id[1]))
                else:
                    assigned_pairs_id_continue.append((track_id, det_id))
            else:
                assigned_pairs_id_continue.append((track_id, det_id))
            
        for (track_id, det_id) in assigned_pairs_id_continue:
            if det_id not in children_det_id:
                continue_pairs_id.append((track_id, det_id))
        
        assigned_det_id = [pairs[1] for pairs in continue_pairs_id] + \
            [pairs[1] for pairs in mitosis_pair_id] + \
                [pairs[2] for pairs in mitosis_pair_id]
        unassigned_detections_id = sorted(set(detections_id) - set(assigned_det_id))

        assigned_trk_id = [pairs[0] for pairs in continue_pairs_id] + \
            [pairs[0] for pairs in mitosis_pair_id]
        unassigned_tracks_id = sorted(set(ids) - set(assigned_trk_id))

        return continue_pairs_id, mitosis_pair_id, unassigned_tracks_id, unassigned_detections_id
 

    def _assignment_heuristic(self, tracks, distance, ids, detections_id=None):
        # find the corresponding by the similar matrix
        track_num = distance.shape[0]
        if track_num > 0:
            box_num = distance.shape[1]
        else:
            box_num = 0
        if detections_id is None:
            detections_id = list(range(box_num))

        row_index, col_index = linear_sum_assignment(distance)

        # verification by iou
        verify_iteration = 0
        while verify_iteration < self.roi_verify_max_iteration:
            is_change = False
            for idx, track_idx in enumerate(row_index):
                track_id = ids[track_idx]
                det_idx = col_index[idx]
                det_id = detections_id[det_idx]
                t = tracks.get_track_by_id(track_id)
                # TODO verify in MHT
                if not t.verify(self.frame_index, self.recorder, det_id):
                    distance[track_idx, det_idx] /= self.roi_verify_punish_rate
                    is_change = True
            if is_change:
                row_index, col_index = linear_sum_assignment(distance)
            else:
                break
            verify_iteration += 1

        assigned_pairs_id = []
        for idx, track_idx in enumerate(row_index):
            track_id = ids[track_idx]
            det_idx = col_index[idx]
            det_id = detections_id[det_idx]
            if distance[track_idx, det_idx] < 0:
                assigned_pairs_id.append((track_id, det_id))

        unassigned_tracks_id = sorted(set(ids) - {pairs[0] for pairs in assigned_pairs_id})
        unassigned_detections_id = sorted(set(detections_id) - {pairs[1] for pairs in assigned_pairs_id})
        return assigned_pairs_id, unassigned_tracks_id, unassigned_detections_id

    def _assignment_lddmm(self, unassigned_tracks_id, unassigned_detections_id):
        lddmm_pair_id = []
        new_unassigned_tracks_id = list(unassigned_tracks_id)
        new_unassigned_detections_id = list(unassigned_detections_id)
        
        if compute_lddmm_cost is None:
            return lddmm_pair_id, new_unassigned_tracks_id, new_unassigned_detections_id

        if len(new_unassigned_detections_id) < 2 or len(new_unassigned_tracks_id) == 0:
            return lddmm_pair_id, new_unassigned_tracks_id, new_unassigned_detections_id

        from itertools import combinations
        
        for track_id in unassigned_tracks_id:
            track = self.tracks.get_track_by_id(track_id)
            if len(track.nodes) == 0:
                continue
                
            parent_node = track.nodes[-1]
            parent_bbox = parent_node.bbox
            parent_mask = parent_node.region.mask
            parent_center = ((parent_bbox[0] + parent_bbox[2]) / 2, (parent_bbox[1] + parent_bbox[3]) / 2)
            
            candidates = []
            for det_id in new_unassigned_detections_id:
                region = self.recorder.get_region(self.frame_index, det_id)
                if region is None:
                    continue
                bbox = region.bbox
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                dist = np.sqrt((center[0] - parent_center[0])**2 + (center[1] - parent_center[1])**2)
                
                if dist < TrackerHyperParams.lddmm_distance_threshold:
                    candidates.append((det_id, region))
            
            if len(candidates) >= 2:
                best_energy = float('inf')
                best_pair = None
                
                for (det1, reg1), (det2, reg2) in combinations(candidates, 2):
                    combined_mask = np.logical_or(reg1.mask > 0, reg2.mask > 0).astype(np.float32)
                    try:
                        energy = compute_lddmm_cost(parent_mask, combined_mask, TrackerHyperParams)
                    except Exception as e:
                        print(f"LDDMM computation failed: {e}")
                        energy = float('inf')
                        
                    if energy < best_energy and energy < TrackerHyperParams.lddmm_energy_threshold:
                        best_energy = energy
                        best_pair = (det1, det2)
                
                if best_pair is not None:
                    lddmm_pair_id.append((track_id, best_pair[0], best_pair[1]))
                    new_unassigned_tracks_id.remove(track_id)
                    new_unassigned_detections_id.remove(best_pair[0])
                    new_unassigned_detections_id.remove(best_pair[1])
                    
        return lddmm_pair_id, new_unassigned_tracks_id, new_unassigned_detections_id

    def update_detections(self, detections, mask, area_th=20, edge_pixel=10):
        def bbox_area(bbox):
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        def edge_filter(bbox, height, width, edge_pixel):
            return not (width - bbox[0] <= edge_pixel or bbox[2] <= edge_pixel or height - bbox[1] <= edge_pixel or bbox[3] <= edge_pixel)
        regions = []
        detections_filter = []
        height, width = mask[0].shape
        for idx in range(detections.shape[0]):
            bbox = detections[idx, 0:4]
            if bbox_area(bbox) > pow(area_th, 2) and edge_filter(bbox, height, width, edge_pixel):
                regions.append(Region(detections[idx, :], mask[idx]))
                detections_filter.append(detections[idx, :])
        return regions, np.array(detections_filter)


    def update_track(self, frame_index, img_size, detections, mask):
        """
        :param frame_index: frame number of video
        :param img_size: image size of original image for this frame
        :param detections: all detections results for one frame
        :return:
        """
        self.frame_index = frame_index
        self.img_sz = img_size  # [height, width]


        regions, detections = self.update_detections(detections, mask, area_th=self.area_th, edge_pixel=self.edge_pixel)
        if len(detections) == 0:
            return True

        self.recorder.update(self.frame_index, regions, detections)

        # Create new branch from the every detection results
        if self.frame_index == 0 or len(self.tracks.tracks) == 0:
            for index in range(detections.shape[0]):
                _ = self.tracks.add_new_track(frame_index=self.frame_index,
                                            det_id=index,
                                            regions=regions[index],
                                            kalman_filter=self.kalman_filter)
            self.tracks.one_frame_pass()
            return True
        else:
            distance = []
            seg_ious = []
            ids = []
            for track in self.tracks.tracks:
                ids.append(track.track_id)
                regions = self.recorder.get_region(self.frame_index, None)
                dis, ious = track.get_distance(self.frame_index, detections, regions,
                                                kf=self.kalman_filter, dis_th=self.dis_th,
                                                using_kfgating=self.using_kfgating)
                distance.append(dis)
                seg_ious.append(ious)
            distance = np.array(distance).squeeze().reshape(-1, len(detections))
            seg_ious = np.array(seg_ious).squeeze().reshape(-1, len(detections))

            if len(distance) > 0:
                assigned_pairs_id, unassigned_tracks_id, unassigned_detections_id = \
                  self._assignment_heuristic(self.tracks, -seg_ious, ids)

                continue_pairs_id, mitosis_pair_id, unassigned_tracks_id, unassigned_detections_id = \
                    self._assignment_mitosis(seg_ious, assigned_pairs_id, unassigned_tracks_id, 
                            unassigned_detections_id, ids, detections_id=None, mitosis_th=self.mitosis_th)

                # Add LDDMM assignment for remaining unassigned ones
                lddmm_pair_id, unassigned_tracks_id, unassigned_detections_id = \
                    self._assignment_lddmm(unassigned_tracks_id, unassigned_detections_id)
                
                # Combine LDDMM discoveries with standard mitosis discoveries
                mitosis_pair_id.extend(lddmm_pair_id)

                # update the tracks for no mitosis
                for track_id, det_id in continue_pairs_id:
                    track = self.tracks.get_track_by_id(track_id)
                    track.update_track(frame_index=self.frame_index,
                                       det_id=det_id,
                                       region=self.recorder.get_region(self.frame_index, det_id),
                                       kalman_filter=self.kalman_filter,
                                       visual_mode=False)

                parent_track = []
                for track_id, det_id_1, det_id_2 in mitosis_pair_id:
                    track = self.tracks.get_track_by_id(track_id)
                    parent_track.append(track_id)
                    child_track = []
                    for det_id in [det_id_1, det_id_2]:
                        t_id = self.tracks.add_new_track(frame_index=self.frame_index,
                                                    det_id=det_id,
                                                    regions=self.recorder.get_region(self.frame_index, det_id),
                                                    kalman_filter=self.kalman_filter,
                                                    parent=track_id)
                        child_track.append(t_id)

                for track_id in unassigned_tracks_id:
                    track = self.tracks.get_track_by_id(track_id)
                    track.update_track(frame_index=self.frame_index,
                                       det_id=None, 
                                       region=None,
                                       kalman_filter=self.kalman_filter,
                                       visual_mode=False)

                # add new track
                for det_id in unassigned_detections_id:
                    _ = self.tracks.add_new_track(frame_index=self.frame_index,
                                                  det_id=det_id,
                                                  regions=self.recorder.get_region(self.frame_index, det_id),
                                                  kalman_filter=self.kalman_filter)

            # remove the old track
            self.tracks.one_frame_pass(saved_ids=parent_track)

        return True