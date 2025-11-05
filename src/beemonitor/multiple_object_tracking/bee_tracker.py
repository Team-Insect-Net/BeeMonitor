# """Bee tracking implementation using Hungarian algorithm.

# This module provides custom tracking functionality for bee detection,
# including position prediction and data association.
# """

# import logging
# from typing import List, Tuple, Optional
# import numpy as np
# from scipy.optimize import linear_sum_assignment

# from beemonitor.utils.geometry import compute_centroid


# logger = logging.getLogger(__name__)

# # Type aliases
# BBox = Tuple[float, float, float, float]
# Point = Tuple[float, float]


# def predict_position(
#     prev_bbox: BBox,
#     prev_prev_bbox: BBox,
#     distance_threshold: float = 50
# ) -> BBox:
#     """Predict next position using linear motion model.
    
#     Uses the previous two positions to predict the next position
#     assuming constant velocity.
    
#     Args:
#         prev_bbox: Most recent bounding box (x1, y1, x2, y2)
#         prev_prev_bbox: Second most recent bounding box
#         distance_threshold: Maximum allowed prediction distance
        
#     Returns:
#         Predicted bounding box for next frame
        
#     Example:
#         >>> bbox1 = (10, 10, 30, 30)
#         >>> bbox2 = (15, 15, 35, 35)
#         >>> predicted = predict_position(bbox2, bbox1)
#     """
#     # Default box size
#     height = 20
#     width = 20
    
#     # Compute centroids
#     centroid_1 = compute_centroid(prev_bbox)
#     centroid_2 = compute_centroid(prev_prev_bbox)
    
#     # Linear motion prediction matrix
#     # P_k = 2*P_k-1 - P_k-2 (constant velocity model)
#     A = np.array([
#         [2, 0, -1, 0],
#         [0, 2, 0, -1]
#     ])
    
#     # Position vector
#     D_vector = np.array([
#         centroid_1[0], centroid_1[1],
#         centroid_2[0], centroid_2[1]
#     ])
    
#     # Predict next position
#     predicted_position = np.dot(A, D_vector)
    
#     # Create predicted bbox
#     predicted_bbox = (
#         predicted_position[0] - width,
#         predicted_position[1] - height,
#         predicted_position[0] + width,
#         predicted_position[1] + height
#     )
    
#     # Check if prediction is reasonable
#     distance = np.linalg.norm(np.array(centroid_1) - predicted_position)
    
#     if distance > distance_threshold:
#         # Prediction too far, return last known position
#         return prev_bbox
    
#     return predicted_bbox


# def hungarian_algorithm(
#     tracked_objects: List[Tuple[BBox, int]],
#     detections: List[BBox],
#     threshold: float = 200
# ) -> List[Tuple[int, int]]:
#     """Hungarian algorithm for optimal detection-track association.
    
#     Solves the assignment problem to optimally match detections to tracks
#     based on distance between predicted positions and detections.
    
#     Args:
#         tracked_objects: List of (predicted_bbox, track_id) tuples
#         detections: List of detection bounding boxes
#         threshold: Maximum distance for valid association
        
#     Returns:
#         List of (detection_index, track_index) associations
        
#     Example:
#         >>> tracks = [((10, 10, 30, 30), 1), ((50, 50, 70, 70), 2)]
#         >>> dets = [(12, 12, 32, 32), (100, 100, 120, 120)]
#         >>> assoc = hungarian_algorithm(tracks, dets, threshold=50)
#     """
#     if not tracked_objects or not detections:
#         return []
    
#     # Build cost matrix (distance between track predictions and detections)
#     cost_matrix = np.zeros((len(tracked_objects), len(detections)))
    
#     for i, track in enumerate(tracked_objects):
#         track_centroid = compute_centroid(track[0])
        
#         for j, det in enumerate(detections):
#             det_centroid = compute_centroid(det)
#             distance = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))
#             cost_matrix[i, j] = distance
    
#     # Solve assignment problem
#     track_indices, det_indices = linear_sum_assignment(cost_matrix)
    
#     # Filter by threshold
#     associations = []
#     for track_idx, det_idx in zip(track_indices, det_indices):
#         if cost_matrix[track_idx, det_idx] < threshold:
#             associations.append((det_idx, track_idx))
    
#     return associations


# class Track:
#     """Represents a single tracked bee.
    
#     A track maintains the history of a bee's trajectory across frames,
#     including position predictions and age management.
    
#     Attributes:
#         track_id: Unique identifier for this track
#         bbox: Current bounding box (x1, y1, x2, y2)
#         trajectory: List of all observed bounding boxes
#         trajectory_frame_numbers: Frame numbers corresponding to trajectory
#         is_dl_predictions: Whether each position was from detection (True) or prediction (False)
#         age: Number of frames since last detection
#         is_terminated: Whether this track is no longer active
    
#     Example:
#         >>> track = Track(bbox=(10, 10, 30, 30), track_id=1)
#         >>> track.update((12, 12, 32, 32), frame_num=1, is_detection=True)
#         >>> trajectory = track.get_trajectory()
#     """
    
#     def __init__(self, bbox: BBox, track_id: int):
#         """Initialize a new track.
        
#         Args:
#             bbox: Initial bounding box (x1, y1, x2, y2)
#             track_id: Unique identifier for this track
#         """
#         self.track_id = track_id
#         self.bbox = bbox
#         self.age = 0
#         self.is_terminated = False
#         self.trajectory: List[BBox] = []
#         self.trajectory_frame_numbers: List[int] = []
#         self.is_dl_predictions: List[bool] = []
    
#     def predict(self, distance_threshold: float) -> None:
#         """Predict next position based on motion history.
        
#         Args:
#             distance_threshold: Maximum allowed prediction distance
#         """
#         if len(self.trajectory) > 1:
#             self.bbox = predict_position(
#                 self.trajectory[-1],
#                 self.trajectory[-2],
#                 distance_threshold
#             )
#         else:
#             # No history, keep same position
#             self.bbox = predict_position(self.bbox, self.bbox, distance_threshold)
    
#     def update(
#         self,
#         bbox: BBox,
#         frame_number: int,
#         is_detection: bool = False
#     ) -> None:
#         """Update track with new observation or prediction.
        
#         Args:
#             bbox: New bounding box
#             frame_number: Current frame number
#             is_detection: True if from detection, False if predicted
#         """
#         if not self.is_terminated:
#             self.bbox = bbox
#             self.trajectory.append(bbox)
#             self.trajectory_frame_numbers.append(frame_number)
#             self.is_dl_predictions.append(is_detection)
#             self.age = 0
    
#     def get_state(self) -> Tuple[BBox, int]:
#         """Get current track state.
        
#         Returns:
#             Tuple of (bbox, track_id)
#         """
#         return self.bbox, self.track_id
    
#     def get_trajectory(self) -> Tuple[int, List[Point], List[BBox], List[int]]:
#         """Get track trajectory with only detection-based positions.
        
#         Returns:
#             Tuple of (track_id, centroids, bboxes, frame_numbers)
#             Only includes positions from actual detections, not predictions.
            
#         Example:
#             >>> track_id, centroids, bboxes, frames = track.get_trajectory()
#             >>> print(f"Track {track_id} has {len(centroids)} detections")
#         """
#         # Filter to only detection-based positions
#         trajectory = [
#             self.trajectory[i]
#             for i in range(len(self.trajectory))
#             if self.is_dl_predictions[i]
#         ]
        
#         trajectory_centroids = [compute_centroid(bbox) for bbox in trajectory]
        
#         trajectory_frame_numbers = [
#             self.trajectory_frame_numbers[i]
#             for i in range(len(self.trajectory_frame_numbers))
#             if self.is_dl_predictions[i]
#         ]
        
#         return self.track_id, trajectory_centroids, trajectory, trajectory_frame_numbers
    
#     def __repr__(self) -> str:
#         """String representation of track."""
#         return (
#             f"Track(id={self.track_id}, age={self.age}, "
#             f"terminated={self.is_terminated}, positions={len(self.trajectory)})"
#         )


# class BeeTracker:
#     """Multi-object tracker for bees using Hungarian algorithm.
    
#     This tracker maintains multiple Track objects and associates new
#     detections with existing tracks using the Hungarian algorithm for
#     optimal assignment.
    
#     Attributes:
#         max_age: Maximum frames to keep track without detection
#         distance_threshold: Threshold for position prediction
#         association_threshold: Maximum distance for valid association
#         next_id: Next available track ID
#         objects: List of all Track objects
    
#     Example:
#         >>> tracker = BeeTracker(max_age=30, distance_threshold=100)
#         >>> frame1_detections = [(10, 10, 30, 30), (50, 50, 70, 70)]
#         >>> tracks = tracker.update(frame1_detections, frame_number=0)
#         >>> frame2_detections = [(12, 12, 32, 32), (52, 52, 72, 72)]
#         >>> tracks = tracker.update(frame2_detections, frame_number=1)
#         >>> all_trajectories = tracker.get_tracks()
#     """
    
#     def __init__(
#         self,
#         max_age: int = 30,
#         distance_threshold: float = 100,
#         association_threshold: float = 200,
#         track_start_id: int = 0
#     ):
#         """Initialize BeeTracker.
        
#         Args:
#             max_age: Maximum frames to keep track without detection
#             distance_threshold: Threshold for position prediction (pixels)
#             association_threshold: Maximum distance for association (pixels)
#             track_start_id: Starting ID for tracks
#         """
#         self.max_age = max_age
#         self.distance_threshold = distance_threshold
#         self.association_threshold = association_threshold
#         self.next_id = track_start_id
#         self.objects: List[Track] = []
        
#         logger.debug(
#             f"Initialized BeeTracker (max_age={max_age}, "
#             f"distance_threshold={distance_threshold}, "
#             f"association_threshold={association_threshold})"
#         )
    
#     def update(
#         self,
#         detections: List[BBox],
#         frame_number: int
#     ) -> List[Tuple[BBox, int]]:
#         """Update tracker with new detections.
        
#         This method:
#         1. Predicts positions for existing tracks
#         2. Associates detections with tracks
#         3. Updates matched tracks
#         4. Creates new tracks for unmatched detections
#         5. Ages and terminates old tracks
        
#         Args:
#             detections: List of detection bounding boxes
#             frame_number: Current frame number
            
#         Returns:
#             List of (bbox, track_id) for active tracks
            
#         Example:
#             >>> tracker = BeeTracker()
#             >>> detections = [(10, 10, 30, 30), (50, 50, 70, 70)]
#             >>> active_tracks = tracker.update(detections, frame_number=0)
#             >>> print(f"{len(active_tracks)} active tracks")
#         """
#         # Step 1: Predict positions for all existing tracks
#         for obj in self.objects:
#             obj.predict(distance_threshold=self.distance_threshold)
        
#         # Step 2: Associate detections with existing tracks
#         if len(detections) > 0:
#             if len(self.objects) == 0:
#                 # No existing tracks, create new ones for all detections
#                 for det in detections:
#                     self.objects.append(Track(det, self.next_id))
#                     self.next_id += 1
#             else:
#                 # Associate detections with existing tracks
#                 associations = hungarian_algorithm(
#                     [obj.get_state() for obj in self.objects],
#                     detections,
#                     threshold=self.association_threshold
#                 )
                
#                 detection_indices = set(range(len(detections)))
#                 track_indices = set(range(len(self.objects)))
                
#                 # Step 3: Update matched tracks
#                 for det_idx, track_idx in associations:
#                     self.objects[track_idx].update(
#                         detections[det_idx],
#                         frame_number,
#                         is_detection=True
#                     )
#                     self.objects[track_idx].age = 0
#                     detection_indices.discard(det_idx)
#                     track_indices.discard(track_idx)
                
#                 # Step 4: Create new tracks for unmatched detections
#                 for det_idx in detection_indices:
#                     self.objects.append(Track(detections[det_idx], self.next_id))
#                     self.next_id += 1
                
#                 # Step 5: Age unmatched tracks
#                 for track_idx in track_indices:
#                     self.objects[track_idx].age += 1
                    
#                     # Terminate if too old
#                     if self.objects[track_idx].age > self.max_age:
#                         self.objects[track_idx].is_terminated = True
#         else:
#             # No detections, age all tracks
#             for obj in self.objects:
#                 obj.age += 1
#                 if obj.age > self.max_age:
#                     obj.is_terminated = True
        
#         # Return active tracks
#         active_tracks = [
#             obj.get_state()
#             for obj in self.objects
#             if not obj.is_terminated
#         ]
        
#         return active_tracks
    
#     def get_tracks(self) -> List[Tuple[int, List[Point], List[BBox], List[int]]]:
#         """Get all track trajectories.
        
#         Returns:
#             List of trajectories, each containing:
#                 (track_id, centroids, bboxes, frame_numbers)
            
#         Example:
#             >>> trajectories = tracker.get_tracks()
#             >>> for track_id, centroids, bboxes, frames in trajectories:
#             ...     print(f"Track {track_id}: {len(centroids)} positions")
#         """
#         return [obj.get_trajectory() for obj in self.objects]
    
#     def get_live_track_count(self) -> int:
#         """Get number of currently active tracks.
        
#         Returns:
#             Count of non-terminated tracks
#         """
#         return len([obj for obj in self.objects if not obj.is_terminated])
    
#     def __repr__(self) -> str:
#         """String representation of tracker."""
#         live_count = self.get_live_track_count()
#         total_count = len(self.objects)
#         return f"BeeTracker(live={live_count}, total={total_count}, next_id={self.next_id})"












# """Bee tracker with multi-species support.

# This module provides tracking functionality for multiple insect species.
# """

# import numpy as np
# from typing import List, Tuple, Optional, Dict
# from scipy.optimize import linear_sum_assignment


# def compute_centroid(bbox):
#     """Compute centroid of a bounding box.
    
#     Args:
#         bbox: Bounding box in format (x1, y1, x2, y2)
        
#     Returns:
#         Centroid as (x, y) tuple
#     """
#     x1, y1, x2, y2 = bbox
#     return ((x1 + x2) / 2, (y1 + y2) / 2)


# def predict_position(D_1, D_2, distance_threshold=50):
#     """Predict the position of an insect in the next frame.
    
#     Uses linear motion model based on previous two positions.
    
#     Args:
#         D_1: Last detected position (x1, y1, x2, y2)
#         D_2: Second last detected position (x1, y1, x2, y2)
#         distance_threshold: Maximum prediction distance
        
#     Returns:
#         Predicted position (x1, y1, x2, y2)
#     """
#     height = 20
#     width = 20
    
#     # Compute centroids
#     D_k_1 = compute_centroid(D_1)
#     D_k_2 = compute_centroid(D_2)
    
#     # Prediction matrix
#     A = np.array([
#         [2, 0, -1, 0],
#         [0, 2, 0, -1]
#     ])
    
#     # Position vector
#     D_vector = np.array([D_k_1[0], D_k_1[1], D_k_2[0], D_k_2[1]])
    
#     # Predict
#     predicted_position = np.dot(A, D_vector)
    
#     predict_bbox = (
#         predicted_position[0] - width,
#         predicted_position[1] - height,
#         predicted_position[0] + width,
#         predicted_position[1] + height
#     )
    
#     # Check if prediction is reasonable
#     distance = np.linalg.norm(np.array(D_k_1) - np.array(predicted_position))
    
#     if distance > distance_threshold:
#         return D_1
#     else:
#         return predict_bbox


# def Hungarian_algorithm(tracked_objects, detections, threshold=200):
#     """Hungarian algorithm for optimal detection-track assignment.
    
#     Args:
#         tracked_objects: List of predicted track states
#         detections: List of new detections
#         threshold: Maximum distance for valid association
        
#     Returns:
#         List of (detection_idx, track_idx) associations
#     """
#     if len(tracked_objects) == 0 or len(detections) == 0:
#         return []
    
#     # Cost matrix
#     cost_matrix = np.zeros((len(tracked_objects), len(detections)))
    
#     for i, track in enumerate(tracked_objects):
#         for j, det in enumerate(detections):
#             track_centroid = compute_centroid(track[0])
#             det_centroid = compute_centroid(det)
#             distance = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))
#             cost_matrix[i, j] = distance
    
#     # Solve assignment problem
#     track_indices, det_indices = linear_sum_assignment(cost_matrix)
    
#     associations = []
#     for track_index, det_index in zip(track_indices, det_indices):
#         if cost_matrix[track_index, det_index] < threshold:
#             associations.append((det_index, track_index))
    
#     return associations


# class Track:
#     """Individual track with species information.
    
#     Represents a single insect's trajectory across frames with species tracking.
#     """
    
#     def __init__(self, bbox, track_id, species: Optional[int] = None):
#         """Initialize track.
        
#         Args:
#             bbox: Initial bounding box
#             track_id: Unique track identifier
#             species: Species class ID (optional)
#         """
#         self.bbox = bbox
#         self.track_id = track_id
#         self.age = 0
#         self.dead = False
#         self.trajectory = []
#         self.trajectory_frame_numbers = []
#         self.is_DL_predictions = []
        
#         # Species tracking
#         self.species = species
#         self.species_votes = {}  # Track species observations
#         if species is not None:
#             self.species_votes[species] = 1
    
#     def predict(self, distance_threshold):
#         """Predict next position based on trajectory."""
#         if len(self.trajectory) > 1:
#             self.bbox = predict_position(
#                 self.trajectory[-1],
#                 self.trajectory[-2],
#                 distance_threshold
#             )
#         else:
#             self.bbox = predict_position(self.bbox, self.bbox, distance_threshold)
    
#     def update(self, bbox, frame_number, is_DL_prediction=False, species: Optional[int] = None):
#         """Update track with new detection.
        
#         Args:
#             bbox: New bounding box
#             frame_number: Current frame number
#             is_DL_prediction: Whether this is from detection (vs prediction)
#             species: Species class ID for this detection
#         """
#         if not self.dead:
#             self.bbox = bbox
#             self.trajectory.append(bbox)
#             self.trajectory_frame_numbers.append(frame_number)
#             self.age = 0
#             self.is_DL_predictions.append(is_DL_prediction)
            
#             # Update species information
#             if species is not None:
#                 if species in self.species_votes:
#                     self.species_votes[species] += 1
#                 else:
#                     self.species_votes[species] = 1
                
#                 # Update primary species based on majority vote
#                 self.species = max(self.species_votes, key=self.species_votes.get)
    
#     def get_state(self):
#         """Get current track state.
        
#         Returns:
#             Tuple of (bbox, track_id, species)
#         """
#         return self.bbox, self.track_id, self.species
    
#     def get_trajectory(self):
#         """Get track trajectory with species information.
        
#         Returns:
#             Tuple of (track_id, centroids, bboxes, frame_numbers, species, species_votes)
#         """
#         # Only keep DL predictions
#         trajectory = [
#             self.trajectory[i]
#             for i in range(len(self.trajectory))
#             if self.is_DL_predictions[i]
#         ]
#         trajectory_centroids = [compute_centroid(bbox) for bbox in trajectory]
#         trajectory_frame_numbers = [
#             self.trajectory_frame_numbers[i]
#             for i in range(len(self.trajectory_frame_numbers))
#             if self.is_DL_predictions[i]
#         ]
        
#         return (
#             self.track_id,
#             trajectory_centroids,
#             trajectory,
#             trajectory_frame_numbers,
#             self.species,
#             dict(self.species_votes)
#         )


# class BeeTracker:
#     """Multi-species insect tracker.
    
#     Tracks multiple insect species across video frames using
#     Hungarian algorithm for data association.
    
#     Attributes:
#         max_age: Maximum frames to keep track alive without detection
#         objects: List of active Track objects
#         next_id: Next available track ID
#         track_species: Whether to track species information
    
#     Example:
#         >>> tracker = BeeTracker(max_age=30, track_species=True)
#         >>> tracks = tracker.update(detections, frame_num, species_labels)
#     """
    
#     def __init__(
#         self,
#         max_age: int = 30,
#         track_start_id: int = 0,
#         distance_threshold: float = 100,
#         association_threshold: float = 200,
#         track_species: bool = False
#     ):
#         """Initialize tracker.
        
#         Args:
#             max_age: Maximum frames without detection before terminating track
#             track_start_id: Starting track ID
#             distance_threshold: Maximum distance for motion prediction
#             association_threshold: Maximum distance for detection association
#             track_species: Whether to track species information
#         """
#         self.max_age = max_age
#         self.objects = []
#         self.next_id = track_start_id
#         self.association_threshold = association_threshold
#         self.distance_threshold = distance_threshold
#         self.track_species = track_species
    
#     def get_tracks(self):
#         """Get all track trajectories with species information.
        
#         Returns:
#             List of track trajectories, each containing:
#             (track_id, centroids, bboxes, frame_numbers, species, species_votes)
#         """
#         return [obj.get_trajectory() for obj in self.objects]
    
#     def get_num_live_tracks(self):
#         """Get number of active tracks.
        
#         Returns:
#             Number of live tracks
#         """
#         return len([obj for obj in self.objects if not obj.dead])
    
#     def update(
#         self,
#         detections: List[Tuple],
#         frame_number: int,
#         species_labels: Optional[List[int]] = None
#     ) -> List[Tuple]:
#         """Update tracker with new detections.
        
#         Args:
#             detections: List of bounding boxes
#             frame_number: Current frame number
#             species_labels: List of species class IDs (one per detection)
            
#         Returns:
#             List of active track states: (bbox, track_id, species)
#         """
#         # Predict state of all existing tracks
#         for obj in self.objects:
#             obj.predict(distance_threshold=self.distance_threshold)
        
#         # Associate detections with existing tracks
#         if len(detections) > 0:
#             if len(self.objects) == 0:
#                 # No existing tracks, create new ones
#                 for i, det in enumerate(detections):
#                     species = species_labels[i] if species_labels and self.track_species else None
#                     self.objects.append(Track(det, self.next_id, species))
#                     self.next_id += 1
#             else:
#                 # Use Hungarian algorithm for association
#                 associations = Hungarian_algorithm(
#                     [obj.get_state() for obj in self.objects],
#                     detections,
#                     threshold=self.association_threshold
#                 )
                
#                 detections_idx = set(range(len(detections)))
#                 tracks_idx = set(range(len(self.objects)))
                
#                 # Update associated tracks
#                 for det_idx, track_idx in associations:
#                     species = species_labels[det_idx] if species_labels and self.track_species else None
#                     self.objects[track_idx].update(
#                         detections[det_idx],
#                         frame_number,
#                         is_DL_prediction=True,
#                         species=species
#                     )
#                     self.objects[track_idx].age = 0
#                     detections_idx.remove(det_idx)
#                     try:
#                         tracks_idx.remove(track_idx)
#                     except KeyError:
#                         pass
                
#                 # Create new tracks for unmatched detections
#                 for det_idx in detections_idx:
#                     species = species_labels[det_idx] if species_labels and self.track_species else None
#                     self.objects.append(Track(detections[det_idx], self.next_id, species))
#                     self.next_id += 1
                
#                 # Age unmatched tracks
#                 for track_idx in tracks_idx:
#                     self.objects[track_idx].age += 1
#                     if self.objects[track_idx].age > self.max_age:
#                         self.objects[track_idx].dead = True
#         else:
#             # No detections, age all tracks
#             for obj in self.objects:
#                 obj.update(obj.bbox, frame_number, is_DL_prediction=False)
#                 obj.age += 1
#                 if obj.age > self.max_age:
#                     obj.dead = True
        
#         # Return live tracks
#         live_objects = [obj for obj in self.objects if not obj.dead]
#         return [obj.get_state() for obj in live_objects]








"""Bee tracker with multi-species support.

This module provides tracking functionality for multiple insect species.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.optimize import linear_sum_assignment


def compute_centroid(bbox):
    """Compute centroid of a bounding box.
    
    Args:
        bbox: Bounding box in format (x1, y1, x2, y2)
        
    Returns:
        Centroid as (x, y) tuple
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def predict_position(D_1, D_2, distance_threshold=50):
    """Predict the position of an insect in the next frame.
    
    Uses linear motion model based on previous two positions.
    
    Args:
        D_1: Last detected position (x1, y1, x2, y2)
        D_2: Second last detected position (x1, y1, x2, y2)
        distance_threshold: Maximum prediction distance
        
    Returns:
        Predicted position (x1, y1, x2, y2)
    """
    height = 20
    width = 20
    
    # Compute centroids
    D_k_1 = compute_centroid(D_1)
    D_k_2 = compute_centroid(D_2)
    
    # Prediction matrix
    A = np.array([
        [2, 0, -1, 0],
        [0, 2, 0, -1]
    ])
    
    # Position vector
    D_vector = np.array([D_k_1[0], D_k_1[1], D_k_2[0], D_k_2[1]])
    
    # Predict
    predicted_position = np.dot(A, D_vector)
    
    predict_bbox = (
        predicted_position[0] - width,
        predicted_position[1] - height,
        predicted_position[0] + width,
        predicted_position[1] + height
    )
    
    # Check if prediction is reasonable
    distance = np.linalg.norm(np.array(D_k_1) - np.array(predicted_position))
    
    if distance > distance_threshold:
        return D_1
    else:
        return predict_bbox


def Hungarian_algorithm(tracked_objects, detections, threshold=200):
    """Hungarian algorithm for optimal detection-track assignment.
    
    Args:
        tracked_objects: List of predicted track states
        detections: List of new detections
        threshold: Maximum distance for valid association
        
    Returns:
        List of (detection_idx, track_idx) associations
    """
    if len(tracked_objects) == 0 or len(detections) == 0:
        return []
    
    # Cost matrix
    cost_matrix = np.zeros((len(tracked_objects), len(detections)))
    
    for i, track in enumerate(tracked_objects):
        for j, det in enumerate(detections):
            track_centroid = compute_centroid(track[0])
            det_centroid = compute_centroid(det)
            distance = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))
            cost_matrix[i, j] = distance
    
    # Solve assignment problem
    track_indices, det_indices = linear_sum_assignment(cost_matrix)
    
    associations = []
    for track_index, det_index in zip(track_indices, det_indices):
        if cost_matrix[track_index, det_index] < threshold:
            associations.append((det_index, track_index))
    
    return associations


class Track:
    """Individual track with species information.
    
    Represents a single insect's trajectory across frames with species tracking.
    """
    
    def __init__(self, bbox, track_id, species: Optional[int] = None):
        """Initialize track.
        
        Args:
            bbox: Initial bounding box
            track_id: Unique track identifier
            species: Species class ID (optional)
        """
        self.bbox = bbox
        self.track_id = track_id
        self.age = 0
        self.dead = False
        self.trajectory = []
        self.trajectory_frame_numbers = []
        self.is_DL_predictions = []
        
        # Species tracking
        self.species = species
        self.species_votes = {}  # Track species observations
        if species is not None:
            self.species_votes[species] = 1
    
    def predict(self, distance_threshold):
        """Predict next position based on trajectory."""
        if len(self.trajectory) > 1:
            self.bbox = predict_position(
                self.trajectory[-1],
                self.trajectory[-2],
                distance_threshold
            )
        else:
            self.bbox = predict_position(self.bbox, self.bbox, distance_threshold)
    
    def update(self, bbox, frame_number, is_DL_prediction=False, species: Optional[int] = None):
        """Update track with new detection.
        
        Args:
            bbox: New bounding box
            frame_number: Current frame number
            is_DL_prediction: Whether this is from detection (vs prediction)
            species: Species class ID for this detection
        """
        if not self.dead:
            self.bbox = bbox
            self.trajectory.append(bbox)
            self.trajectory_frame_numbers.append(frame_number)
            self.age = 0
            self.is_DL_predictions.append(is_DL_prediction)
            
            # Update species information
            if species is not None:
                if species in self.species_votes:
                    self.species_votes[species] += 1
                else:
                    self.species_votes[species] = 1
                
                # Update primary species based on majority vote
                self.species = max(self.species_votes, key=self.species_votes.get)
    
    def get_state(self):
        """Get current track state.
        
        Returns:
            Tuple of (bbox, track_id, species)
        """
        return self.bbox, self.track_id, self.species
    
    def get_trajectory(self):
        """Get track trajectory with species information.
        
        Returns:
            Tuple of (track_id, centroids, bboxes, frame_numbers, species, species_votes)
        """
        # Only keep DL predictions
        trajectory = [
            self.trajectory[i]
            for i in range(len(self.trajectory))
            if self.is_DL_predictions[i]
        ]
        trajectory_centroids = [compute_centroid(bbox) for bbox in trajectory]
        trajectory_frame_numbers = [
            self.trajectory_frame_numbers[i]
            for i in range(len(self.trajectory_frame_numbers))
            if self.is_DL_predictions[i]
        ]
        
        return (
            self.track_id,
            trajectory_centroids,
            trajectory,
            trajectory_frame_numbers,
            self.species,
            dict(self.species_votes)
        )


class BeeTracker:
    """Multi-species insect tracker.
    
    Tracks multiple insect species across video frames using
    Hungarian algorithm for data association.
    
    Attributes:
        max_age: Maximum frames to keep track alive without detection
        objects: List of active Track objects
        next_id: Next available track ID
        track_species: Whether to track species information
    
    Example:
        >>> tracker = BeeTracker(max_age=30, track_species=True)
        >>> tracks = tracker.update(detections, frame_num, species_labels)
    """
    
    def __init__(
        self,
        max_age: int = 30,
        track_start_id: int = 0,
        distance_threshold: float = 100,
        association_threshold: float = 200,
        track_species: bool = False
    ):
        """Initialize tracker.
        
        Args:
            max_age: Maximum frames without detection before terminating track
            track_start_id: Starting track ID
            distance_threshold: Maximum distance for motion prediction
            association_threshold: Maximum distance for detection association
            track_species: Whether to track species information
        """
        self.max_age = max_age
        self.objects = []
        self.next_id = track_start_id
        self.association_threshold = association_threshold
        self.distance_threshold = distance_threshold
        self.track_species = track_species
    
    def get_tracks(self):
        """Get all track trajectories with species information.
        
        Returns:
            List of track trajectories, each containing:
            (track_id, centroids, bboxes, frame_numbers, species, species_votes)
        """
        return [obj.get_trajectory() for obj in self.objects]
    
    def get_num_live_tracks(self):
        """Get number of active tracks.
        
        Returns:
            Number of live tracks
        """
        return len([obj for obj in self.objects if not obj.dead])
    
    def update(
        self,
        detections: List[Tuple],
        frame_number: int,
        species_labels: Optional[List[int]] = None
    ) -> List[Tuple]:
        """Update tracker with new detections.
        
        Args:
            detections: List of bounding boxes
            frame_number: Current frame number
            species_labels: List of species class IDs (one per detection)
            
        Returns:
            List of active track states: (bbox, track_id, species)
        """
        # Predict state of all existing tracks
        for obj in self.objects:
            obj.predict(distance_threshold=self.distance_threshold)
        
        # Associate detections with existing tracks
        if len(detections) > 0:
            if len(self.objects) == 0:
                # No existing tracks, create new ones
                for i, det in enumerate(detections):
                    species = species_labels[i] if species_labels and self.track_species else None
                    self.objects.append(Track(det, self.next_id, species))
                    self.next_id += 1
            else:
                # Use Hungarian algorithm for association
                associations = Hungarian_algorithm(
                    [obj.get_state() for obj in self.objects],
                    detections,
                    threshold=self.association_threshold
                )
                
                detections_idx = set(range(len(detections)))
                tracks_idx = set(range(len(self.objects)))
                
                # Update associated tracks
                for det_idx, track_idx in associations:
                    species = species_labels[det_idx] if species_labels and self.track_species else None
                    self.objects[track_idx].update(
                        detections[det_idx],
                        frame_number,
                        is_DL_prediction=True,
                        species=species
                    )
                    self.objects[track_idx].age = 0
                    detections_idx.remove(det_idx)
                    try:
                        tracks_idx.remove(track_idx)
                    except KeyError:
                        pass
                
                # Create new tracks for unmatched detections
                for det_idx in detections_idx:
                    species = species_labels[det_idx] if species_labels and self.track_species else None
                    self.objects.append(Track(detections[det_idx], self.next_id, species))
                    self.next_id += 1
                
                # Age unmatched tracks
                for track_idx in tracks_idx:
                    self.objects[track_idx].age += 1
                    if self.objects[track_idx].age > self.max_age:
                        self.objects[track_idx].dead = True
        else:
            # No detections, age all tracks
            for obj in self.objects:
                obj.update(obj.bbox, frame_number, is_DL_prediction=False)
                obj.age += 1
                if obj.age > self.max_age:
                    obj.dead = True
        
        # Return live tracks
        live_objects = [obj for obj in self.objects if not obj.dead]
        return [obj.get_state() for obj in live_objects]