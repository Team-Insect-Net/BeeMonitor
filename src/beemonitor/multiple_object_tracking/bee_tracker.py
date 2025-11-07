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
        tracked_objects: List of track state dicts with 'bbox' key
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
        # Handle both dict and tuple formats for backward compatibility
        if isinstance(track, dict):
            track_bbox = track['bbox']
        else:
            track_bbox = track[0]  # Old tuple format
            
        for j, det in enumerate(detections):
            track_centroid = compute_centroid(track_bbox)
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
            self.is_DL_predictions.append(is_DL_prediction)
            
            # Only reset age if it's an actual detection
            if is_DL_prediction:
                self.age = 0
            
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
            Dict with bbox, track_id, and species
        """
        return {
            'bbox': self.bbox,
            'track_id': self.track_id,
            'species': self.species
        }
    
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
    
    def get_active_tracks(self):
        """Get current state of all active tracks.
        
        Returns:
            List of dicts with bbox, track_id, species for active tracks
        """
        return [obj.get_state() for obj in self.objects if not obj.dead]
    
    def update(
        self,
        detections: List[Tuple],
        frame_number: int,
        species_labels: Optional[List[int]] = None
    ) -> List[dict]:
        """Update tracker with new detections.
        
        Args:
            detections: List of bounding boxes
            frame_number: Current frame number
            species_labels: List of species class IDs (one per detection)
            
        Returns:
            List of active track states as dicts
        """
        # Remove dead tracks before processing
        self.objects = [obj for obj in self.objects if not obj.dead]
        
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