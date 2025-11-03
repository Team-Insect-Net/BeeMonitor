"""Event processing for bee tracking data.

This module processes bee trajectories to identify entry and exit events
at nest holes.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

from bee_monitor.core.config import Config
from bee_monitor.processing.trajectory_analyzer import TrajectoryAnalyzer


logger = logging.getLogger(__name__)

# Type aliases
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class EventProcessor:
    """Processor for identifying bee entry/exit events.
    
    This class analyzes bee trajectories to determine when bees enter
    or exit nest holes, creating a timeline of activity events.
    
    Attributes:
        config: Configuration object
        trajectory_analyzer: TrajectoryAnalyzer instance
    
    Example:
        >>> processor = EventProcessor(config)
        >>> events = processor.process_tracks(motion_data, nests)
        >>> print(f"Found {len(events)} events")
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize EventProcessor.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config if config is not None else Config.default()
        self.trajectory_analyzer = TrajectoryAnalyzer(self.config)
    
    def process_tracks(
        self,
        motion_data: pd.DataFrame,
        nests: Dict
    ) -> pd.DataFrame:
        """Process tracking data to identify entry/exit events.
        
        Args:
            motion_data: DataFrame with columns: frame_number, tracks, detections
            nests: Dictionary with 'hotel' ROI and 'nests' mapping
            
        Returns:
            DataFrame with columns: action, nest, frame_number, notes
            
        Example:
            >>> events = processor.process_tracks(motion_data, nests)
            >>> entries = events[events['action'] == 'Entry']
            >>> print(f"Found {len(entries)} entry events")
        """
        logger.info("Processing tracks to identify events...")
        
        # Extract all movements from tracking data
        movements = []
        for period in motion_data.tracks:
            for track in period:
                movements.append(track)
        
        logger.debug(f"Processing {len(movements)} trajectories")
        
        # Process each movement to identify events
        actions = []
        for movement in movements:
            # Skip short trajectories
            if len(movement[1]) < self.config.processing.min_trajectory_length:
                continue
            
            # Classify movement type
            if self.trajectory_analyzer.is_exit_behavior(movement):
                action = self._get_action(
                    movement,
                    nests,
                    window_size=self.config.processing.exit_window_size,
                    padding=self.config.processing.exit_padding
                )
            elif self.trajectory_analyzer.is_entry_behavior(movement):
                action = self._get_action(
                    movement,
                    nests,
                    window_size=self.config.processing.entry_window_size,
                    padding=self.config.processing.entry_padding
                )
            else:
                # Not clearly entry or exit, skip
                continue
            
            # Add actions to list
            if action:
                if isinstance(action, list):
                    actions.extend(action)
                else:
                    actions.append(action)
        
        logger.info(f"Identified {len(actions)} events")
        
        # Convert to DataFrame
        if actions:
            return pd.DataFrame(actions)
        else:
            return pd.DataFrame(columns=['action', 'nest', 'frame_number', 'notes'])
    
    def _get_action(
        self,
        movement: Tuple,
        nests: Dict,
        window_size: int = 3,
        padding: int = 20
    ) -> Optional[Union[Dict, List[Dict]]]:
        """Determine action (entry/exit) from movement trajectory.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
            nests: Dictionary with nest locations
            window_size: Number of frames to analyze
            padding: Padding around nest boxes
            
        Returns:
            Dictionary or list of dictionaries with action details, or None
        """
        start_id, end_id = self._detect_entry_exit(
            movement[1],  # centroids
            nests['nests'],
            window_size=window_size,
            padding=padding
        )
        
        if start_id == -1 and end_id == -1:
            return None
        
        elif start_id != -1 and end_id == -1:
            # Exit only
            return {
                "action": "Exit",
                "nest": str(start_id),
                "frame_number": movement[3][0],  # First frame
                "notes": "Bee exited the nest"
            }
        
        elif start_id == -1 and end_id != -1:
            # Entry only
            return {
                "action": "Entry",
                "nest": str(end_id),
                "frame_number": movement[3][-1],  # Last frame
                "notes": "Bee entered the nest"
            }
        
        elif start_id != -1 and end_id != -1:
            # Both entry and exit (nest-to-nest movement)
            return [
                {
                    "action": "Exit",
                    "nest": str(start_id),
                    "frame_number": movement[3][0],
                    "notes": f"Bee exited the nest to move to another hole {end_id}"
                },
                {
                    "action": "Entry",
                    "nest": str(end_id),
                    "frame_number": movement[3][-1],
                    "notes": f"Bee entered the nest from another hole {start_id}"
                }
            ]
        
        return None
    
    def _detect_entry_exit(
        self,
        bee_trajectory: List[Point],
        hole_bboxes: Dict[str, BBox],
        window_size: int = 3,
        padding: int = 20
    ) -> Tuple[int, int]:
        """Detect if bee enters or exits a hole.
        
        Analyzes the start and end of a trajectory to determine if the bee
        started inside a hole (exit) or ended inside a hole (entry).
        
        Args:
            bee_trajectory: List of (x, y) positions
            hole_bboxes: Dictionary mapping hole IDs to bounding boxes
            window_size: Number of frames to analyze at start/end
            padding: Padding to add around nest boxes
            
        Returns:
            Tuple of (start_hole_id, end_hole_id), -1 if not in any hole
        """
        if len(bee_trajectory) < window_size:
            window_size = max(1, len(bee_trajectory) // 2)
        
        # Analyze start of trajectory
        start_trajectory = bee_trajectory[:window_size]
        start_id = -1
        
        for hole_id, bbox in hole_bboxes.items():
            # Check if all positions in start window are inside this hole
            start_inside = all(
                self._is_inside_bbox(pos, bbox, padding)
                for pos in start_trajectory
            )
            
            if start_inside:
                start_id = hole_id
                break
        
        # Analyze end of trajectory
        end_trajectory = bee_trajectory[-window_size:]
        end_id = -1
        
        for hole_id, bbox in hole_bboxes.items():
            # Check if all positions in end window are inside this hole
            end_inside = all(
                self._is_inside_bbox(pos, bbox, padding)
                for pos in end_trajectory
            )
            
            if end_inside:
                end_id = hole_id
                break
        
        return start_id, end_id
    
    def _is_inside_bbox(
        self,
        bee_position: Point,
        bbox: BBox,
        padding: int = 20
    ) -> bool:
        """Check if a position is inside a bounding box with padding.
        
        Args:
            bee_position: (x, y) coordinates
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            padding: Padding to add around box
            
        Returns:
            True if position is inside padded box
        """
        x, y = bee_position
        x_min, y_min, x_max, y_max = bbox
        
        # Add padding with slightly more vertical padding
        x_min -= padding
        y_min -= int(padding + padding / 2)
        x_max += padding
        y_max += int(padding + padding / 2)
        
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def detect_entry(
        self,
        bee_trajectory: List[Point],
        hole_bboxes: Dict[str, BBox],
        window_size: int = 3,
        padding: int = 20
    ) -> int:
        """Detect if bee enters a hole (analyze start of trajectory).
        
        Args:
            bee_trajectory: List of (x, y) positions
            hole_bboxes: Dictionary mapping hole IDs to bounding boxes
            window_size: Number of frames to analyze
            padding: Padding around boxes
            
        Returns:
            Hole ID if entry detected, -1 otherwise
        """
        if len(bee_trajectory) < window_size:
            window_size = max(1, len(bee_trajectory) // 2)
        
        start_trajectory = bee_trajectory[:window_size]
        
        for hole_id, bbox in hole_bboxes.items():
            start_inside = all(
                self._is_inside_bbox(pos, bbox, padding)
                for pos in start_trajectory
            )
            
            if start_inside:
                return hole_id
        
        return -1
    
    def detect_exit(
        self,
        bee_trajectory: List[Point],
        hole_bboxes: Dict[str, BBox],
        window_size: int = 3,
        padding: int = 20
    ) -> int:
        """Detect if bee exits a hole (analyze end of trajectory).
        
        Args:
            bee_trajectory: List of (x, y) positions
            hole_bboxes: Dictionary mapping hole IDs to bounding boxes
            window_size: Number of frames to analyze
            padding: Padding around boxes
            
        Returns:
            Hole ID if exit detected, -1 otherwise
        """
        if len(bee_trajectory) < window_size:
            window_size = max(1, len(bee_trajectory) // 2)
        
        end_trajectory = bee_trajectory[-window_size:]
        
        for hole_id, bbox in hole_bboxes.items():
            end_inside = all(
                self._is_inside_bbox(pos, bbox, padding)
                for pos in end_trajectory
            )
            
            if end_inside:
                return hole_id
        
        return -1
    
    def process_yolo_tracks(
        self,
        movements: List[Tuple],
        nests: Dict
    ) -> pd.DataFrame:
        """Process YOLO tracking results to identify events.
        
        This is an alternative processing method for trajectories from
        Ultralytics YOLO tracking rather than custom BeeTracker.
        
        Args:
            movements: List of trajectories from UltralyticsTracker
            nests: Dictionary with nest locations
            
        Returns:
            DataFrame with events
            
        Example:
            >>> from bee_monitor.tracking import UltralyticsTracker
            >>> tracker = UltralyticsTracker(model)
            >>> trajectories = tracker.get_tracks("video.mp4")
            >>> events = processor.process_yolo_tracks(trajectories, nests)
        """
        logger.info("Processing YOLO tracks to identify events...")
        
        actions = []
        for movement in movements:
            # Skip short trajectories
            if len(movement[1]) < self.config.processing.min_trajectory_length:
                continue
            
            # Classify movement type
            if self.trajectory_analyzer.is_exit_behavior(movement):
                action = self._get_action(
                    movement,
                    nests,
                    window_size=self.config.processing.exit_window_size,
                    padding=self.config.processing.exit_padding
                )
            elif self.trajectory_analyzer.is_entry_behavior(movement):
                action = self._get_action(
                    movement,
                    nests,
                    window_size=self.config.processing.entry_window_size,
                    padding=self.config.processing.entry_padding
                )
            else:
                continue
            
            # Add actions to list
            if action:
                if isinstance(action, list):
                    actions.extend(action)
                else:
                    actions.append(action)
        
        logger.info(f"Identified {len(actions)} events from YOLO tracks")
        
        if actions:
            return pd.DataFrame(actions)
        else:
            return pd.DataFrame(columns=['action', 'nest', 'frame_number', 'notes'])
    
    def __repr__(self) -> str:
        """String representation of processor."""
        return f"EventProcessor(config={self.config is not None})"