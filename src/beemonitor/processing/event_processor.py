# """Event processing for bee tracking data.

# This module processes bee trajectories to identify entry and exit events
# at nest holes.
# """

# import logging
# from typing import Dict, List, Tuple, Optional, Union
# import numpy as np
# import pandas as pd

# from beemonitor.core.config import Config
# from beemonitor.processing.trajectory_analyzer import TrajectoryAnalyzer


# logger = logging.getLogger(__name__)

# # Type aliases
# Point = Tuple[float, float]
# BBox = Tuple[float, float, float, float]


# class EventProcessor:
#     """Processor for identifying bee entry/exit events.
    
#     This class analyzes bee trajectories to determine when bees enter
#     or exit nest holes, creating a timeline of activity events.
    
#     Attributes:
#         config: Configuration object
#         trajectory_analyzer: TrajectoryAnalyzer instance
    
#     Example:
#         >>> processor = EventProcessor(config)
#         >>> events = processor.process_tracks(motion_data, nests)
#         >>> print(f"Found {len(events)} events")
#     """
    
#     def __init__(self, config: Optional[Config] = None):
#         """Initialize EventProcessor.
        
#         Args:
#             config: Configuration object (optional)
#         """
#         self.config = config if config is not None else Config.default()
#         self.trajectory_analyzer = TrajectoryAnalyzer(self.config)
    
#     def process_tracks(
#         self,
#         motion_data: pd.DataFrame,
#         nests: Dict
#     ) -> pd.DataFrame:
#         """Process tracking data to identify entry/exit events.
        
#         Args:
#             motion_data: DataFrame with columns: frame_number, tracks, detections
#             nests: Dictionary with 'hotel' ROI and 'nests' mapping
            
#         Returns:
#             DataFrame with columns: action, nest, frame_number, notes
            
#         Example:
#             >>> events = processor.process_tracks(motion_data, nests)
#             >>> entries = events[events['action'] == 'Entry']
#             >>> print(f"Found {len(entries)} entry events")
#         """
#         logger.info("Processing tracks to identify events...")
        
#         # Extract all movements from tracking data
#         movements = []
#         for period in motion_data.tracks:
#             for track in period:
#                 movements.append(track)
        
#         logger.debug(f"Processing {len(movements)} trajectories")
        
#         # Process each movement to identify events
#         actions = []
#         for movement in movements:
#             # Skip short trajectories
#             if len(movement[1]) < self.config.processing.min_trajectory_length:
#                 continue
            
#             # Classify movement type
#             if self.trajectory_analyzer.is_exit_behavior(movement):
#                 action = self._get_action(
#                     movement,
#                     nests,
#                     window_size=self.config.processing.exit_window_size,
#                     padding=self.config.processing.exit_padding
#                 )
#             elif self.trajectory_analyzer.is_entry_behavior(movement):
#                 action = self._get_action(
#                     movement,
#                     nests,
#                     window_size=self.config.processing.entry_window_size,
#                     padding=self.config.processing.entry_padding
#                 )
#             else:
#                 # Not clearly entry or exit, skip
#                 continue
            
#             # Add actions to list
#             if action:
#                 if isinstance(action, list):
#                     actions.extend(action)
#                 else:
#                     actions.append(action)
        
#         logger.info(f"Identified {len(actions)} events")
        
#         # Convert to DataFrame
#         if actions:
#             return pd.DataFrame(actions)
#         else:
#             return pd.DataFrame(columns=['action', 'nest', 'frame_number', 'notes'])
    
#     def _get_action(
#         self,
#         movement: Tuple,
#         nests: Dict,
#         window_size: int = 3,
#         padding: int = 20
#     ) -> Optional[Union[Dict, List[Dict]]]:
#         """Determine action (entry/exit) from movement trajectory.
        
#         Args:
#             movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
#             nests: Dictionary with nest locations
#             window_size: Number of frames to analyze
#             padding: Padding around nest boxes
            
#         Returns:
#             Dictionary or list of dictionaries with action details, or None
#         """
#         start_id, end_id = self._detect_entry_exit(
#             movement[1],  # centroids
#             nests['nests'],
#             window_size=window_size,
#             padding=padding
#         )
        
#         if start_id == -1 and end_id == -1:
#             return None
        
#         elif start_id != -1 and end_id == -1:
#             # Exit only
#             return {
#                 "action": "Exit",
#                 "nest": str(start_id),
#                 "frame_number": movement[3][0],  # First frame
#                 "notes": "Bee exited the nest"
#             }
        
#         elif start_id == -1 and end_id != -1:
#             # Entry only
#             return {
#                 "action": "Entry",
#                 "nest": str(end_id),
#                 "frame_number": movement[3][-1],  # Last frame
#                 "notes": "Bee entered the nest"
#             }
        
#         elif start_id != -1 and end_id != -1:
#             # Both entry and exit (nest-to-nest movement)
#             return [
#                 {
#                     "action": "Exit",
#                     "nest": str(start_id),
#                     "frame_number": movement[3][0],
#                     "notes": f"Bee exited the nest to move to another hole {end_id}"
#                 },
#                 {
#                     "action": "Entry",
#                     "nest": str(end_id),
#                     "frame_number": movement[3][-1],
#                     "notes": f"Bee entered the nest from another hole {start_id}"
#                 }
#             ]
        
#         return None
    
#     def _detect_entry_exit(
#         self,
#         bee_trajectory: List[Point],
#         hole_bboxes: Dict[str, BBox],
#         window_size: int = 3,
#         padding: int = 20
#     ) -> Tuple[int, int]:
#         """Detect if bee enters or exits a hole.
        
#         Analyzes the start and end of a trajectory to determine if the bee
#         started inside a hole (exit) or ended inside a hole (entry).
        
#         Args:
#             bee_trajectory: List of (x, y) positions
#             hole_bboxes: Dictionary mapping hole IDs to bounding boxes
#             window_size: Number of frames to analyze at start/end
#             padding: Padding to add around nest boxes
            
#         Returns:
#             Tuple of (start_hole_id, end_hole_id), -1 if not in any hole
#         """
#         if len(bee_trajectory) < window_size:
#             window_size = max(1, len(bee_trajectory) // 2)
        
#         # Analyze start of trajectory
#         start_trajectory = bee_trajectory[:window_size]
#         start_id = -1
        
#         for hole_id, bbox in hole_bboxes.items():
#             # Check if all positions in start window are inside this hole
#             start_inside = all(
#                 self._is_inside_bbox(pos, bbox, padding)
#                 for pos in start_trajectory
#             )
            
#             if start_inside:
#                 start_id = hole_id
#                 break
        
#         # Analyze end of trajectory
#         end_trajectory = bee_trajectory[-window_size:]
#         end_id = -1
        
#         for hole_id, bbox in hole_bboxes.items():
#             # Check if all positions in end window are inside this hole
#             end_inside = all(
#                 self._is_inside_bbox(pos, bbox, padding)
#                 for pos in end_trajectory
#             )
            
#             if end_inside:
#                 end_id = hole_id
#                 break
        
#         return start_id, end_id
    
#     def _is_inside_bbox(
#         self,
#         bee_position: Point,
#         bbox: BBox,
#         padding: int = 20
#     ) -> bool:
#         """Check if a position is inside a bounding box with padding.
        
#         Args:
#             bee_position: (x, y) coordinates
#             bbox: Bounding box (x_min, y_min, x_max, y_max)
#             padding: Padding to add around box
            
#         Returns:
#             True if position is inside padded box
#         """
#         x, y = bee_position
#         x_min, y_min, x_max, y_max = bbox
        
#         # Add padding with slightly more vertical padding
#         x_min -= padding
#         y_min -= int(padding + padding / 2)
#         x_max += padding
#         y_max += int(padding + padding / 2)
        
#         return x_min <= x <= x_max and y_min <= y <= y_max
    
#     def detect_entry(
#         self,
#         bee_trajectory: List[Point],
#         hole_bboxes: Dict[str, BBox],
#         window_size: int = 3,
#         padding: int = 20
#     ) -> int:
#         """Detect if bee enters a hole (analyze start of trajectory).
        
#         Args:
#             bee_trajectory: List of (x, y) positions
#             hole_bboxes: Dictionary mapping hole IDs to bounding boxes
#             window_size: Number of frames to analyze
#             padding: Padding around boxes
            
#         Returns:
#             Hole ID if entry detected, -1 otherwise
#         """
#         if len(bee_trajectory) < window_size:
#             window_size = max(1, len(bee_trajectory) // 2)
        
#         start_trajectory = bee_trajectory[:window_size]
        
#         for hole_id, bbox in hole_bboxes.items():
#             start_inside = all(
#                 self._is_inside_bbox(pos, bbox, padding)
#                 for pos in start_trajectory
#             )
            
#             if start_inside:
#                 return hole_id
        
#         return -1
    
#     def detect_exit(
#         self,
#         bee_trajectory: List[Point],
#         hole_bboxes: Dict[str, BBox],
#         window_size: int = 3,
#         padding: int = 20
#     ) -> int:
#         """Detect if bee exits a hole (analyze end of trajectory).
        
#         Args:
#             bee_trajectory: List of (x, y) positions
#             hole_bboxes: Dictionary mapping hole IDs to bounding boxes
#             window_size: Number of frames to analyze
#             padding: Padding around boxes
            
#         Returns:
#             Hole ID if exit detected, -1 otherwise
#         """
#         if len(bee_trajectory) < window_size:
#             window_size = max(1, len(bee_trajectory) // 2)
        
#         end_trajectory = bee_trajectory[-window_size:]
        
#         for hole_id, bbox in hole_bboxes.items():
#             end_inside = all(
#                 self._is_inside_bbox(pos, bbox, padding)
#                 for pos in end_trajectory
#             )
            
#             if end_inside:
#                 return hole_id
        
#         return -1
    
#     def process_yolo_tracks(
#         self,
#         movements: List[Tuple],
#         nests: Dict
#     ) -> pd.DataFrame:
#         """Process YOLO tracking results to identify events.
        
#         This is an alternative processing method for trajectories from
#         Ultralytics YOLO tracking rather than custom BeeTracker.
        
#         Args:
#             movements: List of trajectories from UltralyticsTracker
#             nests: Dictionary with nest locations
            
#         Returns:
#             DataFrame with events
            
#         Example:
#             >>> from bee_monitor.tracking import UltralyticsTracker
#             >>> tracker = UltralyticsTracker(model)
#             >>> trajectories = tracker.get_tracks("video.mp4")
#             >>> events = processor.process_yolo_tracks(trajectories, nests)
#         """
#         logger.info("Processing YOLO tracks to identify events...")
        
#         actions = []
#         for movement in movements:
#             # Skip short trajectories
#             if len(movement[1]) < self.config.processing.min_trajectory_length:
#                 continue
            
#             # Classify movement type
#             if self.trajectory_analyzer.is_exit_behavior(movement):
#                 action = self._get_action(
#                     movement,
#                     nests,
#                     window_size=self.config.processing.exit_window_size,
#                     padding=self.config.processing.exit_padding
#                 )
#             elif self.trajectory_analyzer.is_entry_behavior(movement):
#                 action = self._get_action(
#                     movement,
#                     nests,
#                     window_size=self.config.processing.entry_window_size,
#                     padding=self.config.processing.entry_padding
#                 )
#             else:
#                 continue
            
#             # Add actions to list
#             if action:
#                 if isinstance(action, list):
#                     actions.extend(action)
#                 else:
#                     actions.append(action)
        
#         logger.info(f"Identified {len(actions)} events from YOLO tracks")
        
#         if actions:
#             return pd.DataFrame(actions)
#         else:
#             return pd.DataFrame(columns=['action', 'nest', 'frame_number', 'notes'])
    
#     def __repr__(self) -> str:
#         """String representation of processor."""
#         return f"EventProcessor(config={self.config is not None})"













# """Process tracking data into entry/exit events with species information.

# This module analyzes trajectories to determine entry and exit events at nest holes,
# including species classification for each event.
# """

# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple, Optional, Union


# def is_inside_bbox(bee_position: Tuple[float, float], bbox: Tuple, padding: float = 20) -> bool:
#     """Check if a point is inside a bounding box with padding.
    
#     Args:
#         bee_position: Position as (x, y)
#         bbox: Bounding box as (x_min, y_min, x_max, y_max)
#         padding: Padding around bbox
        
#     Returns:
#         True if point is inside padded bbox
#     """
#     x, y = bee_position
#     x_min, y_min, x_max, y_max = bbox
#     x_min -= padding
#     y_min -= int(padding + padding / 2)
#     x_max += padding
#     y_max += int(padding + padding / 2)
#     return x_min <= x <= x_max and y_min <= y <= y_max


# def detect_entry_exit(
#     bee_trajectory: List[Tuple],
#     hole_bboxes: Dict,
#     window_size: int = 3,
#     padding: float = 20
# ) -> Tuple[str, str]:
#     """Detect if trajectory represents entry or exit from nest.
    
#     Args:
#         bee_trajectory: List of (x, y) positions
#         hole_bboxes: Dictionary of {hole_id: bbox}
#         window_size: Number of frames to analyze
#         padding: Padding around nest holes
        
#     Returns:
#         Tuple of (start_hole_id, end_hole_id) or (-1, -1) if none
#     """
#     if len(bee_trajectory) < window_size:
#         window_size = int(len(bee_trajectory) / 2)
    
#     start_trajectory = bee_trajectory[:window_size]
#     end_trajectory = bee_trajectory[-window_size:]
    
#     # Check start position
#     start_id = -1
#     for hole_id, bbox in hole_bboxes.items():
#         start_inside = all(is_inside_bbox(pos, bbox, padding) for pos in start_trajectory)
#         if start_inside:
#             start_id = hole_id
#             break
    
#     # Check end position
#     end_id = -1
#     for hole_id, bbox in hole_bboxes.items():
#         end_inside = all(is_inside_bbox(pos, bbox, padding) for pos in end_trajectory)
#         if end_inside:
#             end_id = hole_id
#             break
    
#     return start_id, end_id


# def calculate_speed(trajectory: List[Tuple]) -> List[float]:
#     """Calculate speed from trajectory.
    
#     Args:
#         trajectory: List of (x, y) positions
        
#     Returns:
#         List of speeds
#     """
#     speeds = []
#     for i in range(1, len(trajectory)):
#         x1, y1 = trajectory[i - 1]
#         x2, y2 = trajectory[i]
#         distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         speeds.append(distance)
#     return speeds


# def check_start_and_end_speed(movement: Tuple) -> Tuple[float, float]:
#     """Check start and end speed of movement.
    
#     Args:
#         movement: Trajectory tuple
        
#     Returns:
#         Tuple of (start_speed, end_speed)
#     """
#     speeds = calculate_speed(movement[1])
#     if len(speeds) > 0:
#         return speeds[0], speeds[-1]
#     return 0.0, 0.0


# def is_entry(movement: Tuple, end_speed_threshold: float = 10.0) -> bool:
#     """Check if movement represents entry.
    
#     Args:
#         movement: Trajectory tuple
#         end_speed_threshold: Maximum ending speed for entry
        
#     Returns:
#         True if movement is entry
#     """
#     _, end_speed = check_start_and_end_speed(movement)
#     return end_speed < end_speed_threshold


# def is_exit(movement: Tuple, start_speed_threshold: float = 10.0) -> bool:
#     """Check if movement represents exit.
    
#     Args:
#         movement: Trajectory tuple
#         start_speed_threshold: Maximum starting speed for exit
        
#     Returns:
#         True if movement is exit
#     """
#     start_speed, _ = check_start_and_end_speed(movement)
#     return start_speed < start_speed_threshold


# def get_action_with_species(
#     movement: Tuple,
#     nest: Dict,
#     window_size: int = 3,
#     padding: float = 20,
#     species_map: Optional[Dict[int, str]] = None
# ) -> Optional[Union[Dict, List[Dict]]]:
#     """Determine action from trajectory with species information.
    
#     Args:
#         movement: Trajectory tuple (track_id, centroids, bboxes, frame_numbers, species, species_votes)
#         nest: Dictionary with nest hole locations
#         window_size: Window size for analysis
#         padding: Padding around nests
#         species_map: Mapping of class IDs to species names
        
#     Returns:
#         Dictionary or list of dictionaries with event information including species
#     """
#     start_id, end_id = detect_entry_exit(movement[1], nest['nests'], window_size, padding)
    
#     # Get species information from track
#     species_class = movement[4] if len(movement) > 4 else None
#     species_votes = movement[5] if len(movement) > 5 else {}
    
#     # Determine species name
#     if species_map and species_class is not None:
#         species_name = species_map.get(species_class, 'unknown')
#     else:
#         species_name = 'unknown'
    
#     # Calculate species confidence (proportion of votes for primary species)
#     species_confidence = 0.0
#     if species_votes:
#         total_votes = sum(species_votes.values())
#         if total_votes > 0 and species_class is not None:
#             species_confidence = species_votes.get(species_class, 0) / total_votes
    
#     if start_id == -1 and end_id == -1:
#         return None
    
#     elif start_id != -1 and end_id == -1:
#         # Exit event
#         return {
#             "action": "Exit",
#             "nest": f"{start_id}",
#             "frame_number": movement[3][0],
#             "species": species_name,
#             "species_class": species_class,
#             "species_confidence": species_confidence,
#             "notes": f"{species_name} exited the nest"
#         }
    
#     elif start_id == -1 and end_id != -1:
#         # Entry event
#         return {
#             "action": "Entry",
#             "nest": f"{end_id}",
#             "frame_number": movement[3][-1],
#             "species": species_name,
#             "species_class": species_class,
#             "species_confidence": species_confidence,
#             "notes": f"{species_name} entered the nest"
#         }
    
#     elif start_id != -1 and end_id != -1:
#         # Both exit and entry (nest transfer)
#         return [
#             {
#                 "action": "Exit",
#                 "nest": f"{start_id}",
#                 "frame_number": movement[3][0],
#                 "species": species_name,
#                 "species_class": species_class,
#                 "species_confidence": species_confidence,
#                 "notes": f"{species_name} exited nest to move to another hole {end_id}"
#             },
#             {
#                 "action": "Entry",
#                 "nest": f"{end_id}",
#                 "frame_number": movement[3][-1],
#                 "species": species_name,
#                 "species_class": species_class,
#                 "species_confidence": species_confidence,
#                 "notes": f"{species_name} entered nest from another hole {start_id}"
#             }
#         ]
    
#     return None


# def process_tracking_with_species(
#     motion: pd.DataFrame,
#     nest: Dict,
#     species_map: Optional[Dict[int, str]] = None,
#     min_trajectory_length: int = 5,
#     entry_window: int = 6,
#     exit_window: int = 3,
#     entry_padding: float = 10,
#     exit_padding: float = 20,
#     start_speed_threshold: float = 10.0,
#     end_speed_threshold: float = 10.0
# ) -> pd.DataFrame:
#     """Process tracking data into events with species information.
    
#     Args:
#         motion: DataFrame with tracking data
#         nest: Dictionary with nest information
#         species_map: Mapping of class IDs to species names
#         min_trajectory_length: Minimum trajectory length to process
#         entry_window: Window size for entry detection
#         exit_window: Window size for exit detection
#         entry_padding: Padding for entry detection
#         exit_padding: Padding for exit detection
#         start_speed_threshold: Speed threshold for exit
#         end_speed_threshold: Speed threshold for entry
        
#     Returns:
#         DataFrame with columns: timestamp, nest, action, species, species_class, species_confidence
#     """
#     # Collect all movements
#     movements = []
#     for period in motion.tracks:
#         for track in period:
#             movements.append(track)
    
#     # Process each movement
#     actions = []
#     for movement in movements:
#         # Skip short trajectories
#         if len(movement[1]) < min_trajectory_length:
#             continue
        
#         # Determine if exit or entry
#         action = None
        
#         if is_exit(movement, start_speed_threshold):
#             action = get_action_with_species(
#                 movement,
#                 nest,
#                 exit_window,
#                 exit_padding,
#                 species_map
#             )
#         elif is_entry(movement, end_speed_threshold):
#             action = get_action_with_species(
#                 movement,
#                 nest,
#                 entry_window,
#                 entry_padding,
#                 species_map
#             )
        
#         # Add action(s) to list
#         if action:
#             if isinstance(action, list):
#                 actions.extend(action)
#             else:
#                 actions.append(action)
    
#     # Create DataFrame
#     if actions:
#         return pd.DataFrame(actions)
#     else:
#         # Return empty DataFrame with correct columns
#         return pd.DataFrame(columns=[
#             'action', 'nest', 'frame_number', 'species',
#             'species_class', 'species_confidence', 'notes'
#         ])


# def process_tracking(
#     motion: pd.DataFrame,
#     nest: Dict,
#     species_map: Optional[Dict[int, str]] = None
# ) -> pd.DataFrame:
#     """Process tracking data into events (backward compatible).
    
#     This is the main entry point that maintains backward compatibility
#     while adding species support.
    
#     Args:
#         motion: DataFrame with tracking data
#         nest: Dictionary with nest information
#         species_map: Optional mapping of class IDs to species names
        
#     Returns:
#         DataFrame with event information including species
#     """
#     return process_tracking_with_species(
#         motion,
#         nest,
#         species_map=species_map
#     )


# def process_yolo_tracks(
#     movements: List,
#     nest: Dict,
#     species_map: Optional[Dict[int, str]] = None
# ) -> pd.DataFrame:
#     """Process YOLO tracking results with species information.
    
#     Args:
#         movements: List of track trajectories
#         nest: Dictionary with nest information
#         species_map: Mapping of class IDs to species names
        
#     Returns:
#         DataFrame with event information
#     """
#     actions = []
    
#     for movement in movements:
#         if len(movement[1]) < 5:
#             continue
        
#         action = None
        
#         if is_exit(movement):
#             action = get_action_with_species(
#                 movement,
#                 nest,
#                 window_size=3,
#                 padding=20,
#                 species_map=species_map
#             )
#         elif is_entry(movement):
#             action = get_action_with_species(
#                 movement,
#                 nest,
#                 window_size=6,
#                 padding=10,
#                 species_map=species_map
#             )
        
#         if action:
#             if isinstance(action, list):
#                 actions.extend(action)
#             else:
#                 actions.append(action)
    
#     if actions:
#         return pd.DataFrame(actions)
#     else:
#         return pd.DataFrame(columns=[
#             'action', 'nest', 'frame_number', 'species',
#             'species_class', 'species_confidence', 'notes'
#         ])











"""Event processing for bee tracking data with multi-species support.

This module processes bee trajectories to identify entry and exit events
at nest holes, including species classification.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

from beemonitor.core.config import Config
from beemonitor.processing.trajectory_analyzer import TrajectoryAnalyzer


logger = logging.getLogger(__name__)

# Type aliases
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class EventProcessor:
    """Processor for identifying bee entry/exit events with species tracking.
    
    This class analyzes bee trajectories to determine when bees enter
    or exit nest holes, creating a timeline of activity events with
    species information.
    
    Attributes:
        config: Configuration object
        trajectory_analyzer: TrajectoryAnalyzer instance
    
    Example:
        >>> processor = EventProcessor(config)
        >>> events = processor.process_tracks(motion_data, nests)
        >>> print(f"Found {len(events)} events")
        >>> print(events['species'].value_counts())
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize EventProcessor.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config if config is not None else Config.default()
        self.trajectory_analyzer = TrajectoryAnalyzer(self.config)
        
        logger.debug("EventProcessor initialized with species support")
    
    def process_tracks(
        self,
        motion_data: pd.DataFrame,
        nests: Dict,
        species_map: Optional[Dict[int, str]] = None
    ) -> pd.DataFrame:
        """Process tracking data to identify entry/exit events with species.
        
        Args:
            motion_data: DataFrame with columns: frame_number, tracks, detections
            nests: Dictionary with 'hotel' ROI and 'nests' mapping
            species_map: Optional mapping of class IDs to species names
            
        Returns:
            DataFrame with columns: action, nest, frame_number, species, 
                                   species_class, species_confidence, notes
            
        Example:
            >>> events = processor.process_tracks(motion_data, nests)
            >>> entries = events[events['action'] == 'Entry']
            >>> print(f"Found {len(entries)} entry events")
            >>> # With species
            >>> honeybee_entries = entries[entries['species'] == 'honeybee']
        """
        logger.info("Processing tracks to identify events...")
        
        # Use species_map from config if not provided
        if species_map is None and hasattr(self.config, 'tracking'):
            species_map = self.config.tracking.species_map
        
        # Extract all movements from tracking data
        movements = []
        for period in motion_data.tracks:
            for track in period:
                movements.append(track)
        
        logger.debug(f"Processing {len(movements)} trajectories")
        
        # Get resolution for scaled parameters
        res_width = self.config.video.res_width
        res_height = self.config.video.res_height
        
        # Process each movement to identify events
        actions = []
        for movement in movements:
            # Skip short trajectories
            if len(movement[1]) < self.config.processing.min_trajectory_length:
                continue
            
            # Classify movement type
            if self.trajectory_analyzer.is_exit_behavior(movement):
                # Get scaled parameters for exit
                exit_window = self.config.processing.exit_window_size
                exit_padding = self.config.processing.exit_padding(res_width, res_height)
                
                action = self._get_action(
                    movement,
                    nests,
                    window_size=exit_window,
                    padding=exit_padding,
                    species_map=species_map
                )
            elif self.trajectory_analyzer.is_entry_behavior(movement):
                # Get scaled parameters for entry
                entry_window = self.config.processing.entry_window_size
                entry_padding = self.config.processing.entry_padding(res_width, res_height)
                
                action = self._get_action(
                    movement,
                    nests,
                    window_size=entry_window,
                    padding=entry_padding,
                    species_map=species_map
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
            df = pd.DataFrame(actions)
            # Log species distribution if present
            if 'species' in df.columns:
                species_counts = df['species'].value_counts()
                logger.info(f"Species distribution: {species_counts.to_dict()}")
            return df
        else:
            return pd.DataFrame(columns=[
                'action', 'nest', 'frame_number', 'species',
                'species_class', 'species_confidence', 'notes'
            ])
    
    def _get_action(
        self,
        movement: Tuple,
        nests: Dict,
        window_size: int = 3,
        padding: float = 20,
        species_map: Optional[Dict[int, str]] = None
    ) -> Optional[Union[Dict, List[Dict]]]:
        """Determine action (entry/exit) from movement trajectory with species.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers, 
                               species, species_votes)
            nests: Dictionary with nest locations
            window_size: Number of frames to analyze
            padding: Padding around nest boxes (already scaled)
            species_map: Mapping of class IDs to species names
            
        Returns:
            Dictionary or list of dictionaries with action details, or None
        """
        start_id, end_id = self._detect_entry_exit(
            movement[1],  # centroids
            nests['nests'],
            window_size=window_size,
            padding=padding
        )
        
        # Get species information from track
        species_class = movement[4] if len(movement) > 4 else None
        species_votes = movement[5] if len(movement) > 5 else {}
        
        # Determine species name
        if species_map and species_class is not None:
            species_name = species_map.get(species_class, 'unknown')
        else:
            species_name = 'unknown'
        
        # Calculate species confidence
        species_confidence = 0.0
        if species_votes:
            total_votes = sum(species_votes.values())
            if total_votes > 0 and species_class is not None:
                species_confidence = species_votes.get(species_class, 0) / total_votes
        
        if start_id == -1 and end_id == -1:
            return None
        
        elif start_id != -1 and end_id == -1:
            # Exit only
            return {
                "action": "Exit",
                "nest": str(start_id),
                "frame_number": movement[3][0],  # First frame
                "species": species_name,
                "species_class": species_class,
                "species_confidence": species_confidence,
                "notes": f"{species_name} exited the nest"
            }
        
        elif start_id == -1 and end_id != -1:
            # Entry only
            return {
                "action": "Entry",
                "nest": str(end_id),
                "frame_number": movement[3][-1],  # Last frame
                "species": species_name,
                "species_class": species_class,
                "species_confidence": species_confidence,
                "notes": f"{species_name} entered the nest"
            }
        
        elif start_id != -1 and end_id != -1:
            # Both entry and exit (nest-to-nest movement)
            return [
                {
                    "action": "Exit",
                    "nest": str(start_id),
                    "frame_number": movement[3][0],
                    "species": species_name,
                    "species_class": species_class,
                    "species_confidence": species_confidence,
                    "notes": f"{species_name} exited nest to move to another hole {end_id}"
                },
                {
                    "action": "Entry",
                    "nest": str(end_id),
                    "frame_number": movement[3][-1],
                    "species": species_name,
                    "species_class": species_class,
                    "species_confidence": species_confidence,
                    "notes": f"{species_name} entered nest from another hole {start_id}"
                }
            ]
        
        return None
    
    def _detect_entry_exit(
        self,
        bee_trajectory: List[Point],
        hole_bboxes: Dict[str, BBox],
        window_size: int = 3,
        padding: float = 20
    ) -> Tuple[int, int]:
        """Detect if bee enters or exits a hole.
        
        Analyzes the start and end of a trajectory to determine if the bee
        started inside a hole (exit) or ended inside a hole (entry).
        
        Args:
            bee_trajectory: List of (x, y) positions
            hole_bboxes: Dictionary mapping hole IDs to bounding boxes
            window_size: Number of frames to analyze at start/end
            padding: Padding to add around nest boxes (already scaled)
            
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
        padding: float = 20
    ) -> bool:
        """Check if a position is inside a bounding box with padding.
        
        Args:
            bee_position: (x, y) coordinates
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            padding: Padding to add around box (already scaled)
            
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
        padding: float = 20
    ) -> int:
        """Detect if bee enters a hole (analyze start of trajectory).
        
        Args:
            bee_trajectory: List of (x, y) positions
            hole_bboxes: Dictionary mapping hole IDs to bounding boxes
            window_size: Number of frames to analyze
            padding: Padding around boxes (already scaled)
            
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
        padding: float = 20
    ) -> int:
        """Detect if bee exits a hole (analyze end of trajectory).
        
        Args:
            bee_trajectory: List of (x, y) positions
            hole_bboxes: Dictionary mapping hole IDs to bounding boxes
            window_size: Number of frames to analyze
            padding: Padding around boxes (already scaled)
            
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
        nests: Dict,
        species_map: Optional[Dict[int, str]] = None
    ) -> pd.DataFrame:
        """Process YOLO tracking results to identify events with species.
        
        This is an alternative processing method for trajectories from
        Ultralytics YOLO tracking rather than custom BeeTracker.
        
        Args:
            movements: List of trajectories from UltralyticsTracker
            nests: Dictionary with nest locations
            species_map: Optional mapping of class IDs to species names
            
        Returns:
            DataFrame with events
            
        Example:
            >>> from beemonitor.tracking import UltralyticsTracker
            >>> tracker = UltralyticsTracker(model)
            >>> trajectories = tracker.get_tracks("video.mp4")
            >>> events = processor.process_yolo_tracks(trajectories, nests)
        """
        logger.info("Processing YOLO tracks to identify events...")
        
        # Use species_map from config if not provided
        if species_map is None and hasattr(self.config, 'tracking'):
            species_map = self.config.tracking.species_map
        
        # Get resolution for scaled parameters
        res_width = self.config.video.res_width
        res_height = self.config.video.res_height
        
        actions = []
        for movement in movements:
            # Skip short trajectories
            if len(movement[1]) < self.config.processing.min_trajectory_length:
                continue
            
            # Classify movement type
            if self.trajectory_analyzer.is_exit_behavior(movement):
                exit_window = self.config.processing.exit_window_size
                exit_padding = self.config.processing.exit_padding(res_width, res_height)
                
                action = self._get_action(
                    movement,
                    nests,
                    window_size=exit_window,
                    padding=exit_padding,
                    species_map=species_map
                )
            elif self.trajectory_analyzer.is_entry_behavior(movement):
                entry_window = self.config.processing.entry_window_size
                entry_padding = self.config.processing.entry_padding(res_width, res_height)
                
                action = self._get_action(
                    movement,
                    nests,
                    window_size=entry_window,
                    padding=entry_padding,
                    species_map=species_map
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
            df = pd.DataFrame(actions)
            # Log species distribution if present
            if 'species' in df.columns:
                species_counts = df['species'].value_counts()
                logger.info(f"Species distribution: {species_counts.to_dict()}")
            return df
        else:
            return pd.DataFrame(columns=[
                'action', 'nest', 'frame_number', 'species',
                'species_class', 'species_confidence', 'notes'
            ])
    
    def __repr__(self) -> str:
        """String representation of processor."""
        return f"EventProcessor(config={self.config is not None}, species_tracking={hasattr(self.config, 'tracking')})"


# Backward compatibility functions
def is_inside_bbox(bee_position: Tuple[float, float], bbox: Tuple, padding: float = 20) -> bool:
    """Check if a point is inside a bounding box with padding.
    
    Args:
        bee_position: Position as (x, y)
        bbox: Bounding box as (x_min, y_min, x_max, y_max)
        padding: Padding around bbox
        
    Returns:
        True if point is inside padded bbox
    """
    processor = EventProcessor()
    return processor._is_inside_bbox(bee_position, bbox, padding)


def process_tracking(
    motion: pd.DataFrame,
    nest: Dict,
    species_map: Optional[Dict[int, str]] = None,
    config: Optional[Config] = None
) -> pd.DataFrame:
    """Process tracking data into events (backward compatible function).
    
    Args:
        motion: DataFrame with tracking data
        nest: Dictionary with nest information
        species_map: Optional mapping of class IDs to species names
        config: Optional configuration object
        
    Returns:
        DataFrame with event information including species
    """
    processor = EventProcessor(config)
    return processor.process_tracks(motion, nest, species_map)


def process_yolo_tracks(
    movements: List,
    nest: Dict,
    species_map: Optional[Dict[int, str]] = None,
    config: Optional[Config] = None
) -> pd.DataFrame:
    """Process YOLO tracking results (backward compatible function).
    
    Args:
        movements: List of track trajectories
        nest: Dictionary with nest information
        species_map: Mapping of class IDs to species names
        config: Optional configuration object
        
    Returns:
        DataFrame with event information
    """
    processor = EventProcessor(config)
    return processor.process_yolo_tracks(movements, nest, species_map)