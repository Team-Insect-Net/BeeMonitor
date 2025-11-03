"""Wrapper for Ultralytics YOLO tracking.

This module provides a clean interface to Ultralytics' built-in tracking
algorithms (ByteTrack, BoTSORT, etc.).
"""

import logging
from collections import defaultdict
from typing import List, Tuple, Optional
import cv2

from beemonitor.core.config import Config


logger = logging.getLogger(__name__)

# Type aliases
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class UltralyticsTracker:
    """Wrapper for Ultralytics YOLO tracking.
    
    Provides a simplified interface to use Ultralytics' built-in tracking
    algorithms like ByteTrack and BoTSORT.
    
    Attributes:
        model: YOLO model instance
        tracker_config: Path to tracker configuration file
        config: BeeMonitor configuration
    
    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> tracker = UltralyticsTracker(model, "config/bytetrack.yaml")
        >>> trajectories = tracker.get_tracks("video.mp4")
    """
    
    def __init__(
        self,
        model,
        tracker_config: str = "config/bytetrack.yaml",
        config: Optional[Config] = None
    ):
        """Initialize UltralyticsTracker.
        
        Args:
            model: YOLO model instance
            tracker_config: Path to tracker configuration file
            config: BeeMonitor configuration (optional)
        """
        self.model = model
        self.tracker_config = tracker_config
        self.config = config if config is not None else Config.default()
        
        logger.info(f"Initialized UltralyticsTracker with {tracker_config}")
    
    def get_tracks(
        self,
        video_path: str
    ) -> List[Tuple[int, List[Point], List, List[int]]]:
        """Get trajectories from a video using YOLO tracking.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            List of trajectories, where each trajectory is a tuple containing:
                - track_id (int): The ID of the track
                - track (List[Point]): List of (x, y) coordinates
                - [] (empty list): Placeholder for bbox history
                - track_frame (List[int]): Frame numbers for each position
        
        Example:
            >>> tracker = UltralyticsTracker(model)
            >>> trajectories = tracker.get_tracks("video.mp4")
            >>> for track_id, coords, _, frames in trajectories:
            ...     print(f"Track {track_id}: {len(coords)} positions")
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Store track history
        track_history = defaultdict(lambda: [])
        track_frame_history = defaultdict(lambda: [])
        
        frame_num = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing {total_frames} frames with Ultralytics tracker")
        
        while cap.isOpened():
            # Read frame
            success, frame = cap.read()
            
            if not success:
                break
            
            # Run YOLO tracking
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                verbose=False
            )
            
            # Extract results
            if len(results) > 0:
                result = results[0]
                
                # Check if we have boxes and track IDs
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    
                    # Update track history
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track_frame = track_frame_history[track_id]
                        
                        # Store center point
                        track.append((float(x), float(y)))
                        track_frame.append(frame_num)
            
            frame_num += 1
            
            # Log progress
            if frame_num % 100 == 0:
                logger.debug(f"Processed {frame_num}/{total_frames} frames")
        
        cap.release()
        
        # Convert to trajectory format
        trajectories = []
        for track_id in track_history:
            trajectories.append((
                track_id,
                track_history[track_id],
                [],  # Placeholder for bbox history
                track_frame_history[track_id]
            ))
        
        logger.info(f"Extracted {len(trajectories)} trajectories")
        
        return trajectories
    
    def track_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_labels: bool = True,
        show_conf: bool = False,
        line_width: int = 2
    ) -> str:
        """Track objects in video and save annotated output.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            show_labels: Whether to show labels
            show_conf: Whether to show confidence scores
            line_width: Line width for bounding boxes
            
        Returns:
            Path to output video
            
        Example:
            >>> tracker = UltralyticsTracker(model)
            >>> output = tracker.track_video("input.mp4", "output.mp4")
        """
        if output_path is None:
            output_path = video_path.replace('.mp4', '_tracked.mp4')
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Creating tracked video: {output_path}")
        
        frame_num = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                break
            
            # Run tracking
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                verbose=False
            )
            
            # Annotate frame
            if len(results) > 0:
                annotated_frame = results[0].plot(
                    labels=show_labels,
                    conf=show_conf,
                    line_width=line_width
                )
            else:
                annotated_frame = frame
            
            # Write frame
            out.write(annotated_frame)
            
            frame_num += 1
            
            if frame_num % 100 == 0:
                logger.debug(f"Processed {frame_num}/{total_frames} frames")
        
        # Clean up
        cap.release()
        out.release()
        
        logger.info(f"Saved tracked video to {output_path}")
        
        return output_path
    
    def __repr__(self) -> str:
        """String representation of tracker."""
        return f"UltralyticsTracker(config={self.tracker_config})"