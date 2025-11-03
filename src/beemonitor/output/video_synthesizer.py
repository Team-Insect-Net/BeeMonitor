"""Video synthesis for bee monitoring results.

This module handles generating annotated videos showing bee tracks and events.
"""

import logging
from typing import Dict, Optional
from pathlib import Path
import cv2
import pandas as pd
import numpy as np

from beemonitor.core.config import Config


logger = logging.getLogger(__name__)


class VideoSynthesizer:
    """Synthesizer for annotated video output.
    
    This class creates videos with visual annotations showing:
    - Nest locations and IDs
    - Bee tracking boxes and IDs
    - Entry/exit events
    
    Attributes:
        config: Configuration object
    
    Example:
        >>> synthesizer = VideoSynthesizer(config)
        >>> output_path = synthesizer.synthesize(
        ...     "video.mp4", events, motion_data, nests, "output/"
        ... )
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize VideoSynthesizer.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config if config is not None else Config.default()
    
    def synthesize(
        self,
        video_path: str,
        events: pd.DataFrame,
        motion_data: pd.DataFrame,
        nests: Dict,
        output_folder: str,
        res_height: int = 720,
        res_width: int = 1280
    ) -> str:
        """Generate annotated video with tracking visualization.
        
        Args:
            video_path: Path to input video
            events: DataFrame with events (action, nest, frame_number)
            motion_data: DataFrame with tracking data (frame_number, tracks)
            nests: Dictionary with nest locations
            output_folder: Directory for output
            res_height: Output video height
            res_width: Output video width
            
        Returns:
            Path to generated video file
            
        Example:
            >>> output = synthesizer.synthesize(
            ...     "video.mp4", events, motion_data, nests, "output/"
            ... )
            >>> print(f"Saved to {output}")
        """
        logger.info(f"Synthesizing video from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Create output path
        filename = Path(video_path).stem
        output_path = Path(output_folder) / f"synthesized_video_{filename}.mp4"
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.output.video_codec)
        fps = self.config.output.video_fps
        output_video = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (res_width, res_height)
        )
        
        # Extract event information
        frame_numbers = events.frame_number.tolist() if not events.empty else []
        nest_holes = events.nest.tolist() if not events.empty else []
        actions = events.action.tolist() if not events.empty else []
        
        # Process each tracking period
        for i in range(len(motion_data.frame_number.tolist())):
            try:
                period = motion_data.frame_number.tolist()[i]
                tracks = motion_data.tracks.tolist()[i]
                
                # Create track objects
                track_objects = [
                    self._TrackHelper(track[0], track[2], track[3])
                    for track in tracks
                ]
                
                # Process frames in this period
                for frame_num in range(period[0], period[1] + 1):
                    # Read frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    # Resize frame
                    frame = cv2.resize(frame, (res_width, res_height))
                    
                    # Draw nest boxes and IDs
                    frame = self._draw_nests(frame, nests)
                    
                    # Draw tracks
                    frame = self._draw_tracks(frame, track_objects, frame_num)
                    
                    # Draw events
                    if frame_num in frame_numbers:
                        idx = frame_numbers.index(frame_num)
                        frame = self._draw_event(
                            frame,
                            actions[idx],
                            nest_holes[idx]
                        )
                        
                        # Hold frame for 1 second to highlight event
                        for _ in range(fps):
                            output_video.write(frame)
                    
                    # Write frame
                    output_video.write(frame)
            
            except Exception as e:
                logger.warning(f"Error processing period {i}: {e}")
                continue
        
        # Clean up
        cap.release()
        output_video.release()
        
        logger.info(f"Saved synthesized video to {output_path}")
        
        return str(output_path)
    
    def _draw_nests(self, frame: np.ndarray, nests: Dict) -> np.ndarray:
        """Draw nest boxes and IDs on frame.
        
        Args:
            frame: Input frame
            nests: Dictionary with nest locations
            
        Returns:
            Frame with nest annotations
        """
        for nest_id, bbox in nests['nests'].items():
            # Extract nest number from ID
            nest_num = nest_id.split('_')[-1] if '_' in nest_id else nest_id
            
            x1, y1, x2, y2 = bbox
            
            # Add padding
            x1 -= 5
            y1 -= 7
            x2 += 5
            y2 += 7
            
            # Draw rectangle (blue)
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 0),
                2
            )
            
            # Draw ID
            cv2.putText(
                frame,
                str(nest_num),
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 0, 0),
                2
            )
        
        return frame
    
    def _draw_tracks(
        self,
        frame: np.ndarray,
        track_objects: list,
        frame_num: int
    ) -> np.ndarray:
        """Draw track boxes and IDs on frame.
        
        Args:
            frame: Input frame
            track_objects: List of track helper objects
            frame_num: Current frame number
            
        Returns:
            Frame with track annotations
        """
        for track in track_objects:
            if track.is_in_frame(frame_num):
                bbox = track.get_bbox(frame_num)
                x1, y1, x2, y2 = bbox
                
                # Draw rectangle (green)
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                
                # Draw track ID
                cv2.putText(
                    frame,
                    str(track.get_id()),
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )
        
        return frame
    
    def _draw_event(
        self,
        frame: np.ndarray,
        action: str,
        nest_id: str
    ) -> np.ndarray:
        """Draw event text on frame.
        
        Args:
            frame: Input frame
            action: Event action (Entry/Exit)
            nest_id: Nest ID
            
        Returns:
            Frame with event annotation
        """
        text = f"{action} at nest {nest_id}"
        
        # Choose color based on action
        color = (255, 0, 0) if action == "Exit" else (0, 255, 0)
        
        # Draw text with background for visibility
        cv2.putText(
            frame,
            text,
            (50, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )
        
        return frame
    
    class _TrackHelper:
        """Helper class for managing track information during synthesis."""
        
        def __init__(self, track_id: int, trajectory: list, frame_numbers: list):
            """Initialize track helper.
            
            Args:
                track_id: Track ID
                trajectory: List of bounding boxes
                frame_numbers: List of frame numbers
            """
            self.track_id = track_id
            self.trajectory = trajectory
            self.frame_numbers = frame_numbers
        
        def is_in_frame(self, frame_num: int) -> bool:
            """Check if track is present in frame.
            
            Args:
                frame_num: Frame number to check
                
            Returns:
                True if track is in frame
            """
            return frame_num in self.frame_numbers
        
        def get_bbox(self, frame_num: int) -> tuple:
            """Get bounding box for frame.
            
            Args:
                frame_num: Frame number
                
            Returns:
                Bounding box (x1, y1, x2, y2)
            """
            idx = self.frame_numbers.index(frame_num)
            return self.trajectory[idx]
        
        def get_id(self) -> int:
            """Get track ID.
            
            Returns:
                Track ID
            """
            return self.track_id
    
    def __repr__(self) -> str:
        """String representation of synthesizer."""
        return f"VideoSynthesizer(config={self.config is not None})"