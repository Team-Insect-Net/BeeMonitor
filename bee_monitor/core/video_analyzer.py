"""Main video analyzer class for bee monitoring.

This module provides the BeeMonitor class, which orchestrates the entire
bee detection and tracking pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import pandas as pd
import numpy as np
from ultralytics import YOLO

from bee_monitor.core.config import Config


logger = logging.getLogger(__name__)


class AnalysisResults:
    """Container for video analysis results.
    
    This class holds all the outputs from video analysis and provides
    convenient methods for exporting and accessing results.
    
    Attributes:
        events: DataFrame containing entry/exit events with timestamps
        tracks: List of bee trajectories
        nests: Dictionary mapping nest IDs to bounding boxes
        video_path: Path to the analyzed video
        nest_detections: Raw nest detection DataFrame
        motion_data: Motion detection DataFrame
    
    Example:
        >>> results = monitor.analyze_video("video.mp4")
        >>> results.to_csv("output.csv")
        >>> print(f"Found {len(results.events)} events")
        >>> stats = results.get_statistics()
    """
    
    def __init__(
        self,
        events: pd.DataFrame,
        tracks: List,
        nests: Dict,
        video_path: str,
        nest_detections: Optional[pd.DataFrame] = None,
        motion_data: Optional[pd.DataFrame] = None
    ):
        """Initialize analysis results.
        
        Args:
            events: DataFrame with processed events
            tracks: List of bee trajectories
            nests: Dictionary of nest locations
            video_path: Path to analyzed video
            nest_detections: Raw nest detection data (optional)
            motion_data: Motion detection data (optional)
        """
        self.events = events
        self.tracks = tracks
        self.nests = nests
        self.video_path = video_path
        self.nest_detections = nest_detections
        self.motion_data = motion_data
    
    def to_csv(self, filename: str, columns: Optional[List[str]] = None) -> None:
        """Export events to CSV file.
        
        Args:
            filename: Output CSV file path
            columns: Columns to include (default: all)
            
        Example:
            >>> results.to_csv("output/events.csv")
            >>> results.to_csv("output/events.csv", columns=["timestamp", "nest", "action"])
        """
        if columns is None:
            self.events.to_csv(filename, index=False)
        else:
            available_cols = [col for col in columns if col in self.events.columns]
            self.events[available_cols].to_csv(filename, index=False)
        
        logger.info(f"Saved {len(self.events)} events to {filename}")
    
    def save_video(self, filename: str, output_folder: str = "output") -> None:
        """Save annotated video with tracking visualization.
        
        Args:
            filename: Output video file path
            output_folder: Directory for output (default: "output")
            
        Example:
            >>> results.save_video("annotated.mp4")
        """
        from bee_monitor.output.video_synthesizer import VideoSynthesizer
        from bee_monitor.core.config import Config
        
        config = Config.default()
        synthesizer = VideoSynthesizer(config)
        
        output_path = synthesizer.synthesize(
            self.video_path,
            self.events,
            self.motion_data,
            self.nests,
            output_folder
        )
        
        logger.info(f"Saved annotated video to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics from the analysis.
        
        Returns:
            Dictionary containing analysis statistics
            
        Example:
            >>> stats = results.get_statistics()
            >>> print(f"Total entries: {stats['total_entries']}")
        """
        if self.events.empty:
            return {
                "total_events": 0,
                "total_entries": 0,
                "total_exits": 0,
                "active_nests": 0,
                "total_tracks": 0,
            }
        
        stats = {
            "total_events": len(self.events),
            "total_entries": len(self.events[self.events['action'] == 'Entry']),
            "total_exits": len(self.events[self.events['action'] == 'Exit']),
            "active_nests": len(self.events['nest'].unique()),
            "total_nests": len(self.nests.get('nests', {})),
            "total_tracks": len(self.tracks) if isinstance(self.tracks, list) else 0,
        }
        
        # Add per-nest statistics
        if 'nest' in self.events.columns:
            nest_counts = self.events.groupby('nest')['action'].value_counts().unstack(fill_value=0)
            stats['nest_activity'] = nest_counts.to_dict()
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of results."""
        return (
            f"AnalysisResults(events={len(self.events)}, "
            f"tracks={len(self.tracks) if isinstance(self.tracks, list) else 0}, "
            f"nests={len(self.nests.get('nests', {}))})"
        )


class BeeMonitor:
    """Main interface for bee monitoring video analysis.
    
    This class provides a high-level API for analyzing bee hotel videos,
    including nest detection, motion tracking, and event processing.
    
    Attributes:
        nest_model: YOLO model for nest detection
        tracking_model: YOLO model for bee tracking
        config: Configuration object with all settings
        res_height: Video resolution height
        res_width: Video resolution width
    
    Example:
        >>> # Method 1: From configuration file
        >>> monitor = BeeMonitor.from_config("config/default_config.yaml")
        >>> results = monitor.analyze_video("video.mp4")
        
        >>> # Method 2: With explicit models
        >>> monitor = BeeMonitor(
        ...     nest_model_path="models/nest.pt",
        ...     tracking_model_path="models/tracking.pt"
        ... )
        >>> results = monitor.analyze_video("video.mp4")
    """
    
    def __init__(
        self,
        nest_model_path: str,
        tracking_model_path: str,
        res_height: int = 720,
        res_width: int = 1280,
        config: Optional[Config] = None
    ):
        """Initialize BeeMonitor with model paths and configuration.
        
        Args:
            nest_model_path: Path to nest detection YOLO model
            tracking_model_path: Path to bee tracking YOLO model
            res_height: Video resolution height (default: 720)
            res_width: Video resolution width (default: 1280)
            config: Configuration object (default: None, uses default config)
        
        Raises:
            FileNotFoundError: If model files don't exist
            ValueError: If resolution values are invalid
        """
        # Validate inputs
        if res_height <= 0 or res_width <= 0:
            raise ValueError("Resolution must be positive")
        
        # Load models
        logger.info(f"Loading nest detection model from {nest_model_path}")
        self.nest_model = YOLO(nest_model_path)
        
        logger.info(f"Loading tracking model from {tracking_model_path}")
        self.tracking_model = YOLO(tracking_model_path)
        
        # Store configuration
        self.config = config if config is not None else Config.default()
        self.res_height = res_height
        self.res_width = res_width
        
        logger.info("BeeMonitor initialized successfully")
    
    @classmethod
    def from_config(cls, config_path: str) -> "BeeMonitor":
        """Create BeeMonitor from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Initialized BeeMonitor instance
            
        Example:
            >>> monitor = BeeMonitor.from_config("config/default_config.yaml")
            >>> results = monitor.analyze_video("video.mp4")
        """
        config = Config.from_yaml(config_path)
        
        return cls(
            nest_model_path=config.models.nest_detection,
            tracking_model_path=config.models.tracking,
            res_height=config.video.height,
            res_width=config.video.width,
            config=config
        )
    
    def analyze_video(
        self,
        video_path: str,
        output_folder: Optional[str] = None,
        visualize: bool = False
    ) -> AnalysisResults:
        """Analyze a video to detect and track bee activity.
        
        This is the main method that orchestrates the entire analysis pipeline:
        1. Detect nests in the video
        2. Process nest detections to identify individual holes
        3. Detect motion and track bees
        4. Process tracks to identify entry/exit events
        
        Args:
            video_path: Path to input video file
            output_folder: Directory for output files (default: from config)
            visualize: Whether to save visualization videos (default: False)
            
        Returns:
            AnalysisResults object containing all analysis outputs
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
            
        Example:
            >>> monitor = BeeMonitor.from_config("config/default_config.yaml")
            >>> results = monitor.analyze_video("video.mp4")
            >>> results.to_csv("output.csv")
            >>> print(f"Found {len(results.events)} events")
        """
        # Validate video path
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Set output folder
        if output_folder is None:
            output_folder = self.config.output.base_folder
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting analysis of {video_path}")
        
        # Step 1: Detect nests
        logger.info("Step 1/4: Detecting nests...")
        nest_detections = self.get_nest_detection(video_path)
        
        # Step 2: Process nest detections
        logger.info("Step 2/4: Processing nest detections...")
        nests = self.process_nest_detection(video_path, nest_detections)
        
        # Step 3: Detect motion and track bees
        logger.info("Step 3/4: Detecting motion and tracking bees...")
        motion_data = self.get_motion_tracking(
            video_path,
            nests['hotel'],
            output_folder,
            visualize=visualize
        )
        
        # Step 4: Process tracks to get events
        logger.info("Step 4/4: Processing tracks to identify events...")
        events = self.process_motion_tracking(motion_data, nests)
        
        # Create results object
        results = AnalysisResults(
            events=events,
            tracks=motion_data.get('tracks', []) if isinstance(motion_data, dict) else motion_data['tracks'].tolist(),
            nests=nests,
            video_path=video_path,
            nest_detections=nest_detections,
            motion_data=motion_data
        )
        
        logger.info(f"Analysis complete: {len(events)} events detected")
        
        return results
    
    def get_nest_detection(self, video_path: str) -> pd.DataFrame:
        """Detect nests in the video.
        
        This method will use the nest_detector module to find and identify
        nest holes in the bee hotel.
        
        Args:
            video_path: Path to video file
            
        Returns:
            DataFrame with nest detection results
            
        Note:
            This is a placeholder that will be implemented when we create
            the detection module.
        """
        # Import here to avoid circular imports
        # This will be implemented when we create the detection module
        from bee_monitor.detection.nest_detector import NestDetector
        
        detector = NestDetector(
            model=self.nest_model,
            config=self.config
        )
        
        return detector.detect_nests(
            video_path=video_path,
            res_height=self.res_height,
            res_width=self.res_width
        )
    
    def process_nest_detection(
        self,
        video_path: str,
        nest_detection: pd.DataFrame
    ) -> Dict:
        """Process nest detections to identify individual nest holes.
        
        Args:
            video_path: Path to video file
            nest_detection: DataFrame from get_nest_detection
            
        Returns:
            Dictionary with 'hotel' ROI and 'nests' mapping
        """
        # Import here to avoid circular imports
        from bee_monitor.detection.nest_detector import NestDetector
        
        detector = NestDetector(
            model=self.nest_model,
            config=self.config
        )
        
        return detector.process_detections(
            video_path=video_path,
            nest_detection=nest_detection,
            res_height=self.res_height,
            res_width=self.res_width
        )
    
    def get_motion_tracking(
        self,
        video_path: str,
        hotel_roi: Tuple[float, float, float, float],
        output_folder: str,
        visualize: bool = False
    ) -> pd.DataFrame:
        """Detect motion and track bees in the video.
        
        Args:
            video_path: Path to video file
            hotel_roi: Region of interest (x1, y1, x2, y2)
            output_folder: Directory for output files
            visualize: Whether to save visualization
            
        Returns:
            DataFrame with tracking results
        """
        # Import here to avoid circular imports
        from bee_monitor.detection.motion_detector import MotionDetector
        
        detector = MotionDetector(
            model=self.tracking_model,
            config=self.config
        )
        
        return detector.detect_and_track(
            video_path=video_path,
            site_roi=hotel_roi,
            res_height=self.res_height,
            res_width=self.res_width,
            visualize=visualize,
            output_folder=output_folder
        )
    
    def process_motion_tracking(
        self,
        motion_data: pd.DataFrame,
        nests: Dict
    ) -> pd.DataFrame:
        """Process tracking data to identify entry/exit events.
        
        Args:
            motion_data: DataFrame from get_motion_tracking
            nests: Dictionary from process_nest_detection
            
        Returns:
            DataFrame with events (timestamp, nest, action)
        """
        # Import here to avoid circular imports
        from bee_monitor.processing.event_processor import EventProcessor
        
        processor = EventProcessor(config=self.config)
        
        return processor.process_tracks(
            motion_data=motion_data,
            nests=nests
        )
    
    def synthesize_csv(
        self,
        events: pd.DataFrame,
        video_path: str
    ) -> pd.DataFrame:
        """Generate CSV with timestamps from events.
        
        Args:
            events: DataFrame with events
            video_path: Path to video file (for timestamp calculation)
            
        Returns:
            DataFrame with timestamps added
        """
        # Import here to avoid circular imports
        from bee_monitor.output.csv_generator import CSVGenerator
        
        generator = CSVGenerator(config=self.config)
        
        return generator.generate_csv(
            events=events,
            video_path=video_path
        )
    
    def synthesize_video(
        self,
        video_path: str,
        events: pd.DataFrame,
        motion_data: pd.DataFrame,
        nests: Dict,
        output_folder: str
    ) -> str:
        """Generate annotated video with tracking visualization.
        
        Args:
            video_path: Path to input video
            events: DataFrame with events
            motion_data: DataFrame with tracking data
            nests: Dictionary with nest locations
            output_folder: Directory for output
            
        Returns:
            Path to generated video file
        """
        # Import here to avoid circular imports
        from bee_monitor.output.video_synthesizer import VideoSynthesizer
        
        synthesizer = VideoSynthesizer(config=self.config)
        
        return synthesizer.synthesize(
            video_path=video_path,
            events=events,
            motion_data=motion_data,
            nests=nests,
            output_folder=output_folder,
            res_height=self.res_height,
            res_width=self.res_width
        )
    
    def __repr__(self) -> str:
        """String representation of BeeMonitor."""
        return (
            f"BeeMonitor(resolution={self.res_width}x{self.res_height}, "
            f"config={self.config is not None})"
        )