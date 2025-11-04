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
import os

from beemonitor.core.config import Config

from pathlib import Path
from typing import Dict, Optional
import cv2

import re


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
        self.motion_data = motion_data
    
    def to_csv(self, output_folder: str = "output", columns: Optional[List[str]] = None) -> None:
        """Export events to CSV file.
        
        Args:
            filename: Output CSV file path
            columns: Columns to include (default: all)
            
        Example:
            >>> results.to_csv("output/events.csv")
            >>> results.to_csv("output/events.csv", columns=["timestamp", "nest", "action"])
        """
        filename = self.video_path.replace(".mp4", "_events.csv") 
        filename = filename.split("/")[-1]
        filename = str(Path(output_folder) / Path(filename).name)
        if columns is None:
            self.events.to_csv(filename, index=False)
        else:
            available_cols = [col for col in columns if col in self.events.columns]
            self.events[available_cols].to_csv(filename, index=False)
        
        logger.info(f"Saved {len(self.events)} events to {filename}")
    
    def save_video(self, output_folder: str = "output") -> None:
        """Save annotated video with tracking visualization.
        
        Args:
            filename: Output video file path
            output_folder: Directory for output (default: "output")
            
        Example:
            >>> results.save_video("annotated.mp4")
        """
        from beemonitor.output.video_synthesizer import VideoSynthesizer
        from beemonitor.core.config import Config
        
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
        config: Optional[Config] = None
    ):
        """Initialize BeeMonitor with model paths and configuration.
        
        Args:
            config: Configuration object (default: None, uses default config)

            use config defualt to initialize settings if config is None
        
        Raises:
            FileNotFoundError: If model files don't exist
            ValueError: If resolution values are invalid
        """
        if config is None:
            config = Config.default()
        self.config = config
        self.res_height = config.video.res_height
        self.res_width = config.video.res_width
        self.nest_model = YOLO(config.models.nest_detection)
        self.tracking_model = YOLO(config.models.tracking)

        
        logger.info("BeeMonitor initialized successfully")
    
    @classmethod
    def from_config(cls, config_path: str) -> "BeeMonitor":
        """Create BeeMonitor from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Initialized BeeMonitor instance
            
        Example:
            >>> monitor = BeeMonitor()
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
        nest_video_path: Optional[str] = None,
        output_folder: Optional[str] = None,
        visualize: Optional[bool] = None 
    ) -> AnalysisResults:
        """Analyze a video to detect and track bee activity.
        
        This is the main method that orchestrates the entire analysis pipeline:
        1. Detect nests in the video
        2. Detect motion and track bees
        3. Process tracks to identify entry/exit events
        
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
            >>> monitor = BeeMonitor()
            >>> results = monitor.analyze_video("video.mp4")
            >>> results.to_csv("output.csv")
            >>> print(f"Found {len(results.events)} events")
        """
        # Validate video path
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
         # NEW: Use config settings if not explicitly provided
        if visualize is None:
            visualize = self.config.output.save_tracking_visualizations
        
        # Set output folder
        if output_folder is None:
            output_folder = self.config.output.base_folder
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting analysis of {video_path}")
        
        # Step 1: Detect nests
        if nest_video_path is None:
            nest_video_path = video_path
        logger.info("Step 1/3: Detecting nests...")
        nests = self.get_nest_detections(nest_video_path)

        if nests is None:
            logger.warning("No nests detected, skipping video analysis")
            return None
        
        # Step 3: Detect motion and track bees
        logger.info("Step 2/3: Detecting motion and tracking bees...")
        motion_data = self.get_motion_tracking(
            video_path,
            nests['hotel'],
            output_folder,
            visualize=visualize
        )
        
        # Step 4: Process tracks to get events
        logger.info("Step 3/3: Processing tracks to identify events...")
        events = self.process_motion_tracking(motion_data, nests)
        
        # Create results object
        results = AnalysisResults(
            events=events,
            tracks=motion_data.get('tracks', []) if isinstance(motion_data, dict) else motion_data['tracks'].tolist(),
            nests=nests,
            video_path=video_path,
            motion_data=motion_data
        )
        
        logger.info(f"Analysis complete: {len(events)} events detected")
        
        return results
    
    def analyze_video_with_relative_nests(
        self,
        video_path: str,
        video_files: List[str],
        output_folder: Optional[str] = None,
        visualize: Optional[bool] = None
    )-> AnalysisResults:
        """Analyze a video using nest detections from adjacent videos.
        
        Args:
            video_path: Path to input video file
            video_files: List of all video files in the directory
            output_folder: Directory for output files (default: from config)
            visualize: Whether to save visualization videos (default: False)
        Returns:
            AnalysisResults object containing all analysis outputs
        """ 
        # Get previous and next videos
        prev_video = self._get_prev_video(video_path, video_files)
        next_video = self._get_next_video(video_path, video_files)
        
        # Try previous video first
        if prev_video is not None:
            logger.info(f"Using nest detections from previous video: {prev_video}")
            nests = self.get_nest_detections(prev_video)
            if nests is not None and not nests.empty:
                return self.analyze_video(
                    video_path,
                    prev_video,
                    output_folder=output_folder,
                    visualize=visualize
                )
        
        # Fallback to next video
        if next_video is not None:
            logger.info(f"Using nest detections from next video: {next_video}")
            nests = self.get_nest_detections(next_video)
            if nests is not None and not nests.empty:
                return self.analyze_video(
                    video_path,
                    next_video,
                    output_folder=output_folder,
                    visualize=visualize
                )
        
        logger.warning("No adjacent videos with valid nest detections found")
        return None
    
    def analyze_videos_in_folder(
        self,
        folder_path: str,
        output_folder: Optional[str] = None,
        visualize: Optional[bool] = None
    ) -> Dict[str, AnalysisResults]:
        """Analyze all videos in a specified folder.

        Notes videos in the folder should all be from the same site/hotel.
        
        Args:
            folder_path: Path to folder containing video files
            output_folder: Directory for output files (default: from config)
            visualize: Whether to save visualization videos (default: False)    

        Returns:
            Dictionary mapping video filenames to AnalysisResults   
        """
        video_files = [
            str(f) for f in Path(folder_path).glob("*.mp4")
        ]
        
        results = {}
        for video_path in video_files:
            logger.info(f"Analyzing video: {video_path}")
            analysis_result = self.analyze_video(
                video_path,
                output_folder=output_folder,
                visualize=visualize
            )
            if analysis_result is None:
                logger.warning(f"Analysis failed for video: {video_path} trying relative nests")
                results[video_path] = self.analyze_video_with_relative_nests(
                    video_path,
                    video_files,
                    output_folder=output_folder,
                    visualize=visualize
                )
            else:
                results[video_path] = analysis_result
        
        return results
    

    

    def get_nest_detections(self, video_path: str) -> pd.DataFrame:
        """Detect nests using improved robust detector."""
        from beemonitor.detection.nest_detector import (
            NestDetector
        )
        logger.info("Starting nest detection with improved detector")

        
        # Initialize detector
        detector = NestDetector(
            model=self.nest_model,
            config = self.config
        )
            
        # Detect and assign IDs
        nests = detector.get_nest_detections(
            video_path=video_path,
            # res_height=self.res_height,
            # res_width=self.res_width
        )
    
        return nests
    

    
    # def process_nest_detection(
    #     self,
    #     video_path: str,
    #     nest_detection: pd.DataFrame
    # ) -> Dict:
    #     """Process nest detections to identify individual nest holes.
        
    #     Args:
    #         video_path: Path to video file
    #         nest_detection: DataFrame from get_nest_detection
            
    #     Returns:
    #         Dictionary with 'hotel' ROI and 'nests' mapping
    #     """
    #     # Import here to avoid circular imports
    #     from beemonitor.detection.nest_detector import NestDetector
        
    #     detector = NestDetector(
    #         model=self.nest_model,
    #         config=self.config
    #     )
        
    #     return detector.process_detections(
    #         video_path=video_path,
    #         nest_detection=nest_detection,
    #         res_height=self.res_height,
    #         res_width=self.res_width
    #     )


    # def process_nest_detection(
    #     self,
    #     video_path: str,
    #     nest_detection: pd.DataFrame
    # ) -> Dict:
    #     """Process nest detections into format needed by motion tracking."""
        
    #     logger.info("Processing nest detections...")
        
    #     # Extract nest dictionary from DataFrame
    #     if 'nest_dict' in nest_detection.columns:
    #         nest_dict = nest_detection.iloc[0]['nest_dict']
    #     else:
    #         # Legacy format
    #         coordinates = nest_detection.iloc[0]['coordinates']
    #         nest_dict = {str(i+1): box for i, box in enumerate(coordinates)}
        
    #     # Calculate hotel ROI
    #     all_boxes = list(nest_dict.values())
        
    #     if not all_boxes:
    #         raise ValueError("No nests detected")
        
    #     x_min = max(0, int(min(box[0] for box in all_boxes) - 100))
    #     y_min = max(0, int(min(box[1] for box in all_boxes) - 50))
    #     x_max = min(self.res_width, int(max(box[2] for box in all_boxes) + 100))
    #     y_max = min(self.res_height, int(max(box[3] for box in all_boxes) + 50))
        
    #     hotel_roi = (x_min, y_min, x_max, y_max)
        
    #     logger.info(f"Hotel ROI: {hotel_roi}")
    #     logger.info(f"Nests processed: {len(nest_dict)}")
        
    #     # Save visualization
    #     #self._save_nest_visualization(video_path, hotel_roi, nest_dict)
        
    #     return {
    #         'hotel': hotel_roi,
    #         'nests': nest_dict
    #     }
    
    # def _save_nest_visualization(
    #     self,
    #     video_path: str,
    #     hotel_roi: tuple,
    #     nest_dict: Dict
    # ):
    #     """Save visualization of detected nests with IDs."""
    #     cap = cv2.VideoCapture(video_path)
    #     ret, frame = cap.read()
    #     cap.release()
        
    #     if not ret or frame is None:
    #         logger.warning("Could not read frame for visualization")
    #         return
        
    #     # Resize and draw
    #     frame = cv2.resize(frame, (self.res_width, self.res_height))
        
    #     # Draw hotel ROI
    #     x1, y1, x2, y2 = hotel_roi
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        
    #     # Draw nests with IDs
    #     for nest_id, box in nest_dict.items():
    #         x1, y1, x2, y2 = [int(v) for v in box]
    #         x1 -= 5
    #         y1 -= 7
    #         x2 += 5
    #         y2 += 7
            
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(
    #             frame, nest_id, (x1, y1 - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
    #         )
        
    #     # Save
    #     output_path = Path(video_path).parent / f"{Path(video_path).stem}_nest_detection.png"
    #     cv2.imwrite(str(output_path), frame)
    #     logger.info(f"Saved visualization: {output_path}")


    def _get_next_video(
            self,
            video_path: str,
            video_files: List[str],
    ) -> str:
        """Get the next consecutive video file after the current video

        1. Get the timestamp from the filename (i.e.,"2024-05-11_10_50_00")
        2. Extract the date from the timestamp
        3. Filter files in video_files to those with the same date as the timestamp from the current video
        4. Sort the filtered files
        6. Find the index of the current video file
        7. Return the next file if it exists, else return None
        """

        def extract_ts(fp: str) -> Optional[str]:
            m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}", Path(fp).stem)
            return m.group(0) if m else None

        current_path = Path(video_path)
        current_ts = extract_ts(current_path.name)

        # If no timestamp found, fall back to lexicographic ordering of all files
        if current_ts is None:
            logger.debug("No timestamp found in current filename; falling back to filename ordering")
            sorted_files = sorted(video_files)
        else:
            date_part = current_ts.split("_")[0]
            # Filter files that contain the same date in their timestamp
            same_date_files = []
            for f in video_files:
                ts = extract_ts(f)
                if ts and ts.startswith(date_part):
                    same_date_files.append(f)
            if not same_date_files:
                logger.debug("No same-date files found for %s", video_path)
                return None
            # Sort by full timestamp (lexicographic sort works because format is sortable)
            sorted_files = sorted(same_date_files, key=lambda p: extract_ts(p) or "")

        # Normalize names for matching
        sorted_paths = [str(Path(p)) for p in sorted_files]
        try:
            idx = next(i for i, p in enumerate(sorted_paths) if Path(p).name == current_path.name)
        except StopIteration:
            logger.debug("Current video %s not found in filtered list", video_path)
            return None

        next_idx = idx + 1
        if next_idx < len(sorted_paths):
            return sorted_paths[next_idx]
        return None

    def _get_prev_video(
            self,
            video_path: str,
            video_files: List[str],
    ) -> str:
        """Get the next consecutive video file after the current video

        1. Get the timestamp from the filename (i.e.,"2024-05-11_10_50_00")
        2. Extract the date from the timestamp
        3. Filter files in video_files to those with the same date as the timestamp from the current video
        4. Sort the filtered files
        6. Find the index of the current video file
        7. Return the prev file if it exists, else return None
        """

        def extract_ts(fp: str) -> Optional[str]:
            m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}", Path(fp).stem)
            return m.group(0) if m else None

        current_path = Path(video_path)
        current_ts = extract_ts(current_path.name)

        # If no timestamp found, fall back to lexicographic ordering of all files
        if current_ts is None:
            logger.debug("No timestamp found in current filename; falling back to filename ordering")
            sorted_files = sorted(video_files)
        else:
            date_part = current_ts.split("_")[0]
            # Filter files that contain the same date in their timestamp
            same_date_files = []
            for f in video_files:
                ts = extract_ts(f)
                if ts and ts.startswith(date_part):
                    same_date_files.append(f)
            if not same_date_files:
                logger.debug("No same-date files found for %s", video_path)
                return None
            # Sort by full timestamp (lexicographic sort works because format is sortable)
            sorted_files = sorted(same_date_files, key=lambda p: extract_ts(p) or "")

        # Normalize names for matching
        sorted_paths = [str(Path(p)) for p in sorted_files]
        try:
            idx = next(i for i, p in enumerate(sorted_paths) if Path(p).name == current_path.name)
        except StopIteration:
            logger.debug("Current video %s not found in filtered list", video_path)
            return None

        prev_idx = idx - 1
        if prev_idx >= 0:
            return sorted_paths[prev_idx]
        return None

    
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
        from beemonitor.detection.motion_detector import MotionDetector
        
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
        from beemonitor.processing.event_processor import EventProcessor
        
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
        from beemonitor.output.csv_generator import CSVGenerator
        
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
        from beemonitor.output.video_synthesizer import VideoSynthesizer
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