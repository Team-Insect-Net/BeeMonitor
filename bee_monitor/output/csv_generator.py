"""CSV generation for bee monitoring results.

This module handles generating CSV files with timestamps from event data.
"""

import logging
from datetime import time, datetime, timedelta
from typing import Optional
from pathlib import Path
import pandas as pd

from bee_monitor.core.config import Config


logger = logging.getLogger(__name__)


class CSVGenerator:
    """Generator for CSV output files.
    
    This class handles converting event data to properly formatted CSV files
    with timestamps calculated from video filenames and frame numbers.
    
    Attributes:
        config: Configuration object
    
    Example:
        >>> generator = CSVGenerator(config)
        >>> csv_df = generator.generate_csv(events, "site_2024-01-15_14_30_00.mp4")
        >>> csv_df.to_csv("output.csv", index=False)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize CSVGenerator.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config if config is not None else Config.default()
    
    def generate_csv(
        self,
        events: pd.DataFrame,
        video_path: str
    ) -> pd.DataFrame:
        """Generate CSV with timestamps from events.
        
        Adds timestamp column to events based on video filename and frame numbers.
        
        Args:
            events: DataFrame with columns: action, nest, frame_number, notes
            video_path: Path to video file (used for timestamp calculation)
            
        Returns:
            DataFrame with timestamp column added
            
        Example:
            >>> events = pd.DataFrame({
            ...     'action': ['Entry', 'Exit'],
            ...     'nest': ['1', '2'],
            ...     'frame_number': [100, 200]
            ... })
            >>> csv_df = generator.generate_csv(events, "site_2024-01-15_14_30_00.mp4")
            >>> print(csv_df.columns)
            ['timestamp', 'nest', 'action']
        """
        logger.info(f"Generating CSV from {len(events)} events")
        
        if events.empty:
            logger.warning("No events to process")
            return pd.DataFrame(columns=self.config.output.csv_columns)
        
        # Get base datetime from filename
        base_datetime = self._get_start_time(video_path)
        
        # Calculate timestamps
        fps = self.config.video.fps
        events['timestamp'] = events['frame_number'].apply(
            lambda frame: self._get_timestamp(frame, base_datetime, fps)
        )
        
        # Add filename
        events['filename'] = Path(video_path).name
        
        # Select and order columns
        columns = self.config.output.csv_columns.copy()
        if 'filename' not in columns:
            columns.append('filename')
        
        available_columns = [col for col in columns if col in events.columns]
        
        logger.info(f"Generated CSV with {len(events)} rows")
        
        return events[available_columns]
    
    def _get_start_time(self, filename: str) -> datetime:
        """Extract start time from video filename.
        
        Expected filename format: site_YYYY-MM-DD_HH_MM_SS.mp4
        
        Args:
            filename: Video filename or path
            
        Returns:
            Datetime object representing video start time
            
        Raises:
            ValueError: If filename doesn't match expected format
            
        Example:
            >>> start_time = generator._get_start_time("site_2024-01-15_14_30_00.mp4")
            >>> print(start_time)
            2024-01-15 14:30:00
        """
        # Extract filename without path and extension
        filename = Path(filename).stem
        
        try:
            # Split filename
            parts = filename.split('_')
            
            if len(parts) < 5:
                raise ValueError(
                    f"Filename '{filename}' doesn't match expected format: "
                    "site_YYYY-MM-DD_HH_MM_SS"
                )
            
            # Extract components
            # Format: site_YYYY-MM-DD_HH_MM_SS
            date_str = parts[1]  # YYYY-MM-DD
            hour = int(parts[2])
            minute = int(parts[3])
            second = int(parts[4])
            
            # Parse date
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Create time object
            time_obj = time(hour, minute, second)
            
            # Combine into datetime
            base_datetime = datetime.combine(date_obj, time_obj)
            
            logger.debug(f"Parsed start time: {base_datetime}")
            
            return base_datetime
        
        except (IndexError, ValueError) as e:
            logger.error(f"Error parsing filename '{filename}': {e}")
            # Return current time as fallback
            logger.warning("Using current time as fallback")
            return datetime.now()
    
    def _get_timestamp(
        self,
        frame_number: int,
        base_datetime: datetime,
        fps: int = 30
    ) -> datetime:
        """Calculate timestamp for a frame number.
        
        Args:
            frame_number: Frame number in video
            base_datetime: Video start time
            fps: Frames per second
            
        Returns:
            Datetime for the specified frame
            
        Example:
            >>> base = datetime(2024, 1, 15, 14, 30, 0)
            >>> timestamp = generator._get_timestamp(900, base, 30)
            >>> print(timestamp)  # 30 seconds later
            2024-01-15 14:30:30
        """
        seconds_elapsed = int(frame_number / fps)
        return base_datetime + timedelta(seconds=seconds_elapsed)
    
    def process_csv(
        self,
        events: pd.DataFrame,
        filename: str
    ) -> pd.DataFrame:
        """Process events to CSV format (legacy method).
        
        This is an alias for generate_csv for backwards compatibility.
        
        Args:
            events: DataFrame with events
            filename: Video filename
            
        Returns:
            DataFrame with timestamps
        """
        return self.generate_csv(events, filename)
    
    def save_csv(
        self,
        events: pd.DataFrame,
        video_path: str,
        output_path: str
    ) -> None:
        """Generate and save CSV file.
        
        Args:
            events: DataFrame with events
            video_path: Path to video file
            output_path: Path for output CSV file
            
        Example:
            >>> generator.save_csv(events, "video.mp4", "output/results.csv")
        """
        csv_df = self.generate_csv(events, video_path)
        csv_df.to_csv(output_path, index=False)
        logger.info(f"Saved CSV to {output_path}")
    
    def __repr__(self) -> str:
        """String representation of generator."""
        return f"CSVGenerator(config={self.config is not None})"