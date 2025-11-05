# """CSV generation for bee monitoring results.

# This module handles generating CSV files with timestamps from event data.
# """

# import logging
# from datetime import time, datetime, timedelta
# from typing import Optional
# from pathlib import Path
# import pandas as pd

# from beemonitor.core.config import Config


# logger = logging.getLogger(__name__)


# class CSVGenerator:
#     """Generator for CSV output files.
    
#     This class handles converting event data to properly formatted CSV files
#     with timestamps calculated from video filenames and frame numbers.
    
#     Attributes:
#         config: Configuration object
    
#     Example:
#         >>> generator = CSVGenerator(config)
#         >>> csv_df = generator.generate_csv(events, "site_2024-01-15_14_30_00.mp4")
#         >>> csv_df.to_csv("output.csv", index=False)
#     """
    
#     def __init__(self, config: Optional[Config] = None):
#         """Initialize CSVGenerator.
        
#         Args:
#             config: Configuration object (optional)
#         """
#         self.config = config if config is not None else Config.default()
    
#     def generate_csv(
#         self,
#         events: pd.DataFrame,
#         video_path: str
#     ) -> pd.DataFrame:
#         """Generate CSV with timestamps from events.
        
#         Adds timestamp column to events based on video filename and frame numbers.
        
#         Args:
#             events: DataFrame with columns: action, nest, frame_number, notes
#             video_path: Path to video file (used for timestamp calculation)
            
#         Returns:
#             DataFrame with timestamp column added
            
#         Example:
#             >>> events = pd.DataFrame({
#             ...     'action': ['Entry', 'Exit'],
#             ...     'nest': ['1', '2'],
#             ...     'frame_number': [100, 200]
#             ... })
#             >>> csv_df = generator.generate_csv(events, "site_2024-01-15_14_30_00.mp4")
#             >>> print(csv_df.columns)
#             ['timestamp', 'nest', 'action']
#         """
#         logger.info(f"Generating CSV from {len(events)} events")
        
#         if events.empty:
#             logger.warning("No events to process")
#             return pd.DataFrame(columns=self.config.output.csv_columns)
        
#         # Get base datetime from filename
#         base_datetime = self._get_start_time(video_path)
        
#         # Calculate timestamps
#         fps = self.config.video.fps
#         events['timestamp'] = events['frame_number'].apply(
#             lambda frame: self._get_timestamp(frame, base_datetime, fps)
#         )
        
#         # Add filename
#         events['filename'] = Path(video_path).name
        
#         # Select and order columns
#         columns = self.config.output.csv_columns.copy()
#         if 'filename' not in columns:
#             columns.append('filename')
        
#         available_columns = [col for col in columns if col in events.columns]
        
#         logger.info(f"Generated CSV with {len(events)} rows")
        
#         return events[available_columns]
    
#     def _get_start_time(self, filename: str) -> datetime:
#         """Extract start time from video filename.
        
#         Expected filename format: site_YYYY-MM-DD_HH_MM_SS.mp4
        
#         Args:
#             filename: Video filename or path
            
#         Returns:
#             Datetime object representing video start time
            
#         Raises:
#             ValueError: If filename doesn't match expected format
            
#         Example:
#             >>> start_time = generator._get_start_time("site_2024-01-15_14_30_00.mp4")
#             >>> print(start_time)
#             2024-01-15 14:30:00
#         """
#         # Extract filename without path and extension
#         filename = Path(filename).stem
        
#         try:
#             # Split filename
#             parts = filename.split('_')
            
#             if len(parts) < 5:
#                 raise ValueError(
#                     f"Filename '{filename}' doesn't match expected format: "
#                     "site_YYYY-MM-DD_HH_MM_SS"
#                 )
            
#             # Extract components
#             # Format: site_YYYY-MM-DD_HH_MM_SS
#             date_str = parts[1]  # YYYY-MM-DD
#             hour = int(parts[2])
#             minute = int(parts[3])
#             second = int(parts[4])
            
#             # Parse date
#             date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
#             # Create time object
#             time_obj = time(hour, minute, second)
            
#             # Combine into datetime
#             base_datetime = datetime.combine(date_obj, time_obj)
            
#             logger.debug(f"Parsed start time: {base_datetime}")
            
#             return base_datetime
        
#         except (IndexError, ValueError) as e:
#             logger.error(f"Error parsing filename '{filename}': {e}")
#             # Return current time as fallback
#             logger.warning("Using current time as fallback")
#             return datetime.now()
    
#     def _get_timestamp(
#         self,
#         frame_number: int,
#         base_datetime: datetime,
#         fps: int = 30
#     ) -> datetime:
#         """Calculate timestamp for a frame number.
        
#         Args:
#             frame_number: Frame number in video
#             base_datetime: Video start time
#             fps: Frames per second
            
#         Returns:
#             Datetime for the specified frame
            
#         Example:
#             >>> base = datetime(2024, 1, 15, 14, 30, 0)
#             >>> timestamp = generator._get_timestamp(900, base, 30)
#             >>> print(timestamp)  # 30 seconds later
#             2024-01-15 14:30:30
#         """
#         seconds_elapsed = int(frame_number / fps)
#         return base_datetime + timedelta(seconds=seconds_elapsed)
    
#     def process_csv(
#         self,
#         events: pd.DataFrame,
#         filename: str
#     ) -> pd.DataFrame:
#         """Process events to CSV format (legacy method).
        
#         This is an alias for generate_csv for backwards compatibility.
        
#         Args:
#             events: DataFrame with events
#             filename: Video filename
            
#         Returns:
#             DataFrame with timestamps
#         """
#         return self.generate_csv(events, filename)
    
#     def save_csv(
#         self,
#         events: pd.DataFrame,
#         video_path: str,
#         output_path: str
#     ) -> None:
#         """Generate and save CSV file.
        
#         Args:
#             events: DataFrame with events
#             video_path: Path to video file
#             output_path: Path for output CSV file
            
#         Example:
#             >>> generator.save_csv(events, "video.mp4", "output/results.csv")
#         """
#         csv_df = self.generate_csv(events, video_path)
#         csv_df.to_csv(output_path, index=False)
#         logger.info(f"Saved CSV to {output_path}")
    
#     def __repr__(self) -> str:
#         """String representation of generator."""
#         return f"CSVGenerator(config={self.config is not None})"

# """Synthesize CSV files with species information.

# This module processes event data and creates CSV files with timestamps
# and species information.
# """

# from datetime import time, datetime, timedelta
# from typing import Optional
# import pandas as pd


# def get_start_time(filename: str) -> datetime:
#     """Extract start time from video filename.
    
#     Expected format: site_YYYY-MM-DD_HH_MM_SS.mp4
    
#     Args:
#         filename: Video filename
        
#     Returns:
#         Datetime object representing video start time
        
#     Example:
#         >>> get_start_time("site1_2024-03-15_14_30_45.mp4")
#         datetime(2024, 3, 15, 14, 30, 45)
#     """
#     try:
#         # Extract components from filename
#         parts = filename.split('.')[0].split('_')
        
#         # Assuming format: site_YYYY-MM-DD_HH_MM_SS
#         if len(parts) >= 5:
#             site = parts[0]
#             date_str = parts[1]
#             hour = int(parts[2])
#             minute = int(parts[3])
#             second = int(parts[4])
            
#             # Parse date
#             date = datetime.strptime(date_str, "%Y-%m-%d")
            
#             # Create time object
#             original_time = time(hour, minute, second)
            
#             # Combine date and time
#             base_datetime = datetime.combine(date, original_time)
            
#             return base_datetime
#         else:
#             # Fallback to current time if format doesn't match
#             return datetime.now()
    
#     except Exception as e:
#         print(f"Error parsing filename '{filename}': {e}")
#         print("Using current time as fallback")
#         return datetime.now()


# def get_timestamp(frame_number: int, base_datetime: datetime, fps: int = 30) -> datetime:
#     """Calculate timestamp for a frame number.
    
#     Args:
#         frame_number: Frame number in video
#         base_datetime: Video start datetime
#         fps: Frames per second
        
#     Returns:
#         Datetime representing the frame's timestamp
#     """
#     return base_datetime + timedelta(seconds=int(frame_number / fps))


# def process_csv_with_species(
#     events: pd.DataFrame,
#     filename: str,
#     fps: int = 30,
#     include_confidence: bool = True,
#     include_class_id: bool = False
# ) -> pd.DataFrame:
#     """Process events into CSV format with species information.
    
#     Args:
#         events: DataFrame with event data including species
#         filename: Video filename for timestamp extraction
#         fps: Video frames per second
#         include_confidence: Whether to include species confidence
#         include_class_id: Whether to include species class ID
        
#     Returns:
#         DataFrame with formatted CSV data
        
#     Example:
#         >>> events = pd.DataFrame({
#         ...     'frame_number': [100, 150],
#         ...     'nest': ['1', '2'],
#         ...     'action': ['Entry', 'Exit'],
#         ...     'species': ['honeybee', 'bumblebee'],
#         ...     'species_confidence': [0.95, 0.88]
#         ... })
#         >>> csv = process_csv_with_species(events, "video_2024-03-15_14_30_00.mp4")
#     """
#     if events.empty:
#         # Return empty DataFrame with expected columns
#         columns = ['timestamp', 'nest', 'action', 'species', 'filename']
#         if include_confidence:
#             columns.insert(4, 'species_confidence')
#         if include_class_id:
#             columns.insert(4, 'species_class')
#         return pd.DataFrame(columns=columns)
    
#     # Get base datetime from filename
#     base_datetime = get_start_time(filename)
    
#     # Calculate timestamps
#     events = events.copy()
#     events['timestamp'] = events['frame_number'].apply(
#         lambda x: get_timestamp(x, base_datetime, fps)
#     )
    
#     # Add filename
#     events['filename'] = filename
    
#     # Select and order columns
#     columns = ['timestamp', 'nest', 'action', 'species']
    
#     # Add optional columns
#     if include_confidence and 'species_confidence' in events.columns:
#         columns.append('species_confidence')
    
#     if include_class_id and 'species_class' in events.columns:
#         columns.append('species_class')
    
#     # Always include filename at end
#     columns.append('filename')
    
#     # Add notes if available
#     if 'notes' in events.columns:
#         columns.append('notes')
    
#     # Filter to existing columns
#     available_columns = [col for col in columns if col in events.columns]
    
#     return events[available_columns]


# def process_csv(
#     events: pd.DataFrame,
#     filename: str,
#     fps: int = 30
# ) -> pd.DataFrame:
#     """Process events into CSV format (backward compatible).
    
#     This is the main entry point that maintains backward compatibility
#     while adding species support.
    
#     Args:
#         events: DataFrame with event data
#         filename: Video filename
#         fps: Video frames per second
        
#     Returns:
#         DataFrame with formatted CSV data including species if available
#     """
#     # Check if species information is present
#     has_species = 'species' in events.columns
    
#     if has_species:
#         return process_csv_with_species(
#             events,
#             filename,
#             fps=fps,
#             include_confidence=True,
#             include_class_id=False
#         )
#     else:
#         # Backward compatibility: original format without species
#         if events.empty:
#             return pd.DataFrame(columns=['timestamp', 'nest', 'action', 'filename'])
        
#         base_datetime = get_start_time(filename)
#         events = events.copy()
#         events['timestamp'] = events['frame_number'].apply(
#             lambda x: get_timestamp(x, base_datetime, fps)
#         )
#         events['filename'] = filename
        
#         return events[['timestamp', 'nest', 'action', 'filename']]


# def save_csv(
#     events_df: pd.DataFrame,
#     output_path: str,
#     include_index: bool = False
# ) -> None:
#     """Save events DataFrame to CSV file.
    
#     Args:
#         events_df: DataFrame with event data
#         output_path: Path for output CSV file
#         include_index: Whether to include row index
#     """
#     events_df.to_csv(output_path, index=include_index)
#     print(f"Saved events to {output_path}")


# def generate_species_summary(
#     events_df: pd.DataFrame
# ) -> pd.DataFrame:
#     """Generate summary statistics by species.
    
#     Args:
#         events_df: DataFrame with event data including species
        
#     Returns:
#         DataFrame with summary by species
        
#     Example:
#         >>> summary = generate_species_summary(events_df)
#         >>> print(summary)
#            species  total_events  entries  exits  avg_confidence
#         0  honeybee            25       13     12            0.92
#         1  bumblebee            18        9      9            0.87
#     """
#     if 'species' not in events_df.columns:
#         return pd.DataFrame(columns=[
#             'species', 'total_events', 'entries', 'exits'
#         ])
    
#     summary = []
    
#     for species in events_df['species'].unique():
#         species_df = events_df[events_df['species'] == species]
        
#         entry_data = {
#             'species': species,
#             'total_events': len(species_df),
#             'entries': len(species_df[species_df['action'] == 'Entry']),
#             'exits': len(species_df[species_df['action'] == 'Exit'])
#         }
        
#         # Add confidence if available
#         if 'species_confidence' in species_df.columns:
#             entry_data['avg_confidence'] = species_df['species_confidence'].mean()
        
#         summary.append(entry_data)
    
#     return pd.DataFrame(summary)


# def generate_nest_summary(
#     events_df: pd.DataFrame
# ) -> pd.DataFrame:
#     """Generate summary statistics by nest.
    
#     Args:
#         events_df: DataFrame with event data
        
#     Returns:
#         DataFrame with summary by nest and species
        
#     Example:
#         >>> summary = generate_nest_summary(events_df)
#         >>> print(summary)
#            nest     species  entries  exits  total_visits
#         0     1    honeybee        5      5            10
#         1     1   bumblebee        2      2             4
#         2     2    honeybee        8      7            15
#     """
#     if 'species' not in events_df.columns:
#         # Backward compatibility without species
#         summary = []
#         for nest in events_df['nest'].unique():
#             nest_df = events_df[events_df['nest'] == nest]
#             summary.append({
#                 'nest': nest,
#                 'entries': len(nest_df[nest_df['action'] == 'Entry']),
#                 'exits': len(nest_df[nest_df['action'] == 'Exit']),
#                 'total_visits': len(nest_df)
#             })
#         return pd.DataFrame(summary)
    
#     # With species information
#     summary = []
    
#     for nest in events_df['nest'].unique():
#         nest_df = events_df[events_df['nest'] == nest]
        
#         for species in nest_df['species'].unique():
#             species_nest_df = nest_df[nest_df['species'] == species]
            
#             summary.append({
#                 'nest': nest,
#                 'species': species,
#                 'entries': len(species_nest_df[species_nest_df['action'] == 'Entry']),
#                 'exits': len(species_nest_df[species_nest_df['action'] == 'Exit']),
#                 'total_visits': len(species_nest_df)
#             })
    
#     return pd.DataFrame(summary)


# def export_full_report(
#     events_df: pd.DataFrame,
#     output_folder: str,
#     filename_base: str
# ) -> Dict[str, str]:
#     """Export complete report with multiple files.
    
#     Creates:
#     - events.csv: All events with timestamps and species
#     - species_summary.csv: Summary by species
#     - nest_summary.csv: Summary by nest and species
    
#     Args:
#         events_df: DataFrame with event data
#         output_folder: Output directory
#         filename_base: Base name for output files
        
#     Returns:
#         Dictionary with paths to created files
#     """
#     import os
    
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     output_files = {}
    
#     # Save main events file
#     events_path = os.path.join(output_folder, f"{filename_base}_events.csv")
#     save_csv(events_df, events_path)
#     output_files['events'] = events_path
    
#     # Save species summary if species info available
#     if 'species' in events_df.columns:
#         species_summary = generate_species_summary(events_df)
#         species_path = os.path.join(output_folder, f"{filename_base}_species_summary.csv")
#         save_csv(species_summary, species_path)
#         output_files['species_summary'] = species_path
    
#     # Save nest summary
#     nest_summary = generate_nest_summary(events_df)
#     nest_path = os.path.join(output_folder, f"{filename_base}_nest_summary.csv")
#     save_csv(nest_summary, nest_path)
#     output_files['nest_summary'] = nest_path
    
#     return output_files




# """CSV synthesis module with resolution-adaptive configuration.

# This module handles the conversion of tracking events to CSV format with timestamps.
# While CSV generation is mostly resolution-independent, this module is updated to
# work seamlessly with the Config system for consistency.
# """

# from datetime import time, datetime, timedelta
# from typing import Optional
# import pandas as pd


# class CSVGenerator:
#     """Handles synthesis of tracking events into CSV format.
    
#     This class converts tracking events with frame numbers into timestamped
#     CSV records suitable for analysis and reporting.
    
#     Attributes:
#         fps: Frames per second of the source video
#         columns: List of column names to include in output CSV
    
#     Example:
#         >>> from config import Config
#         >>> config = Config.default()
#         >>> synthesizer = CSVSynthesizer(config)
#         >>> csv_df = synthesizer.process(events, "site_2024-01-15_14_30_00.mp4")
#     """
    
#     def __init__(self, config):
#         """Initialize CSV synthesizer with configuration.
        
#         Args:
#             config: Config object containing video and output settings
#         """
#         self.fps = config.video.fps
#         self.columns = config.output.csv_columns
    
#     def extract_start_time(self, filename: str) -> datetime:
#         """Extract the start time from a video filename.
        
#         Expected filename format: site_YYYY-MM-DD_HH_MM_SS.mp4
        
#         Args:
#             filename: Video filename (with or without path)
            
#         Returns:
#             datetime object representing the video start time
            
#         Raises:
#             ValueError: If filename format is invalid
            
#         Example:
#             >>> synthesizer = CSVSynthesizer(config)
#             >>> dt = synthesizer.extract_start_time("site_2024-01-15_14_30_00.mp4")
#             >>> print(dt)
#             2024-01-15 14:30:00
#         """
#         try:
#             # Remove path and extension
#             basename = filename.split('/')[-1].split('.')[0]
            
#             # Parse components
#             parts = basename.split('_')
#             if len(parts) < 5:
#                 raise ValueError(f"Filename must have format: site_YYYY-MM-DD_HH_MM_SS, got: {basename}")
            
#             site = parts[0]
#             date_str = parts[1]
#             hour = int(parts[2])
#             minute = int(parts[3])
#             second = int(parts[4])
            
#             # Parse date
#             date = datetime.strptime(date_str, "%Y-%m-%d")
            
#             # Create time object
#             time_obj = time(hour, minute, second)
            
#             # Combine into datetime
#             base_datetime = datetime.combine(date, time_obj)
            
#             return base_datetime
            
#         except (ValueError, IndexError) as e:
#             raise ValueError(
#                 f"Invalid filename format: {filename}. "
#                 f"Expected format: site_YYYY-MM-DD_HH_MM_SS.mp4"
#             ) from e
    
#     def frame_to_timestamp(self, frame_number: int, base_datetime: datetime) -> datetime:
#         """Convert a frame number to a timestamp.
        
#         Args:
#             frame_number: Frame number in the video
#             base_datetime: Video start time
            
#         Returns:
#             datetime object for the frame
            
#         Example:
#             >>> base = datetime(2024, 1, 15, 14, 30, 0)
#             >>> timestamp = synthesizer.frame_to_timestamp(90, base)  # Frame 90 at 30fps = 3 seconds
#             >>> print(timestamp)
#             2024-01-15 14:30:03
#         """
#         seconds = frame_number / self.fps
#         return base_datetime + timedelta(seconds=seconds)
    
#     def process(self, events: pd.DataFrame, filename: str, 
#                 output_filename: Optional[str] = None) -> pd.DataFrame:
#         """Process events DataFrame to add timestamps and format for output.
        
#         Args:
#             events: DataFrame with columns ['frame_number', 'nest', 'action', ...]
#             filename: Source video filename for timestamp calculation
#             output_filename: Optional custom output filename
            
#         Returns:
#             DataFrame with timestamps and configured columns
            
#         Example:
#             >>> events = pd.DataFrame({
#             ...     'frame_number': [100, 200, 300],
#             ...     'nest': ['1', '2', '1'],
#             ...     'action': ['Exit', 'Entry', 'Entry']
#             ... })
#             >>> csv_df = synthesizer.process(events, "site_2024-01-15_14_30_00.mp4")
#         """
#         if events.empty:
#             # Return empty DataFrame with correct columns
#             empty_df = pd.DataFrame(columns=self.columns + ['filename'])
#             return empty_df
        
#         # Extract base datetime from filename
#         base_datetime = self.extract_start_time(filename)
        
#         # Add timestamps
#         events = events.copy()
#         events['timestamp'] = events['frame_number'].apply(
#             lambda x: self.frame_to_timestamp(x, base_datetime)
#         )
        
#         # Add filename
#         events['filename'] = output_filename if output_filename else filename
        
#         # Select and order columns
#         output_columns = self.columns + ['filename']
#         available_columns = [col for col in output_columns if col in events.columns]
        
#         return events[available_columns]
    
#     def save(self, events: pd.DataFrame, filename: str, output_path: str) -> str:
#         """Process events and save to CSV file.
        
#         Args:
#             events: DataFrame with tracking events
#             filename: Source video filename
#             output_path: Path where CSV should be saved
            
#         Returns:
#             Path to saved CSV file
            
#         Example:
#             >>> csv_path = synthesizer.save(events, "site_2024-01-15_14_30_00.mp4", 
#             ...                              "/output/results.csv")
#         """
#         processed_df = self.process(events, filename)
#         processed_df.to_csv(output_path, index=False)
#         return output_path
    

#         # Convenience function
#     def generate_csv(events: pd.DataFrame, filename: str, config, 
#                     output_path: Optional[str] = None) -> pd.DataFrame:
#         """Convenience function to synthesize CSV with config.
        
#         Args:
#             events: DataFrame with tracking events
#             filename: Source video filename
#             config: Config object
#             output_path: Optional path to save CSV
            
#         Returns:
#             Processed DataFrame
            
#         Example:
#             >>> from config import Config
#             >>> config = Config.default()
#             >>> csv_df = synthesize_csv(events, "video.mp4", config, "output.csv")
#         """
#         synthesizer = CSVGenerator(config)
#         processed_df = synthesizer.process(events, filename)
        
#         if output_path:
#             processed_df.to_csv(output_path, index=False)
        
#         return processed_df
    

#     # Legacy function for backward compatibility
#     def processCSV(events: pd.DataFrame, filename: str, fps: int = 30) -> pd.DataFrame:
#         """Legacy function for backward compatibility.
        
#         Args:
#             events: DataFrame with events
#             filename: Video filename
#             fps: Frames per second
            
#         Returns:
#             Processed DataFrame
            
#         Note:
#             This function is deprecated. Use CSVSynthesizer class instead.
#         """
#         # Create a minimal config for legacy support
#         from dataclasses import dataclass, field
        
#         @dataclass
#         class LegacyVideoConfig:
#             fps: int = fps
        
#         @dataclass  
#         class LegacyOutputConfig:
#             csv_columns: list = field(default_factory=lambda: ["timestamp", "nest", "action"])
        
#         @dataclass
#         class LegacyConfig:
#             video: LegacyVideoConfig = field(default_factory=LegacyVideoConfig)
#             output: LegacyOutputConfig = field(default_factory=LegacyOutputConfig)
        
#         config = LegacyConfig()
#         synthesizer = CSVGenerator(config)
#         return synthesizer.process(events, filename)


# # Legacy function for backward compatibility
# def processCSV(events: pd.DataFrame, filename: str, fps: int = 30) -> pd.DataFrame:
#     """Legacy function for backward compatibility.
    
#     Args:
#         events: DataFrame with events
#         filename: Video filename
#         fps: Frames per second
        
#     Returns:
#         Processed DataFrame
        
#     Note:
#         This function is deprecated. Use CSVSynthesizer class instead.
#     """
#     # Create a minimal config for legacy support
#     from dataclasses import dataclass, field
    
#     @dataclass
#     class LegacyVideoConfig:
#         fps: int = fps
    
#     @dataclass  
#     class LegacyOutputConfig:
#         csv_columns: list = field(default_factory=lambda: ["timestamp", "nest", "action"])
    
#     @dataclass
#     class LegacyConfig:
#         video: LegacyVideoConfig = field(default_factory=LegacyVideoConfig)
#         output: LegacyOutputConfig = field(default_factory=LegacyOutputConfig)
    
#     config = LegacyConfig()
#     synthesizer = CSVGenerator(config)
#     return synthesizer.process(events, filename)


# # Convenience function
# def generate_csv(events: pd.DataFrame, filename: str, config, 
#                    output_path: Optional[str] = None) -> pd.DataFrame:
#     """Convenience function to synthesize CSV with config.
    
#     Args:
#         events: DataFrame with tracking events
#         filename: Source video filename
#         config: Config object
#         output_path: Optional path to save CSV
        
#     Returns:
#         Processed DataFrame
        
#     Example:
#         >>> from config import Config
#         >>> config = Config.default()
#         >>> csv_df = synthesize_csv(events, "video.mp4", config, "output.csv")
#     """
#     synthesizer = CSVGenerator(config)
#     processed_df = synthesizer.process(events, filename)
    
#     if output_path:
#         processed_df.to_csv(output_path, index=False)
    
#     return processed_df


















"""CSV synthesis module with resolution-adaptive configuration.

This module handles the conversion of tracking events to CSV format with timestamps.
While CSV generation is mostly resolution-independent, this module is updated to
work seamlessly with the Config system for consistency.
"""

from datetime import time, datetime, timedelta
from typing import Optional
import pandas as pd


class CSVSynthesizer:
    """Handles synthesis of tracking events into CSV format.
    
    This class converts tracking events with frame numbers into timestamped
    CSV records suitable for analysis and reporting.
    
    Attributes:
        fps: Frames per second of the source video
        columns: List of column names to include in output CSV
    
    Example:
        >>> from config import Config
        >>> config = Config.default()
        >>> synthesizer = CSVSynthesizer(config)
        >>> csv_df = synthesizer.process(events, "site_2024-01-15_14_30_00.mp4")
    """
    
    def __init__(self, config):
        """Initialize CSV synthesizer with configuration.
        
        Args:
            config: Config object containing video and output settings
        """
        self.fps = config.video.fps
        self.columns = config.output.csv_columns
    
    def extract_start_time(self, filename: str) -> datetime:
        """Extract the start time from a video filename.
        
        Expected filename format: site_YYYY-MM-DD_HH_MM_SS.mp4
        
        Args:
            filename: Video filename (with or without path)
            
        Returns:
            datetime object representing the video start time
            
        Raises:
            ValueError: If filename format is invalid
            
        Example:
            >>> synthesizer = CSVSynthesizer(config)
            >>> dt = synthesizer.extract_start_time("site_2024-01-15_14_30_00.mp4")
            >>> print(dt)
            2024-01-15 14:30:00
        """
        try:
            # Remove path and extension
            basename = filename.split('/')[-1].split('.')[0]
            
            # Parse components
            parts = basename.split('_')
            if len(parts) < 5:
                raise ValueError(f"Filename must have format: site_YYYY-MM-DD_HH_MM_SS, got: {basename}")
            
            site = parts[0]
            date_str = parts[1]
            hour = int(parts[2])
            minute = int(parts[3])
            second = int(parts[4])
            
            # Parse date
            date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Create time object
            time_obj = time(hour, minute, second)
            
            # Combine into datetime
            base_datetime = datetime.combine(date, time_obj)
            
            return base_datetime
            
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid filename format: {filename}. "
                f"Expected format: site_YYYY-MM-DD_HH_MM_SS.mp4"
            ) from e
    
    def frame_to_timestamp(self, frame_number: int, base_datetime: datetime) -> datetime:
        """Convert a frame number to a timestamp.
        
        Args:
            frame_number: Frame number in the video
            base_datetime: Video start time
            
        Returns:
            datetime object for the frame
            
        Example:
            >>> base = datetime(2024, 1, 15, 14, 30, 0)
            >>> timestamp = synthesizer.frame_to_timestamp(90, base)  # Frame 90 at 30fps = 3 seconds
            >>> print(timestamp)
            2024-01-15 14:30:03
        """
        seconds = frame_number / self.fps
        return base_datetime + timedelta(seconds=seconds)
    
    def process(self, events: pd.DataFrame, filename: str, 
                output_filename: Optional[str] = None) -> pd.DataFrame:
        """Process events DataFrame to add timestamps and format for output.
        
        Args:
            events: DataFrame with columns ['frame_number', 'nest', 'action', ...]
            filename: Source video filename for timestamp calculation
            output_filename: Optional custom output filename
            
        Returns:
            DataFrame with timestamps and configured columns
            
        Example:
            >>> events = pd.DataFrame({
            ...     'frame_number': [100, 200, 300],
            ...     'nest': ['1', '2', '1'],
            ...     'action': ['Exit', 'Entry', 'Entry']
            ... })
            >>> csv_df = synthesizer.process(events, "site_2024-01-15_14_30_00.mp4")
        """
        if events.empty:
            # Return empty DataFrame with correct columns
            empty_df = pd.DataFrame(columns=self.columns + ['filename'])
            return empty_df
        
        # Extract base datetime from filename
        base_datetime = self.extract_start_time(filename)
        
        # Add timestamps
        events = events.copy()
        events['timestamp'] = events['frame_number'].apply(
            lambda x: self.frame_to_timestamp(x, base_datetime)
        )

        #events['site'] = filename.split('/')[-1].split('_')[0]
        
        # Add filename
        #events['filename'] = output_filename if output_filename else filename
        
        # # Select and order columns
        # output_columns = self.columns + ['filename']
        # available_columns = [col for col in output_columns if col in events.columns]
        
        return events   #[available_columns]
    
    def save(self, events: pd.DataFrame, filename: str, output_path: str) -> str:
        """Process events and save to CSV file.
        
        Args:
            events: DataFrame with tracking events
            filename: Source video filename
            output_path: Path where CSV should be saved
            
        Returns:
            Path to saved CSV file
            
        Example:
            >>> csv_path = synthesizer.save(events, "site_2024-01-15_14_30_00.mp4", 
            ...                              "/output/results.csv")
        """
        processed_df = self.process(events, filename)
        processed_df.to_csv(output_path, index=False)
        return output_path
    
    def generate_csv(self, events: pd.DataFrame, video_path: str) -> pd.DataFrame:
        """Generate CSV with timestamps from events.
        
        This method is an alias for process() provided for API compatibility.
        
        Args:
            events: DataFrame with tracking events  
            video_path: Source video path for timestamp calculation
            
        Returns:
            Processed DataFrame with timestamps
            
        Example:
            >>> csv_df = synthesizer.generate_csv(events, "video.mp4")
        """
        return self.process(events, video_path)


# Alias for backward compatibility with BeeMonitor codebase
CSVGenerator = CSVSynthesizer


# Legacy function for backward compatibility
def processCSV(events: pd.DataFrame, filename: str, fps: int = 30) -> pd.DataFrame:
    """Legacy function for backward compatibility.
    
    Args:
        events: DataFrame with events
        filename: Video filename
        fps: Frames per second
        
    Returns:
        Processed DataFrame
        
    Note:
        This function is deprecated. Use CSVSynthesizer class instead.
    """
    # Create a minimal config for legacy support
    from dataclasses import dataclass, field
    
    @dataclass
    class LegacyVideoConfig:
        fps: int = fps
    
    @dataclass  
    class LegacyOutputConfig:
        csv_columns: list = field(default_factory=lambda: ["timestamp", "nest", "action"])
    
    @dataclass
    class LegacyConfig:
        video: LegacyVideoConfig = field(default_factory=LegacyVideoConfig)
        output: LegacyOutputConfig = field(default_factory=LegacyOutputConfig)
    
    config = LegacyConfig()
    synthesizer = CSVSynthesizer(config)
    return synthesizer.process(events, filename)


# Convenience function
def synthesize_csv(events: pd.DataFrame, filename: str, config, 
                   output_path: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to synthesize CSV with config.
    
    Args:
        events: DataFrame with tracking events
        filename: Source video filename
        config: Config object
        output_path: Optional path to save CSV
        
    Returns:
        Processed DataFrame
        
    Example:
        >>> from config import Config
        >>> config = Config.default()
        >>> csv_df = synthesize_csv(events, "video.mp4", config, "output.csv")
    """
    synthesizer = CSVSynthesizer(config)
    processed_df = synthesizer.process(events, filename)
    
    if output_path:
        processed_df.to_csv(output_path, index=False)
    
    return processed_df