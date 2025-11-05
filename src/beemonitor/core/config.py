# """Configuration management for Bee Monitor."""

# import yaml
# from pathlib import Path
# from typing import Any, Dict, Optional, List
# from dataclasses import dataclass, field


# @dataclass
# class VideoConfig:
#     """Video processing configuration."""
#     height: int = 720
#     width: int = 1280
#     fps: int = 30


# @dataclass
# class ModelConfig:
#     """Model paths configuration."""
#     nest_detection: str = "models/nest_detection_model.pt"
#     tracking: str = "models/bee_tracking_model.pt"


# @dataclass
# class TrackingConfig:
#     """Tracking algorithm configuration."""
#     max_age: int = 30
#     distance_threshold: int = 100
#     association_threshold: int = 200
#     track_start_id: int = 0


# @dataclass
# class DetectionConfig:
#     """Detection parameters configuration."""
#     confidence_threshold: float = 0.25
#     iou_threshold: float = 0.5
#     motion_threshold: int = 5
#     min_contour_area: int = 100
#     aspect_ratio_min: float = 0.5
#     aspect_ratio_max: float = 2.0
#     tracking_classes: List[int] = field(default_factory=lambda: [3])  

# @dataclass
# class NestGridConfig:
#     """Configuration for nest grid structure."""
#     rows: int = 6
#     columns: int = 10
#     expected_total: int = 50
#     tolerance: int = 2
    
#     # Detection strategy
#     exhaustive_search: bool = True
#     max_frames_to_scan: int = 1000
    
#     # ID assignment
#     id_method: str = "grid_based"
    
#     # Reference matching
#     use_reference: bool = False
#     reference_path: Optional[str] = None

#     expected_columns: int = 10
#     expected_rows: int = 6
#     #min_nests_per_row: int = 10
#     #row_tolerance: int = 15
#     fill_missing: bool = True
#     auto_detect_rows: bool = True


# @dataclass
# class NestConfig:
#     """Nest detection configuration."""
#     confidence_threshold: float = 0.9
#     frame_skip: int = 30
#     min_detections: int = 60
#     row_threshold: int = 10
#     col_threshold: int = 10
#     min_row_size: int = 5
#     nest_width: int = 38
#     nest_height: int = 28
#     padding_x: int = 5
#     padding_y: int = 7
#     hotel_padding_x: int = 100
#     hotel_padding_y: int = 50
#     nest_class: int = 2
#     hotel_class: int = 1



# @dataclass
# class ProcessingConfig:
#     """Event processing configuration."""
#     window_size: int = 3
#     padding: int = 20
#     entry_window_size: int = 6
#     entry_padding: int = 10
#     exit_window_size: int = 3
#     exit_padding: int = 20
#     min_trajectory_length: int = 5
#     start_speed_threshold: int = 10
#     end_speed_threshold: int = 10


# @dataclass
# class OutputConfig:
#     """Output generation configuration."""
#     base_folder: str = "output"
#     save_visualizations: bool = False
#     save_intermediate_frames: bool = False
#     video_codec: str = "mp4v"
#     video_fps: int = 30
#     csv_columns: list = field(default_factory=lambda: ["timestamp", "nest", "action"])


# class Config:
#     """Main configuration class for Bee Monitor.
    
#     This class manages all configuration settings for the bee monitoring system.
#     It can load settings from YAML files or use default values.
    
#     Attributes:
#         video: Video processing settings
#         models: Model file paths
#         tracking: Tracking algorithm parameters
#         detection: Detection parameters
#         nest: Nest detection parameters
#         processing: Event processing parameters
#         output: Output generation settings
    
#     Example:
#         >>> config = Config.from_yaml("config/my_config.yaml")
#         >>> print(config.video.height)
#         720
#         >>> config.tracking.max_age = 50  # Modify settings
#     """
    
#     def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
#         """Initialize configuration from dictionary.
        
#         Args:
#             config_dict: Dictionary containing configuration settings.
#                         If None, uses default values.
#         """
#         if config_dict is None:
#             config_dict = {}
        
#         # Initialize sub-configurations
#         video_cfg = config_dict.get("video", {})
#         resolution = video_cfg.get("resolution", {})
#         self.video = VideoConfig(
#             height=resolution.get("height", 720),
#             width=resolution.get("width", 1280),
#             fps=video_cfg.get("fps", 30)
#         )
        
#         models_cfg = config_dict.get("models", {})
#         self.models = ModelConfig(**models_cfg) if models_cfg else ModelConfig()
        
#         tracking_cfg = config_dict.get("tracking", {})
#         self.tracking = TrackingConfig(**tracking_cfg) if tracking_cfg else TrackingConfig()
        
#         detection_cfg = config_dict.get("detection", {})
#         self.detection = DetectionConfig(**detection_cfg) if detection_cfg else DetectionConfig()

#         processing_cfg = config_dict.get("processing", {})
#         self.processing = ProcessingConfig(**processing_cfg) if processing_cfg else ProcessingConfig()

#         nest_grid_cfg = config_dict.get("nest_grid", {})
#         self.nest_grid = NestGridConfig(**nest_grid_cfg) if nest_grid_cfg else NestGridConfig()
        
#         nest_cfg = config_dict.get("nest", {})
#         self.nest = NestConfig(**nest_cfg) if nest_cfg else NestConfig()
       
#         processing_cfg = config_dict.get("processing", {})
#         self.processing = ProcessingConfig(**processing_cfg) if processing_cfg else ProcessingConfig()
        
#         output_cfg = config_dict.get("output", {})
#         self.output = OutputConfig(**output_cfg) if output_cfg else OutputConfig()
    
#     @classmethod
#     def from_yaml(cls, config_path: str) -> "Config":
#         """Load configuration from YAML file.
        
#         Args:
#             config_path: Path to YAML configuration file
            
#         Returns:
#             Config object with loaded settings
            
#         Raises:
#             FileNotFoundError: If config file doesn't exist
#             yaml.YAMLError: If config file is invalid
#         """
#         config_file = Path(config_path)
#         if not config_file.exists():
#             raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
#         with open(config_file, "r", encoding="utf-8") as f:
#             config_dict = yaml.safe_load(f)
        
#         return cls(config_dict)
    
#     @classmethod
#     def default(cls) -> "Config":
#         """Create configuration with default values.
        
#         Returns:
#             Config object with default settings
#         """
#         return cls()
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert configuration to dictionary.
        
#         Returns:
#             Dictionary representation of configuration
#         """
#         return {
#             "video": {
#                 "resolution": {
#                     "height": self.video.height,
#                     "width": self.video.width,
#                 },
#                 "fps": self.video.fps,
#             },
#             "models": {
#                 "nest_detection": self.models.nest_detection,
#                 "tracking": self.models.tracking,
#             },
#             "tracking": {
#                 "max_age": self.tracking.max_age,
#                 "distance_threshold": self.tracking.distance_threshold,
#                 "association_threshold": self.tracking.association_threshold,
#                 "track_start_id": self.tracking.track_start_id,
#             },
#             "detection": {
#                 "confidence_threshold": self.detection.confidence_threshold,
#                 "iou_threshold": self.detection.iou_threshold,
#                 "motion_threshold": self.detection.motion_threshold,
#                 "min_contour_area": self.detection.min_contour_area,
#                 "aspect_ratio_min": self.detection.aspect_ratio_min,
#                 "aspect_ratio_max": self.detection.aspect_ratio_max,
#             },
#         }
    
#     def save(self, output_path: str) -> None:
#         """Save configuration to YAML file.
        
#         Args:
#             output_path: Path where configuration will be saved
#         """
#         with open(output_path, "w", encoding="utf-8") as f:
#             yaml.dump(self.to_dict(), f, default_flow_style=False)


#__-____________________________________________________________________________________>>>>>>>>

# """Configuration module for BeeMonitor.

# This module defines configuration classes for all components of the bee monitoring system.
# """

# import yaml
# from pathlib import Path
# from typing import Any, Dict, Optional, List
# from dataclasses import dataclass, field


# @dataclass
# class NestConfig:
#     """Configuration for nest detection and processing.
    
#     This class contains all parameters for detecting and processing bee hotel nests,
#     including detection thresholds, quality criteria, and geometric parameters.
#     """
    
#     # Detection parameters
#     min_detections: int = 50
#     """Minimum number of nest holes to detect before stopping (should be close to expected_total_nests)"""
    
#     frame_skip: int = 30
#     """Number of frames to skip between detection attempts (30 frames ≈ 1 second at 30fps)"""
    
#     confidence_threshold: float = 0.9
#     """YOLO confidence threshold for nest detection (0.0-1.0)"""
    
#     max_detection_attempts: int = 10
#     """Maximum number of detection attempts before giving up"""
    
#     # Clustering parameters
#     row_threshold: int = 10
#     """Maximum Y-distance (pixels) for two nests to be in the same row"""
    
#     col_threshold: int = 10
#     """Maximum X-distance (pixels) for two nests to be in the same column (not heavily used)"""
    
#     min_row_size: int = 10
#     """Minimum number of nests in a row for it to be considered valid"""
    
#     # Geometric parameters for nest bounding boxes
#     nest_width: int = 38
#     """Base width of a single nest hole in pixels"""
    
#     nest_height: int = 28
#     """Base height of a single nest hole in pixels"""
    
#     padding_x: int = 5
#     """Additional horizontal padding around nest bounding box"""
    
#     padding_y: int = 7
#     """Additional vertical padding around nest bounding box"""
    
#     # Hotel ROI parameters
#     hotel_padding_x: int = 100
#     """Horizontal padding around the entire hotel"""
    
#     hotel_padding_y: int = 50
#     """Vertical padding around the entire hotel"""
    
#     # Quality check parameters - Expected structure
#     expected_total_nests: int = 60
#     """Expected total number of nest holes in the hotel (e.g., 6 rows × 10 columns = 60)"""
    
#     expected_rows: int = 6
#     """Expected number of rows in the hotel"""
    
#     expected_nests_per_row: int = 10
#     """Expected number of nests in each row"""
    
#     nest_count_tolerance: int = 0
#     """Allowed deviation from expected_total_nests (±tolerance)"""
    
#     # Quality check parameters - Spacing and alignment
#     spacing_tolerance: float = 30.0
#     """Maximum allowed deviation from average horizontal spacing between nests (pixels)"""
    
#     x_position_tolerance: float = 30.0
#     """Maximum allowed deviation from average X position for nests in same column (pixels)"""
    
#     y_position_tolerance: float = 30.0
#     """Maximum allowed deviation from average Y position for nests in same row (pixels)"""

#     nest_tube_class: int = 2
#     """YOLO class ID for nest tube detection (adjust based on your model)"""

#     hotel_class: int = 1
#     """YOLO class ID for hotel detection (adjust based on your model)"""
    
#     def __post_init__(self):
#         """Validate configuration after initialization."""
#         # Ensure expected values are consistent
#         if self.expected_rows * self.expected_nests_per_row != self.expected_total_nests:
#             raise ValueError(
#                 f"Inconsistent nest configuration: "
#                 f"{self.expected_rows} rows × {self.expected_nests_per_row} nests/row "
#                 f"!= {self.expected_total_nests} total nests"
#             )
        
#         # Validate ranges
#         if not 0.0 <= self.confidence_threshold <= 1.0:
#             raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
#         if self.min_detections < 1:
#             raise ValueError("min_detections must be at least 1")
        
#         if self.max_detection_attempts < 1:
#             raise ValueError("max_detection_attempts must be at least 1")


# @dataclass
# class TrackingConfig:
#     """Configuration for bee tracking.
    
#     This class contains parameters for tracking bee movements using
#     motion detection and object detection/tracking.
#     """
    
#     # Tracker parameters
#     max_age: int = 30
#     """Maximum number of frames to keep track alive without detection"""
    
#     association_threshold: float = 200.0
#     """Maximum distance (pixels) for associating detection with track"""
    
#     distance_threshold: float = 100.0
#     """Maximum distance (pixels) for motion prediction"""
    
#     # Motion detection parameters
#     motion_threshold: int = 5
#     """Pixel intensity threshold for motion detection"""
    
#     min_contour_area: int = 100
#     """Minimum contour area to consider as motion"""
    
#     aspect_ratio_min: float = 0.5
#     """Minimum aspect ratio (width/height) for valid detections"""
    
#     aspect_ratio_max: float = 2.0
#     """Maximum aspect ratio (width/height) for valid detections"""
    
#     no_motion_frames: int = 30
#     """Number of frames without detections before ending tracking"""
    
#     # YOLO inference parameters
#     iou_threshold: float = 0.5
#     """IoU threshold for YOLO detection"""

#     tracking_classes: List[int] = field(
#         default_factory=lambda: [0, 1, 2, 3]
#     )
#     """List of YOLO class IDs to track (e.g., [0, 1, 2, 3])"""

#     classes_map : Dict[int, str] = field(
#         default_factory=lambda: {0: 'Diptera', 1: 'Hymenoptera', 2: 'Lepidoptera', 3: 'Coleoptera'}
#     )
#     """Mapping of YOLO class IDs to names"""

#     track_species_confidence: bool = True
#     """Whether to track species classification confidence"""
    
#     species_colors: Dict[int, tuple] = field(default_factory=lambda: {
#         0: (0, 255, 0),      # Green - diptera
#         1: (255, 0, 0),      # Blue - hymenoptera (bees) 
#         2: (0, 0, 255),      # Red - lepidoptera
#         3: (255, 255, 0),    # Cyan - coleoptera
#     })
#     """BGR colors for visualizing different species"""


# @dataclass
# class ProcessingConfig:
#     """Configuration for processing trajectories into entry/exit events.
    
#     This class contains parameters for analyzing bee trajectories to
#     determine entry and exit events at nest holes.
#     """
    
#     # Window sizes for trajectory analysis
#     entry_window_size: int = 6
#     """Number of frames to analyze at end of trajectory for entry detection"""
    
#     exit_window_size: int = 3
#     """Number of frames to analyze at start of trajectory for exit detection"""
    
#     # Padding for nest hole regions
#     entry_padding: int = 10
#     """Pixels of padding around nest hole for entry detection"""
    
#     exit_padding: int = 20
#     """Pixels of padding around nest hole for exit detection"""
    
#     # Speed thresholds for classification
#     start_speed_threshold: float = 10.0
#     """Maximum starting speed to classify as exit"""
    
#     end_speed_threshold: float = 10.0
#     """Maximum ending speed to classify as entry"""
    
#     # Minimum trajectory length
#     min_trajectory_length: int = 5
#     """Minimum number of points in trajectory to process"""


# @dataclass
# class VideoConfig:
#     """Configuration for video processing.
    
#     This class contains parameters for reading and writing videos.
#     """
    
#     # Video resolution
#     res_height: int = 720
#     """Target height for video frames"""
    
#     res_width: int = 1280
#     """Target width for video frames"""
    
#     # Output parameters
#     fps: int = 30
#     """Frames per second for video processing"""
    
#     codec: str = 'mp4v'
#     """Video codec for output (e.g., 'mp4v', 'avc1')"""
    
#     visualization_enabled: bool = False
#     """Whether to generate visualization videos"""

# @dataclass
# class ModelConfig:
#     """Configuration for model file paths.
    
#     This class contains file paths for the various models used
#     in the bee monitoring system.
#     """
    
#     nest_detection: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/nest.pt"
#     """Path to the nest detection model file"""
    
#     tracking: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee.pt"
#     """Path to the bee tracking model file"""


# @dataclass
# class OutputConfig:
#     """Configuration for output generation.
    
#     This class contains parameters for saving outputs such as
#     videos and CSV files.
#     """
    
#     base_folder: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/output"
#     """Base folder for saving outputs"""
    
#     save_tracking_visualizations: bool = False
#     """Whether to save tracking visualization videos"""
    
#     save_nest_visualization_frame: bool = True
#     """Whether to save a frame showing detected nests"""
    
#     video_codec: str = "mp4v"
#     """Video codec for output videos"""
    
#     video_fps: int = 30
#     """Frames per second for output videos"""
    
#     csv_columns: List[str] = field(
#         default_factory=lambda: ["timestamp", "nest", "action"]
#     )
#     """Columns to include in output CSV files"""


# @dataclass
# class Config:
#     """Main configuration class for BeeMonitor.
    
#     This class aggregates all configuration components and provides
#     factory methods for creating common configurations.

#     Attributes:
#         models: ModelConfig
#         nest: NestConfig
#         tracking: TrackingConfig
#         processing: ProcessingConfig
#         video: VideoConfig
#         output: OutputConfig

    
#     Example:
#         >>> config = Config.default()
#         >>> config.nest.confidence_threshold = 0.85
#         >>> config.tracking.max_age = 40
#     """
#     models: ModelConfig = field(default_factory=ModelConfig)
#     """Model file paths configuration"""
    
#     nest: NestConfig = field(default_factory=NestConfig)
#     """Nest detection configuration"""
    
#     tracking: TrackingConfig = field(default_factory=TrackingConfig)
#     """Bee tracking configuration"""
    
#     processing: ProcessingConfig = field(default_factory=ProcessingConfig)
#     """Event processing configuration"""
    
#     video: VideoConfig = field(default_factory=VideoConfig)
#     """Video processing configuration"""

#     output : OutputConfig = field(default_factory=OutputConfig)
#     """Output generation configuration"""
    
#     @classmethod
#     def default(cls) -> 'Config':
#         """Create a Config instance with default values.
        
#         Returns:
#             Config instance with all default parameters
            
#         Example:
#             >>> config = Config.default()
#         """
#         return cls()
    
#     @classmethod
#     def high_quality(cls) -> 'Config':
#         """Create a Config optimized for high-quality detection.
        
#         Uses stricter thresholds and quality checks for more accurate
#         results at the cost of potentially more retries.
        
#         Returns:
#             Config instance optimized for quality
            
#         Example:
#             >>> config = Config.high_quality()
#         """
#         config = cls()
        
#         # Stricter nest detection
#         config.nest.confidence_threshold = 0.95
#         config.nest.min_detections = 60
#         config.nest.nest_count_tolerance = 0
#         config.nest.spacing_tolerance = 10.0
#         config.nest.x_position_tolerance = 10.0
#         config.nest.y_position_tolerance = 8.0
        
#         # More conservative tracking
#         config.tracking.max_age = 20
#         config.tracking.association_threshold = 150.0
        
#         return config
    
#     @classmethod
#     def fast(cls) -> 'Config':
#         """Create a Config optimized for speed.
        
#         Uses more lenient thresholds and fewer quality checks for
#         faster processing at the cost of potentially lower accuracy.
        
#         Returns:
#             Config instance optimized for speed
            
#         Example:
#             >>> config = Config.fast()
#         """
#         config = cls()
        
#         # Faster nest detection
#         config.nest.confidence_threshold = 0.8
#         config.nest.min_detections = 55
#         config.nest.max_detection_attempts = 5
#         config.nest.nest_count_tolerance = 5
#         config.nest.spacing_tolerance = 20.0
        
#         # Faster tracking
#         config.tracking.no_motion_frames = 20
        
#         return config
    
#     @classmethod
#     def from_dict(cls, config_dict: dict) -> 'Config':
#         """Create a Config from a dictionary.
        
#         Args:
#             config_dict: Dictionary with configuration values
            
#         Returns:
#             Config instance
            
#         Example:
#             >>> config_dict = {
#             ...     'nest': {'confidence_threshold': 0.85},
#             ...     'tracking': {'max_age': 40}
#             ... }
#             >>> config = Config.from_dict(config_dict)
#         """
#         models_config = ModelConfig(**config_dict.get('models', {}))
#         nest_config = NestConfig(**config_dict.get('nest', {}))
#         tracking_config = TrackingConfig(**config_dict.get('tracking', {}))
#         processing_config = ProcessingConfig(**config_dict.get('processing', {}))
#         video_config = VideoConfig(**config_dict.get('video', {}))
#         output = OutputConfig(**config_dict.get('output', {}))
        
#         return cls(
#             models=models_config,
#             nest=nest_config,
#             tracking=tracking_config,
#             processing=processing_config,
#             video=video_config,
#             output=output
#         )
    
#     def to_dict(self) -> dict:
#         """Convert Config to a dictionary.
        
#         Returns:
#             Dictionary representation of the configuration
            
#         Example:
#             >>> config = Config.default()
#             >>> config_dict = config.to_dict()
#         """
#         from dataclasses import asdict
#         return asdict(self)
    
#     def validate(self) -> bool:
#         """Validate the entire configuration.
        
#         Returns:
#             True if configuration is valid
            
#         Raises:
#             ValueError: If configuration is invalid
#         """
#         # NestConfig has its own validation in __post_init__
#         # Add any cross-component validation here
        
#         if self.tracking.max_age < 1:
#             raise ValueError("tracking.max_age must be at least 1")
        
#         if self.video.fps < 1:
#             raise ValueError("video.fps must be at least 1")
        
#         return True
    
#     @classmethod
#     def from_yaml(cls, config_path: str) -> "Config":
#         """Load configuration from YAML file.
        
#         Args:
#             config_path: Path to YAML configuration file
            
#         Returns:
#             Config object with loaded settings
            
#         Raises:
#             FileNotFoundError: If config file doesn't exist
#             yaml.YAMLError: If config file is invalid
#         """
#         config_file = Path(config_path)
#         if not config_file.exists():
#             raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
#         with open(config_file, "r", encoding="utf-8") as f:
#             config_dict = yaml.safe_load(f)
        
#         return cls(config_dict)
    
#     @classmethod
#     def default(cls) -> "Config":
#         """Create configuration with default values.
        
#         Returns:
#             Config object with default settings
#         """
#         return cls()
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert configuration to dictionary.
        
#         Returns:
#             Dictionary representation of configuration
#         """
#         return {
#             "video": {
#                 "resolution": {
#                     "height": self.video.height,
#                     "width": self.video.width,
#                 },
#                 "fps": self.video.fps,
#             },
#             "models": {
#                 "nest_detection": self.models.nest_detection,
#                 "tracking": self.models.tracking,
#             },
#             "tracking": {
#                 "max_age": self.tracking.max_age,
#                 "distance_threshold": self.tracking.distance_threshold,
#                 "association_threshold": self.tracking.association_threshold,
#                 "track_start_id": self.tracking.track_start_id,
#             },
#             "detection": {
#                 "confidence_threshold": self.detection.confidence_threshold,
#                 "iou_threshold": self.detection.iou_threshold,
#                 "motion_threshold": self.detection.motion_threshold,
#                 "min_contour_area": self.detection.min_contour_area,
#                 "aspect_ratio_min": self.detection.aspect_ratio_min,
#                 "aspect_ratio_max": self.detection.aspect_ratio_max,
#             },
#         }
    
#     def save(self, output_path: str) -> None:
#         """Save configuration to YAML file.
        
#         Args:
#             output_path: Path where configuration will be saved
#         """
#         with open(output_path, "w", encoding="utf-8") as f:
#             yaml.dump(self.to_dict(), f, default_flow_style=False)


# # Example configurations for common scenarios
# def get_config_for_scenario(scenario: str) -> Config:
#     """Get a pre-configured Config for common scenarios.
    
#     Args:
#         scenario: One of 'default', 'high_quality', 'fast', 'small_hotel', 'large_hotel'
        
#     Returns:
#         Configured Config instance
        
#     Example:
#         >>> config = get_config_for_scenario('high_quality')
#     """
#     if scenario == 'default':
#         return Config.default()
    
#     elif scenario == 'high_quality':
#         return Config.high_quality()
    
#     elif scenario == 'fast':
#         return Config.fast()
    
#     elif scenario == 'small_hotel':
#         # Configuration for smaller hotels (e.g., 4 rows × 8 columns)
#         config = Config.default()
#         config.nest.expected_total_nests = 32
#         config.nest.expected_rows = 4
#         config.nest.expected_nests_per_row = 8
#         config.nest.min_detections = 35
#         return config
    
#     elif scenario == 'large_hotel':
#         # Configuration for larger hotels (e.g., 8 rows × 12 columns)
#         config = Config.default()
#         config.nest.expected_total_nests = 96
#         config.nest.expected_rows = 8
#         config.nest.expected_nests_per_row = 12
#         config.nest.min_detections = 90
#         return config
    
#     else:
#         raise ValueError(f"Unknown scenario: {scenario}")




import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class NestConfig:
    """Configuration for nest detection and processing.
    
    This class contains all parameters for detecting and processing bee hotel nests.
    All pixel-based parameters are defined at a reference resolution (1280x720)
    and automatically scale when used with different resolutions.
    """
    
    # Reference resolution for base parameter values
    reference_width: int = 1280
    """Reference width for pixel-based parameters"""
    
    reference_height: int = 720
    """Reference height for pixel-based parameters"""
    
    # Detection parameters (resolution-independent)
    min_detections: int = 50
    """Minimum number of nest holes to detect before stopping"""
    
    frame_skip: int = 30
    """Number of frames to skip between detection attempts"""
    
    confidence_threshold: float = 0.9
    """YOLO confidence threshold for nest detection (0.0-1.0)"""
    
    max_detection_attempts: int = 10
    """Maximum number of detection attempts before giving up"""
    
    # Clustering parameters (at reference resolution)
    row_threshold_base: int = 10
    """Maximum Y-distance (pixels) for same row at reference resolution"""
    
    col_threshold_base: int = 10
    """Maximum X-distance (pixels) for same column at reference resolution"""
    
    min_row_size: int = 10
    """Minimum number of nests in a row (resolution-independent)"""
    
    # Geometric parameters for nest bounding boxes (at reference resolution)
    nest_width_base: int = 38
    """Base width of single nest hole at reference resolution"""
    
    nest_height_base: int = 28
    """Base height of single nest hole at reference resolution"""
    
    padding_x_base: int = 5
    """Additional horizontal padding at reference resolution"""
    
    padding_y_base: int = 7
    """Additional vertical padding at reference resolution"""
    
    # Hotel ROI parameters (at reference resolution)
    hotel_padding_x_base: int = 100
    """Horizontal padding around hotel at reference resolution"""
    
    hotel_padding_y_base: int = 50
    """Vertical padding around hotel at reference resolution"""
    
    # Quality check parameters - Expected structure (resolution-independent)
    expected_total_nests: int = 60
    """Expected total number of nest holes"""
    
    expected_rows: int = 6
    """Expected number of rows in the hotel"""
    
    expected_nests_per_row: int = 10
    """Expected number of nests in each row"""
    
    nest_count_tolerance: int = 0
    """Allowed deviation from expected_total_nests"""
    
    # Quality check parameters - Spacing and alignment (at reference resolution)
    spacing_tolerance_base: float = 30.0
    """Maximum deviation from average horizontal spacing at reference resolution"""
    
    x_position_tolerance_base: float = 30.0
    """Maximum deviation from average X position at reference resolution"""
    
    y_position_tolerance_base: float = 30.0
    """Maximum deviation from average Y position at reference resolution"""
    
    # YOLO classes (resolution-independent)
    nest_tube_class: int = 2
    """YOLO class ID for nest tube detection"""
    
    hotel_class: int = 1
    """YOLO class ID for hotel detection"""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure expected values are consistent
        if self.expected_rows * self.expected_nests_per_row != self.expected_total_nests:
            raise ValueError(
                f"Inconsistent nest configuration: "
                f"{self.expected_rows} rows × {self.expected_nests_per_row} nests/row "
                f"!= {self.expected_total_nests} total nests"
            )
        
        # Validate ranges
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if self.min_detections < 1:
            raise ValueError("min_detections must be at least 1")
        
        if self.max_detection_attempts < 1:
            raise ValueError("max_detection_attempts must be at least 1")
    
    def get_scale_factor(self, target_width: int, target_height: int) -> Tuple[float, float]:
        """Calculate scaling factors for target resolution.
        
        Args:
            target_width: Target video width
            target_height: Target video height
            
        Returns:
            Tuple of (x_scale, y_scale) factors
        """
        x_scale = target_width / self.reference_width
        y_scale = target_height / self.reference_height
        return x_scale, y_scale
    
    def scale_value(self, value: float, target_width: int, target_height: int, 
                    use_x: bool = True) -> int:
        """Scale a pixel value to target resolution.
        
        Args:
            value: Base value at reference resolution
            target_width: Target video width
            target_height: Target video height
            use_x: If True, use x_scale; if False, use y_scale
            
        Returns:
            Scaled value rounded to nearest integer
        """
        x_scale, y_scale = self.get_scale_factor(target_width, target_height)
        scale = x_scale if use_x else y_scale
        return int(round(value * scale))
    
    # Computed properties that scale with resolution
    def row_threshold(self, target_width: int, target_height: int) -> int:
        """Get row threshold scaled for target resolution."""
        return self.scale_value(self.row_threshold_base, target_width, target_height, use_x=False)
    
    def col_threshold(self, target_width: int, target_height: int) -> int:
        """Get column threshold scaled for target resolution."""
        return self.scale_value(self.col_threshold_base, target_width, target_height, use_x=True)
    
    def nest_width(self, target_width: int, target_height: int) -> int:
        """Get nest width scaled for target resolution."""
        return self.scale_value(self.nest_width_base, target_width, target_height, use_x=True)
    
    def nest_height(self, target_width: int, target_height: int) -> int:
        """Get nest height scaled for target resolution."""
        return self.scale_value(self.nest_height_base, target_width, target_height, use_x=False)
    
    def padding_x(self, target_width: int, target_height: int) -> int:
        """Get horizontal padding scaled for target resolution."""
        return self.scale_value(self.padding_x_base, target_width, target_height, use_x=True)
    
    def padding_y(self, target_width: int, target_height: int) -> int:
        """Get vertical padding scaled for target resolution."""
        return self.scale_value(self.padding_y_base, target_width, target_height, use_x=False)
    
    def hotel_padding_x(self, target_width: int, target_height: int) -> int:
        """Get hotel horizontal padding scaled for target resolution."""
        return self.scale_value(self.hotel_padding_x_base, target_width, target_height, use_x=True)
    
    def hotel_padding_y(self, target_width: int, target_height: int) -> int:
        """Get hotel vertical padding scaled for target resolution."""
        return self.scale_value(self.hotel_padding_y_base, target_width, target_height, use_x=False)
    
    def spacing_tolerance(self, target_width: int, target_height: int) -> float:
        """Get spacing tolerance scaled for target resolution."""
        x_scale, _ = self.get_scale_factor(target_width, target_height)
        return self.spacing_tolerance_base * x_scale
    
    def x_position_tolerance(self, target_width: int, target_height: int) -> float:
        """Get X position tolerance scaled for target resolution."""
        x_scale, _ = self.get_scale_factor(target_width, target_height)
        return self.x_position_tolerance_base * x_scale
    
    def y_position_tolerance(self, target_width: int, target_height: int) -> float:
        """Get Y position tolerance scaled for target resolution."""
        _, y_scale = self.get_scale_factor(target_width, target_height)
        return self.y_position_tolerance_base * y_scale


@dataclass
class TrackingConfig:
    """Configuration for bee tracking.
    
    This class contains parameters for tracking bee movements.
    Pixel-based parameters automatically scale with resolution.
    """
    
    # Reference resolution
    reference_width: int = 1280
    """Reference width for pixel-based parameters"""
    
    reference_height: int = 720
    """Reference height for pixel-based parameters"""
    
    # Tracker parameters (resolution-independent)
    max_age: int = 30
    """Maximum number of frames to keep track alive without detection"""
    
    # Distance thresholds (at reference resolution)
    association_threshold_base: float = 200.0
    """Maximum distance for associating detection with track at reference resolution"""
    
    distance_threshold_base: float = 100.0
    """Maximum distance for motion prediction at reference resolution"""
    
    # Motion detection parameters (at reference resolution)
    motion_threshold: int = 5
    """Pixel intensity threshold for motion detection (resolution-independent)"""
    
    min_contour_area_base: int = 100
    """Minimum contour area at reference resolution"""
    
    # Aspect ratio (resolution-independent)
    aspect_ratio_min: float = 0.5
    """Minimum aspect ratio (width/height) for valid detections"""
    
    aspect_ratio_max: float = 2.0
    """Maximum aspect ratio (width/height) for valid detections"""
    
    no_motion_frames: int = 30
    """Number of frames without detections before ending tracking"""
    
    # YOLO inference parameters (resolution-independent)
    iou_threshold: float = 0.5
    """IoU threshold for YOLO detection"""
    
    tracking_classes: List[int] = field(
        default_factory=lambda: [0]
    )
    """List of YOLO class IDs to track"""

    species_map : Dict[int, str] = field(
        #default_factory=lambda: {0: 'Diptera', 1: 'Hymenoptera', 2: 'Lepidoptera', 3: 'Coleoptera'}
        default_factory=lambda: {0: 'osmia_cornifrons'}
    )
    """Mapping of YOLO class IDs to names"""

    track_classes_confidence: bool = True
    """Whether to track classes classification confidence"""

    classes_colors: Dict[int, tuple] = field(default_factory=lambda: {
        0: (0, 255, 0),      # Green - diptera
        # 1: (255, 0, 0),      # Blue - hymenoptera (bees) 
        # 2: (0, 0, 255),      # Red - lepidoptera
        # 3: (255, 255, 0),    # Cyan - coleoptera
    })

    bee_classification: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee_classification.pt"
    """Path to the bee species classification model file"""
    
    def get_scale_factor(self, target_width: int, target_height: int) -> Tuple[float, float]:
        """Calculate scaling factors for target resolution.
        
        Args:
            target_width: Target video width
            target_height: Target video height
            
        Returns:
            Tuple of (x_scale, y_scale) factors
        """
        x_scale = target_width / self.reference_width
        y_scale = target_height / self.reference_height
        return x_scale, y_scale
    
    def association_threshold(self, target_width: int, target_height: int) -> float:
        """Get association threshold scaled for target resolution.
        
        Uses average of x and y scales since this is a distance threshold.
        """
        x_scale, y_scale = self.get_scale_factor(target_width, target_height)
        avg_scale = (x_scale + y_scale) / 2
        return self.association_threshold_base * avg_scale
    
    def distance_threshold(self, target_width: int, target_height: int) -> float:
        """Get distance threshold scaled for target resolution.
        
        Uses average of x and y scales since this is a distance threshold.
        """
        x_scale, y_scale = self.get_scale_factor(target_width, target_height)
        avg_scale = (x_scale + y_scale) / 2
        return self.distance_threshold_base * avg_scale
    
    def min_contour_area(self, target_width: int, target_height: int) -> int:
        """Get minimum contour area scaled for target resolution.
        
        Area scales with both dimensions, so we use the product.
        """
        x_scale, y_scale = self.get_scale_factor(target_width, target_height)
        area_scale = x_scale * y_scale
        return int(round(self.min_contour_area_base * area_scale))


@dataclass
class ProcessingConfig:
    """Configuration for processing trajectories into entry/exit events.
    
    This class contains parameters for analyzing bee trajectories.
    Pixel-based parameters automatically scale with resolution.
    """
    
    # Reference resolution
    reference_width: int = 1280
    """Reference width for pixel-based parameters"""
    
    reference_height: int = 720
    """Reference height for pixel-based parameters"""
    
    # Window sizes (resolution-independent)
    entry_window_size: int = 6
    """Number of frames to analyze at end of trajectory for entry detection"""
    
    exit_window_size: int = 3
    """Number of frames to analyze at start of trajectory for exit detection"""
    
    # Padding for nest hole regions (at reference resolution)
    entry_padding_base: int = 10
    """Pixels of padding around nest hole for entry detection at reference resolution"""
    
    exit_padding_base: int = 20
    """Pixels of padding around nest hole for exit detection at reference resolution"""
    
    # Speed thresholds (at reference resolution)
    start_speed_threshold_base: float = 10.0
    """Maximum starting speed to classify as exit at reference resolution"""
    
    end_speed_threshold_base: float = 10.0
    """Maximum ending speed to classify as entry at reference resolution"""
    
    # Minimum trajectory length (resolution-independent)
    min_trajectory_length: int = 5
    """Minimum number of points in trajectory to process"""
    
    def get_scale_factor(self, target_width: int, target_height: int) -> Tuple[float, float]:
        """Calculate scaling factors for target resolution."""
        x_scale = target_width / self.reference_width
        y_scale = target_height / self.reference_height
        return x_scale, y_scale
    
    def entry_padding(self, target_width: int, target_height: int) -> int:
        """Get entry padding scaled for target resolution."""
        x_scale, y_scale = self.get_scale_factor(target_width, target_height)
        avg_scale = (x_scale + y_scale) / 2
        return int(round(self.entry_padding_base * avg_scale))
    
    def exit_padding(self, target_width: int, target_height: int) -> int:
        """Get exit padding scaled for target resolution."""
        x_scale, y_scale = self.get_scale_factor(target_width, target_height)
        avg_scale = (x_scale + y_scale) / 2
        return int(round(self.exit_padding_base * avg_scale))
    
    def start_speed_threshold(self, target_width: int, target_height: int) -> float:
        """Get start speed threshold scaled for target resolution."""
        x_scale, y_scale = self.get_scale_factor(target_width, target_height)
        avg_scale = (x_scale + y_scale) / 2
        return self.start_speed_threshold_base * avg_scale
    
    def end_speed_threshold(self, target_width: int, target_height: int) -> float:
        """Get end speed threshold scaled for target resolution."""
        x_scale, y_scale = self.get_scale_factor(target_width, target_height)
        avg_scale = (x_scale + y_scale) / 2
        return self.end_speed_threshold_base * avg_scale


@dataclass
class VideoConfig:
    """Configuration for video processing.
    
    This class contains parameters for reading and writing videos.
    """
    
    # Video resolution
    res_height: int = 720
    """Target height for video frames"""
    
    res_width: int = 1280
    """Target width for video frames"""
    
    # Output parameters
    fps: int = 30
    """Frames per second for video processing"""
    
    codec: str = 'mp4v'
    """Video codec for output (e.g., 'mp4v', 'avc1')"""
    
    visualization_enabled: bool = False
    """Whether to generate visualization videos"""


@dataclass
class ModelConfig:
    """Configuration for model file paths.
    
    This class contains file paths for the various models used
    in the bee monitoring system.
    """
    
    nest_detection: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/nest.pt"
    """Path to the nest detection model file"""
    
    #tracking: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee_original.pt"
    tracking: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee_tracking.pt"
    """Path to the bee tracking model file"""

    bee_classification: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee_tracking.pt"
    """Path to the bee species classification model file"""


@dataclass
class OutputConfig:
    """Configuration for output generation.
    
    This class contains parameters for saving outputs such as
    videos and CSV files.
    """
    
    base_folder: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/output"
    """Base folder for saving outputs"""
    
    save_tracking_visualizations: bool = False
    """Whether to save tracking visualization videos"""
    
    save_nest_visualization_frame: bool = True
    """Whether to save a frame showing detected nests"""
    
    video_codec: str = "mp4v"
    """Video codec for output videos"""
    
    video_fps: int = 30
    """Frames per second for output videos"""
    
    csv_columns: List[str] = field(
        default_factory=lambda: ["timestamp", "nest", "action"]
    )
    """Columns to include in output CSV files"""


@dataclass
class Config:
    """Main configuration class for BeeMonitor.
    
    This class aggregates all configuration components and provides
    factory methods for creating common configurations. All pixel-based
    parameters automatically scale with the video resolution specified
    in VideoConfig.
    
    Attributes:
        models: ModelConfig
        nest: NestConfig
        tracking: TrackingConfig
        processing: ProcessingConfig
        video: VideoConfig
        output: OutputConfig
    
    Example:
        >>> config = Config.default()
        >>> # All pixel values will scale with resolution
        >>> config.video.res_width = 1920
        >>> config.video.res_height = 1080
        >>> # Now nest.nest_width(1920, 1080) returns scaled value
    """
    models: ModelConfig = field(default_factory=ModelConfig)
    """Model file paths configuration"""
    
    nest: NestConfig = field(default_factory=NestConfig)
    """Nest detection configuration"""
    
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    """Bee tracking configuration"""
    
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    """Event processing configuration"""
    
    video: VideoConfig = field(default_factory=VideoConfig)
    """Video processing configuration"""
    
    output: OutputConfig = field(default_factory=OutputConfig)
    """Output generation configuration"""
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current video resolution as (width, height)."""
        return (self.video.res_width, self.video.res_height)
    
    def get_nest_params(self) -> Dict[str, Any]:
        """Get all nest parameters scaled for current resolution.
        
        Returns:
            Dictionary with scaled parameter values
            
        Example:
            >>> config = Config.default()
            >>> params = config.get_nest_params()
            >>> print(params['nest_width'])  # Scaled for current resolution
        """
        width, height = self.resolution
        return {
            'row_threshold': self.nest.row_threshold(width, height),
            'col_threshold': self.nest.col_threshold(width, height),
            'nest_width': self.nest.nest_width(width, height),
            'nest_height': self.nest.nest_height(width, height),
            'padding_x': self.nest.padding_x(width, height),
            'padding_y': self.nest.padding_y(width, height),
            'hotel_padding_x': self.nest.hotel_padding_x(width, height),
            'hotel_padding_y': self.nest.hotel_padding_y(width, height),
            'spacing_tolerance': self.nest.spacing_tolerance(width, height),
            'x_position_tolerance': self.nest.x_position_tolerance(width, height),
            'y_position_tolerance': self.nest.y_position_tolerance(width, height),
        }
    
    def get_tracking_params(self) -> Dict[str, Any]:
        """Get all tracking parameters scaled for current resolution.
        
        Returns:
            Dictionary with scaled parameter values
        """
        width, height = self.resolution
        return {
            'association_threshold': self.tracking.association_threshold(width, height),
            'distance_threshold': self.tracking.distance_threshold(width, height),
            'min_contour_area': self.tracking.min_contour_area(width, height),
        }
    
    def get_processing_params(self) -> Dict[str, Any]:
        """Get all processing parameters scaled for current resolution.
        
        Returns:
            Dictionary with scaled parameter values
        """
        width, height = self.resolution
        return {
            'entry_padding': self.processing.entry_padding(width, height),
            'exit_padding': self.processing.exit_padding(width, height),
            'start_speed_threshold': self.processing.start_speed_threshold(width, height),
            'end_speed_threshold': self.processing.end_speed_threshold(width, height),
        }
    
    def print_scaled_values(self) -> None:
        """Print all scaled parameter values for current resolution.
        
        Useful for debugging and understanding how parameters scale.
        """
        width, height = self.resolution
        print(f"Configuration for resolution: {width}x{height}")
        print(f"Reference resolution: {self.nest.reference_width}x{self.nest.reference_height}")
        
        x_scale, y_scale = self.nest.get_scale_factor(width, height)
        print(f"Scale factors: X={x_scale:.3f}, Y={y_scale:.3f}")
        print()
        
        print("Nest Detection Parameters:")
        nest_params = self.get_nest_params()
        for key, value in nest_params.items():
            print(f"  {key}: {value}")
        print()
        
        print("Tracking Parameters:")
        tracking_params = self.get_tracking_params()
        for key, value in tracking_params.items():
            print(f"  {key}: {value}")
        print()
        
        print("Processing Parameters:")
        processing_params = self.get_processing_params()
        for key, value in processing_params.items():
            print(f"  {key}: {value}")
    
    @classmethod
    def default(cls) -> 'Config':
        """Create a Config instance with default values.
        
        Returns:
            Config instance with all default parameters
            
        Example:
            >>> config = Config.default()
        """
        return cls()
    
    @classmethod
    def high_quality(cls) -> 'Config':
        """Create a Config optimized for high-quality detection.
        
        Uses stricter thresholds and quality checks for more accurate
        results at the cost of potentially more retries.
        
        Returns:
            Config instance optimized for quality
            
        Example:
            >>> config = Config.high_quality()
        """
        config = cls()
        
        # Stricter nest detection
        config.nest.confidence_threshold = 0.95
        config.nest.min_detections = 60
        config.nest.nest_count_tolerance = 0
        config.nest.spacing_tolerance_base = 10.0
        config.nest.x_position_tolerance_base = 10.0
        config.nest.y_position_tolerance_base = 8.0
        
        # More conservative tracking
        config.tracking.max_age = 20
        config.tracking.association_threshold_base = 150.0
        
        return config
    
    @classmethod
    def fast(cls) -> 'Config':
        """Create a Config optimized for speed.
        
        Uses more lenient thresholds and fewer quality checks for
        faster processing at the cost of potentially lower accuracy.
        
        Returns:
            Config instance optimized for speed
            
        Example:
            >>> config = Config.fast()
        """
        config = cls()
        
        # Faster nest detection
        config.nest.confidence_threshold = 0.8
        config.nest.min_detections = 55
        config.nest.max_detection_attempts = 5
        config.nest.nest_count_tolerance = 5
        config.nest.spacing_tolerance_base = 20.0
        
        # Faster tracking
        config.tracking.no_motion_frames = 20
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create a Config from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            Config instance
            
        Example:
            >>> config_dict = {
            ...     'nest': {'confidence_threshold': 0.85},
            ...     'tracking': {'max_age': 40}
            ... }
            >>> config = Config.from_dict(config_dict)
        """
        models_config = ModelConfig(**config_dict.get('models', {}))
        nest_config = NestConfig(**config_dict.get('nest', {}))
        tracking_config = TrackingConfig(**config_dict.get('tracking', {}))
        processing_config = ProcessingConfig(**config_dict.get('processing', {}))
        video_config = VideoConfig(**config_dict.get('video', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        
        return cls(
            models=models_config,
            nest=nest_config,
            tracking=tracking_config,
            processing=processing_config,
            video=video_config,
            output=output_config
        )
    
    def to_dict(self) -> dict:
        """Convert Config to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
            
        Example:
            >>> config = Config.default()
            >>> config_dict = config.to_dict()
        """
        from dataclasses import asdict
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate the entire configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # NestConfig has its own validation in __post_init__
        # Add any cross-component validation here
        
        if self.tracking.max_age < 1:
            raise ValueError("tracking.max_age must be at least 1")
        
        if self.video.fps < 1:
            raise ValueError("video.fps must be at least 1")
        
        return True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config object with loaded settings
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def save_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path where configuration will be saved
        """
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Example configurations for common scenarios
def get_config_for_scenario(scenario: str) -> Config:
    """Get a pre-configured Config for common scenarios.
    
    Args:
        scenario: One of 'default', 'high_quality', 'fast', 'small_hotel', 'large_hotel'
        
    Returns:
        Configured Config instance
        
    Example:
        >>> config = get_config_for_scenario('high_quality')
    """
    if scenario == 'default':
        return Config.default()
    
    elif scenario == 'high_quality':
        return Config.high_quality()
    
    elif scenario == 'fast':
        return Config.fast()
    
    elif scenario == 'small_hotel':
        # Configuration for smaller hotels (e.g., 4 rows × 8 columns)
        config = Config.default()
        config.nest.expected_total_nests = 32
        config.nest.expected_rows = 4
        config.nest.expected_nests_per_row = 8
        config.nest.min_detections = 35
        return config
    
    elif scenario == 'large_hotel':
        # Configuration for larger hotels (e.g., 8 rows × 12 columns)
        config = Config.default()
        config.nest.expected_total_nests = 96
        config.nest.expected_rows = 8
        config.nest.expected_nests_per_row = 12
        config.nest.min_detections = 90
        return config
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")