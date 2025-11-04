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


"""Configuration module for BeeMonitor.

This module defines configuration classes for all components of the bee monitoring system.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class NestConfig:
    """Configuration for nest detection and processing.
    
    This class contains all parameters for detecting and processing bee hotel nests,
    including detection thresholds, quality criteria, and geometric parameters.
    """
    
    # Detection parameters
    min_detections: int = 50
    """Minimum number of nest holes to detect before stopping (should be close to expected_total_nests)"""
    
    frame_skip: int = 30
    """Number of frames to skip between detection attempts (30 frames ≈ 1 second at 30fps)"""
    
    confidence_threshold: float = 0.9
    """YOLO confidence threshold for nest detection (0.0-1.0)"""
    
    max_detection_attempts: int = 10
    """Maximum number of detection attempts before giving up"""
    
    # Clustering parameters
    row_threshold: int = 10
    """Maximum Y-distance (pixels) for two nests to be in the same row"""
    
    col_threshold: int = 10
    """Maximum X-distance (pixels) for two nests to be in the same column (not heavily used)"""
    
    min_row_size: int = 10
    """Minimum number of nests in a row for it to be considered valid"""
    
    # Geometric parameters for nest bounding boxes
    nest_width: int = 38
    """Base width of a single nest hole in pixels"""
    
    nest_height: int = 28
    """Base height of a single nest hole in pixels"""
    
    padding_x: int = 5
    """Additional horizontal padding around nest bounding box"""
    
    padding_y: int = 7
    """Additional vertical padding around nest bounding box"""
    
    # Hotel ROI parameters
    hotel_padding_x: int = 100
    """Horizontal padding around the entire hotel"""
    
    hotel_padding_y: int = 50
    """Vertical padding around the entire hotel"""
    
    # Quality check parameters - Expected structure
    expected_total_nests: int = 60
    """Expected total number of nest holes in the hotel (e.g., 6 rows × 10 columns = 60)"""
    
    expected_rows: int = 6
    """Expected number of rows in the hotel"""
    
    expected_nests_per_row: int = 10
    """Expected number of nests in each row"""
    
    nest_count_tolerance: int = 0
    """Allowed deviation from expected_total_nests (±tolerance)"""
    
    # Quality check parameters - Spacing and alignment
    spacing_tolerance: float = 30.0
    """Maximum allowed deviation from average horizontal spacing between nests (pixels)"""
    
    x_position_tolerance: float = 30.0
    """Maximum allowed deviation from average X position for nests in same column (pixels)"""
    
    y_position_tolerance: float = 30.0
    """Maximum allowed deviation from average Y position for nests in same row (pixels)"""
    
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


@dataclass
class TrackingConfig:
    """Configuration for bee tracking.
    
    This class contains parameters for tracking bee movements using
    motion detection and object detection/tracking.
    """
    
    # Tracker parameters
    max_age: int = 30
    """Maximum number of frames to keep track alive without detection"""
    
    association_threshold: float = 200.0
    """Maximum distance (pixels) for associating detection with track"""
    
    distance_threshold: float = 100.0
    """Maximum distance (pixels) for motion prediction"""
    
    # Motion detection parameters
    motion_threshold: int = 5
    """Pixel intensity threshold for motion detection"""
    
    min_contour_area: int = 100
    """Minimum contour area to consider as motion"""
    
    aspect_ratio_min: float = 0.5
    """Minimum aspect ratio (width/height) for valid detections"""
    
    aspect_ratio_max: float = 2.0
    """Maximum aspect ratio (width/height) for valid detections"""
    
    no_motion_frames: int = 30
    """Number of frames without detections before ending tracking"""
    
    # YOLO inference parameters
    iou_threshold: float = 0.5
    """IoU threshold for YOLO detection"""
    
    detection_class: int = 3
    """YOLO class ID for bee detection (adjust based on your model)"""


@dataclass
class ProcessingConfig:
    """Configuration for processing trajectories into entry/exit events.
    
    This class contains parameters for analyzing bee trajectories to
    determine entry and exit events at nest holes.
    """
    
    # Window sizes for trajectory analysis
    entry_window_size: int = 6
    """Number of frames to analyze at end of trajectory for entry detection"""
    
    exit_window_size: int = 3
    """Number of frames to analyze at start of trajectory for exit detection"""
    
    # Padding for nest hole regions
    entry_padding: int = 10
    """Pixels of padding around nest hole for entry detection"""
    
    exit_padding: int = 20
    """Pixels of padding around nest hole for exit detection"""
    
    # Speed thresholds for classification
    start_speed_threshold: float = 10.0
    """Maximum starting speed to classify as exit"""
    
    end_speed_threshold: float = 10.0
    """Maximum ending speed to classify as entry"""
    
    # Minimum trajectory length
    min_trajectory_length: int = 5
    """Minimum number of points in trajectory to process"""


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
    
    tracking: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee.pt"
    """Path to the bee tracking model file"""


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
    factory methods for creating common configurations.

    Attributes:
        models: ModelConfig
        nest: NestConfig
        tracking: TrackingConfig
        processing: ProcessingConfig
        video: VideoConfig
        output: OutputConfig

    
    Example:
        >>> config = Config.default()
        >>> config.nest.confidence_threshold = 0.85
        >>> config.tracking.max_age = 40
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

    output : OutputConfig = field(default_factory=OutputConfig)
    """Output generation configuration"""
    
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
        config.nest.spacing_tolerance = 10.0
        config.nest.x_position_tolerance = 10.0
        config.nest.y_position_tolerance = 8.0
        
        # More conservative tracking
        config.tracking.max_age = 20
        config.tracking.association_threshold = 150.0
        
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
        config.nest.spacing_tolerance = 20.0
        
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
        output = OutputConfig(**config_dict.get('output', {}))
        
        return cls(
            models=models_config,
            nest=nest_config,
            tracking=tracking_config,
            processing=processing_config,
            video=video_config,
            output=output
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
        
        return cls(config_dict)
    
    @classmethod
    def default(cls) -> "Config":
        """Create configuration with default values.
        
        Returns:
            Config object with default settings
        """
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "video": {
                "resolution": {
                    "height": self.video.height,
                    "width": self.video.width,
                },
                "fps": self.video.fps,
            },
            "models": {
                "nest_detection": self.models.nest_detection,
                "tracking": self.models.tracking,
            },
            "tracking": {
                "max_age": self.tracking.max_age,
                "distance_threshold": self.tracking.distance_threshold,
                "association_threshold": self.tracking.association_threshold,
                "track_start_id": self.tracking.track_start_id,
            },
            "detection": {
                "confidence_threshold": self.detection.confidence_threshold,
                "iou_threshold": self.detection.iou_threshold,
                "motion_threshold": self.detection.motion_threshold,
                "min_contour_area": self.detection.min_contour_area,
                "aspect_ratio_min": self.detection.aspect_ratio_min,
                "aspect_ratio_max": self.detection.aspect_ratio_max,
            },
        }
    
    def save(self, output_path: str) -> None:
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
