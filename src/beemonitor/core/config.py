"""Configuration management for Bee Monitor."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class VideoConfig:
    """Video processing configuration."""
    height: int = 720
    width: int = 1280
    fps: int = 30


@dataclass
class ModelConfig:
    """Model paths configuration."""
    nest_detection: str = "models/nest_detection_model.pt"
    tracking: str = "models/bee_tracking_model.pt"


@dataclass
class TrackingConfig:
    """Tracking algorithm configuration."""
    max_age: int = 30
    distance_threshold: int = 100
    association_threshold: int = 200
    track_start_id: int = 0


@dataclass
class DetectionConfig:
    """Detection parameters configuration."""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    motion_threshold: int = 5
    min_contour_area: int = 100
    aspect_ratio_min: float = 0.5
    aspect_ratio_max: float = 2.0
    tracking_classes: List[int] = field(default_factory=lambda: [3])  

@dataclass
class NestGridConfig:
    """Configuration for nest grid structure."""
    rows: int = 6
    columns: int = 10
    expected_total: int = 50
    tolerance: int = 2
    
    # Detection strategy
    exhaustive_search: bool = True
    max_frames_to_scan: int = 1000
    
    # ID assignment
    id_method: str = "grid_based"
    
    # Reference matching
    use_reference: bool = False
    reference_path: Optional[str] = None

    expected_columns: int = 10
    min_nests_per_row: int = 10
    row_tolerance: int = 15
    fill_missing: bool = True
    auto_detect_rows: bool = True


@dataclass
class NestConfig:
    """Nest detection configuration."""
    confidence_threshold: float = 0.9
    frame_skip: int = 30
    min_detections: int = 60
    row_threshold: int = 10
    col_threshold: int = 10
    min_row_size: int = 5
    nest_width: int = 38
    nest_height: int = 28
    padding_x: int = 5
    padding_y: int = 7
    hotel_padding_x: int = 100
    hotel_padding_y: int = 50
    nest_class: int = 2


@dataclass
class ProcessingConfig:
    """Event processing configuration."""
    window_size: int = 3
    padding: int = 20
    entry_window_size: int = 6
    entry_padding: int = 10
    exit_window_size: int = 3
    exit_padding: int = 20
    min_trajectory_length: int = 5
    start_speed_threshold: int = 10
    end_speed_threshold: int = 10


@dataclass
class OutputConfig:
    """Output generation configuration."""
    base_folder: str = "output"
    save_visualizations: bool = False
    save_intermediate_frames: bool = False
    video_codec: str = "mp4v"
    video_fps: int = 30
    csv_columns: list = field(default_factory=lambda: ["timestamp", "nest", "action"])


class Config:
    """Main configuration class for Bee Monitor.
    
    This class manages all configuration settings for the bee monitoring system.
    It can load settings from YAML files or use default values.
    
    Attributes:
        video: Video processing settings
        models: Model file paths
        tracking: Tracking algorithm parameters
        detection: Detection parameters
        nest: Nest detection parameters
        processing: Event processing parameters
        output: Output generation settings
    
    Example:
        >>> config = Config.from_yaml("config/my_config.yaml")
        >>> print(config.video.height)
        720
        >>> config.tracking.max_age = 50  # Modify settings
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration settings.
                        If None, uses default values.
        """
        if config_dict is None:
            config_dict = {}
        
        # Initialize sub-configurations
        video_cfg = config_dict.get("video", {})
        resolution = video_cfg.get("resolution", {})
        self.video = VideoConfig(
            height=resolution.get("height", 720),
            width=resolution.get("width", 1280),
            fps=video_cfg.get("fps", 30)
        )
        
        models_cfg = config_dict.get("models", {})
        self.models = ModelConfig(**models_cfg) if models_cfg else ModelConfig()
        
        tracking_cfg = config_dict.get("tracking", {})
        self.tracking = TrackingConfig(**tracking_cfg) if tracking_cfg else TrackingConfig()
        
        detection_cfg = config_dict.get("detection", {})
        self.detection = DetectionConfig(**detection_cfg) if detection_cfg else DetectionConfig()

        processing_cfg = config_dict.get("processing", {})
        self.processing = ProcessingConfig(**processing_cfg) if processing_cfg else ProcessingConfig()

        nest_grid_cfg = config_dict.get("nest_grid", {})
        self.nest_grid = NestGridConfig(**nest_grid_cfg) if nest_grid_cfg else NestGridConfig()
        
        nest_cfg = config_dict.get("nest", {})
        if nest_cfg:
            padding = nest_cfg.get("padding", {})
            hotel_padding = nest_cfg.get("hotel_padding", {})
            self.nest = NestConfig(
                confidence_threshold=nest_cfg.get("confidence_threshold", 0.9),
                frame_skip=nest_cfg.get("frame_skip", 30),
                min_detections=nest_cfg.get("min_detections", 60),
                row_threshold=nest_cfg.get("row_threshold", 10),
                col_threshold=nest_cfg.get("col_threshold", 10),
                min_row_size=nest_cfg.get("min_row_size", 5),
                nest_width=nest_cfg.get("nest_width", 38),
                nest_height=nest_cfg.get("nest_height", 28),
                padding_x=padding.get("x", 5),
                padding_y=padding.get("y", 7),
                hotel_padding_x=hotel_padding.get("x", 100),
                hotel_padding_y=hotel_padding.get("y", 50)
            )
        else:
            self.nest = NestConfig()
        
        processing_cfg = config_dict.get("processing", {})
        if processing_cfg:
            speed_threshold = processing_cfg.get("speed_threshold", {})
            self.processing = ProcessingConfig(
                window_size=processing_cfg.get("window_size", 3),
                padding=processing_cfg.get("padding", 20),
                entry_window_size=processing_cfg.get("entry_window_size", 6),
                entry_padding=processing_cfg.get("entry_padding", 10),
                exit_window_size=processing_cfg.get("exit_window_size", 3),
                exit_padding=processing_cfg.get("exit_padding", 20),
                min_trajectory_length=processing_cfg.get("min_trajectory_length", 5),
                start_speed_threshold=speed_threshold.get("start", 10),
                end_speed_threshold=speed_threshold.get("end", 10)
            )
        else:
            self.processing = ProcessingConfig()
        
        output_cfg = config_dict.get("output", {})
        self.output = OutputConfig(**output_cfg) if output_cfg else OutputConfig()
    
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
