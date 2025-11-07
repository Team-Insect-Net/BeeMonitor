"""Enhanced configuration with hotel box-aware parameter scaling.

This module extends the configuration system to support parameter scaling
based on both video resolution AND the hotel box position/distance from camera.
"""

import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Video processing configuration with auto-detection support."""
    res_width: int = 1920
    res_height: int = 1080
    fps: int = 30
    auto_detect_from_video: bool = True  # Automatically detect FPS and resolution from video
    
    def __post_init__(self):
        """Validate video configuration."""
        if self.res_width < 1 or self.res_height < 1:
            raise ValueError("Video resolution must be positive")
        if self.fps < 1:
            raise ValueError("FPS must be at least 1")
    
    def update_from_video(self, video_path: str) -> None:
        """Auto-detect and update FPS and resolution from video file.
        
        Args:
            video_path: Path to video file
            
        Raises:
            ValueError: If video cannot be opened
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            # Get FPS from video
            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            if detected_fps > 0:
                self.fps = int(round(detected_fps))
            
            # Get resolution from video
            detected_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            detected_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if detected_width > 0 and detected_height > 0:
                self.res_width = detected_width
                self.res_height = detected_height
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Auto-detected video properties: {self.res_width}x{self.res_height} @ {self.fps} FPS"
            )
        
        finally:
            cap.release()
    
    def get_video_properties(self, video_path: str) -> Tuple[int, int, int]:
        """Get video properties without modifying config.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (width, height, fps)
            
        Raises:
            ValueError: If video cannot be opened
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            return (width, height, fps)
        
        finally:
            cap.release()


@dataclass
class ModelConfig:
    """Model paths configuration."""
    nest_detection: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/nest.pt"
    tracking: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee_tracking.pt"
    classification: Optional[str] = None


@dataclass  
class HotelBoxConfig:
    """Hotel box position and distance configuration.
    
    These parameters define the physical hotel box location in the frame
    and are used to scale pixel-based thresholds appropriately.
    """
    # Hotel box position in frame (normalized 0-1 or pixel coordinates)
    x_center: float = 0.5  # Horizontal center position (0-1)
    y_center: float = 0.5  # Vertical center position (0-1)
    width_ratio: float = 0.7  # Width relative to frame width (0-1)
    height_ratio: float = 0.6  # Height relative to frame height (0-1)
    
    # Distance/scale parameters
    distance_factor: float = 1.0  # Scale factor based on camera distance (0.5-2.0)
    # 1.0 = reference distance
    # < 1.0 = hotel is closer (objects appear larger)
    # > 1.0 = hotel is farther (objects appear smaller)
    
    # Auto-detection settings
    auto_detect_box: bool = True  # Auto-detect hotel box from first frames
    min_box_confidence: float = 0.8  # Minimum confidence for auto-detection
    
    def __post_init__(self):
        """Validate hotel box configuration."""
        if not (0 <= self.x_center <= 1):
            raise ValueError("x_center must be between 0 and 1")
        if not (0 <= self.y_center <= 1):
            raise ValueError("y_center must be between 0 and 1")
        if not (0 < self.width_ratio <= 1):
            raise ValueError("width_ratio must be between 0 and 1")
        if not (0 < self.height_ratio <= 1):
            raise ValueError("height_ratio must be between 0 and 1")
        if not (0.1 <= self.distance_factor <= 5.0):
            raise ValueError("distance_factor must be between 0.1 and 5.0")
    
    def get_box_bounds(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Get pixel coordinates of hotel box.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        box_width = int(frame_width * self.width_ratio)
        box_height = int(frame_height * self.height_ratio)
        
        x_min = int(frame_width * self.x_center - box_width / 2)
        y_min = int(frame_height * self.y_center - box_height / 2)
        x_max = x_min + box_width
        y_max = y_min + box_height
        
        return (x_min, y_min, x_max, y_max)
    
    def get_scale_factor(self) -> float:
        """Get overall scale factor combining distance.
        
        Returns:
            Combined scale factor
        """
        return self.distance_factor


@dataclass
class NestConfig:
    """Nest detection configuration with hotel-aware scaling."""
    
    # Reference resolution (baseline for parameter tuning)
    reference_width: int = 1920
    reference_height: int = 1080
    reference_distance_factor: float = 1.0
    
    # Detection parameters
    confidence_threshold: float = 0.5
    min_detections: int = 35
    frame_skip: int = 30
    max_detection_attempts: int = 10
    nest_tube_class: int = 1
    hotel_class: int = 0
    
    # Grid parameters
    expected_total_nests: int = 60
    expected_rows: int = 6
    expected_nests_per_row: int = 10
    nest_count_tolerance: int = 3
    
    # Base pixel measurements (at reference resolution and distance)
    # Base nest size expressed as pixels at the reference resolution.
    # The existing nest_width() / nest_height() methods multiply these by
    # the runtime resolution scale (width / reference_width, height / reference_height)
    # and by any hotel distance factor, so these values represent the reference-size.
    # nest_width_base: float = 38.0
    # nest_height_base: float = 28.0

    nest_width_base: float = 24 #38.0 #  need to update these dynamically based on actual nest box detections
    nest_height_base: float = 14 #28.0

    # # Convenience ratio fields (derived from the reference) in case callers
    # # prefer working with proportions. These are not part of __init__.
    # nest_width_ratio: float = field(init=False, default=38.0 / 1920)
    # nest_height_ratio: float = field(init=False, default=28.0 / 1080)

    # @property
    # def nest_width_at_reference(self) -> float:
    #     """Pixel width of a nest at the configured reference resolution."""
    #     return self.nest_width_base

    # @property
    # def nest_height_at_reference(self) -> float:
    #     """Pixel height of a nest at the configured reference resolution."""
    #     return self.nest_height_base
    padding_x_base: float = 5.0
    padding_y_base: float = 7.0
    hotel_padding_x_base: float = 100.0
    hotel_padding_y_base: float = 50.0

    # Quality check tolerances (base values)
    spacing_tolerance_base: float = 25.0  # Tolerance for spacing between nests
    x_position_tolerance_base: float = 20.0  # Tolerance for X position
    y_position_tolerance_base: float = 15.0  # Tolerance for Y position

    
    # Dynamic padding options
    use_dynamic_padding: bool = True  # Calculate padding from actual nest spacing
    dynamic_padding_ratio: float = 0.15  # Padding as fraction of spacing (15% of gap)
    min_padding_x: float = 3.0  # Minimum X padding even with dynamic calculation
    min_padding_y: float = 3.0  # Minimum Y padding even with dynamic calculation
    
    # Quality check tolerances (base values) - RENAMED FOR CLARITY
    # These control how strict the quality checks are for nest detection
    nest_spacing_tolerance_base: float = 15.0  # Tolerance for spacing between nests in same row
    row_alignment_tolerance_base: float = 70.0  # Tolerance for vertical alignment within a row (Y-axis)
    column_alignment_tolerance_base: float = 12.0  # Tolerance for horizontal alignment within a column (X-axis)
    

    
    # Clustering thresholds (base values)
    row_threshold_base: float = 30.0  # Y-axis distance for row clustering
    col_threshold_base: float = 50.0  # X-axis distance for column clustering

    min_row_size: int = 6  # Minimum nests to form a valid row
    min_col_size: int = 10  # Minimum nests to form a valid column
    
    # Scaling methods for different parameters
    def get_scale_factor(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> Tuple[float, float]:
        """Calculate scale factors based on resolution and hotel position.
        
        Args:
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Tuple of (x_scale, y_scale)
        """
        # Basic resolution scaling
        x_scale = width / self.reference_width
        y_scale = height / self.reference_height
        
        # Apply distance/hotel scaling if available
        if hotel_box is not None:
            distance_scale = hotel_box.get_scale_factor()
            x_scale *= distance_scale
            y_scale *= distance_scale
        
        return (x_scale, y_scale)
    
    def nest_width(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled nest width."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return int(self.nest_width_base * x_scale)
    
    def nest_height(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled nest height."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return int(self.nest_height_base * y_scale)
    
    def padding_x(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled X padding."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return int(self.padding_x_base * x_scale)
    
    def padding_y(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled Y padding."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return int(self.padding_y_base * y_scale)
    
    def hotel_padding_x(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled hotel X padding."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return int(self.hotel_padding_x_base * x_scale)
    
    def hotel_padding_y(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled hotel Y padding."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return int(self.hotel_padding_y_base * y_scale)
    
    def spacing_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled spacing tolerance."""
        x_scale, y_scale = self.get_scale_factor(width, height, hotel_box)
        avg_scale = (x_scale + y_scale) / 2
        return self.spacing_tolerance_base * avg_scale * 2.5
    
    def x_position_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled X position tolerance."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return self.x_position_tolerance_base * x_scale * 3.0
    
    def y_position_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled Y position tolerance."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return self.y_position_tolerance_base * y_scale * 2.5
    
    def row_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled row clustering threshold (Y-axis distance).
        
        Args:
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Scaled row threshold for clustering detections into rows
        """
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return self.row_threshold_base * y_scale
    
    def col_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled column clustering threshold (X-axis distance).
        
        Args:
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Scaled column threshold for clustering detections into columns
        """
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return self.col_threshold_base * x_scale
    
    # NEW: Better-named tolerance methods for quality checks
    def nest_spacing_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled tolerance for horizontal spacing between nests in same row.
        
        This controls how much variation is allowed in the distance between 
        consecutive nests within a row during quality checks.
        
        Args:
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Scaled tolerance for nest spacing in pixels
        """
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return self.nest_spacing_tolerance_base * x_scale
    
    def row_alignment_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled tolerance for vertical alignment of nests within same row.
        
        This controls how much vertical (Y-axis) deviation is allowed for nests
        that should be in the same row during quality checks.
        
        Args:
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Scaled tolerance for row alignment in pixels
        """
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return self.row_alignment_tolerance_base * y_scale
    
    def column_alignment_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled tolerance for horizontal alignment of nests within same column.
        
        This controls how much horizontal (X-axis) deviation is allowed for nests
        that should be in the same column during quality checks.
        
        Args:
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Scaled tolerance for column alignment in pixels
        """
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return self.column_alignment_tolerance_base * x_scale
    
    def calculate_dynamic_padding(
        self,
        rows: List[List[Tuple[float, float]]],
        width: int,
        height: int,
        hotel_box: Optional[HotelBoxConfig] = None
    ) -> Tuple[int, int]:
        """Calculate dynamic padding based on actual nest spacing in detected rows.
        
        This method analyzes the actual spacing between nests and between rows
        to calculate appropriate padding that prevents overlapping boxes while
        fitting tightly around each nest.
        
        Args:
            rows: List of rows, each containing nest points (x, y)
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Tuple of (pad_x, pad_y) in pixels
        """
        if not self.use_dynamic_padding or not rows:
            # Fall back to static padding
            return self.padding_x(width, height, hotel_box), self.padding_y(height, height, hotel_box)
        
        # Calculate average horizontal spacing between nests in same row
        all_x_spacings = []
        for row in rows:
            if len(row) > 1:
                sorted_row = sorted(row, key=lambda p: p[0])
                x_positions = [p[0] for p in sorted_row]
                spacings = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
                all_x_spacings.extend(spacings)
        
        # Calculate average vertical spacing between rows
        all_y_spacings = []
        if len(rows) > 1:
            # Calculate average Y position for each row
            row_y_positions = []
            for row in rows:
                if row:
                    avg_y = sum(p[1] for p in row) / len(row)
                    row_y_positions.append(avg_y)
            
            # Sort by Y position
            row_y_positions.sort()
            
            # Calculate spacing between consecutive rows
            for i in range(len(row_y_positions)-1):
                spacing = row_y_positions[i+1] - row_y_positions[i]
                all_y_spacings.append(spacing)
        
        # Calculate padding as a fraction of the spacing
        if all_x_spacings:
            avg_x_spacing = sum(all_x_spacings) / len(all_x_spacings)
            pad_x = int(avg_x_spacing * self.dynamic_padding_ratio)
            pad_x = max(pad_x, int(self.min_padding_x))
        else:
            pad_x = self.padding_x(width, height, hotel_box)
        
        if all_y_spacings:
            avg_y_spacing = sum(all_y_spacings) / len(all_y_spacings)
            pad_y = int(avg_y_spacing * self.dynamic_padding_ratio)
            pad_y = max(pad_y, int(self.min_padding_y))
        else:
            pad_y = self.padding_y(width, height, hotel_box)
        
        return pad_x, pad_y


@dataclass
class TrackingConfig:
    """Tracking configuration with hotel-aware scaling."""
    
    # Reference resolution (baseline for parameter tuning)
    reference_width: int = 1920
    reference_height: int = 1080
    
    # Base tracking parameters (at reference resolution/distance)
    max_age: int = 30
    no_motion_frames: int = 30
    track_start_id: int = 0
    
    # Base distance thresholds
    distance_threshold_base: float = 80.0
    association_threshold_base: float = 200.0
    
    # Motion detection base parameters (at reference resolution)
    motion_threshold: int = 25  # Threshold for frame difference
    min_contour_area_base: float = 100.0  # Minimum contour area in pixels
    aspect_ratio_min: float = 0.3  # Minimum aspect ratio for detections
    aspect_ratio_max: float = 3.0  # Maximum aspect ratio for detections
    
    # Species tracking
    enable_species_tracking: bool = False
    label_map: Dict[int, str] = field(default_factory=lambda: {0: 'osmia_cornifrons'})

    tracking_classes: List[int] = field(default_factory=lambda: [0])  # Classes to track (e.g., bees and nest tubes)

    confidence_threshold: float = 0.5  # Confidence threshold for detections
    iou_threshold: float = 0.3  # IOU threshold for NMS
    
    def distance_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled distance threshold for tracking association.
        
        Args:
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Scaled distance threshold in pixels
        """
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.distance_threshold_base * x_scale
    
    def association_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled association threshold for track matching.
        
        Args:
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Scaled association threshold in pixels
        """
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.association_threshold_base * x_scale
    
    def min_contour_area(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled minimum contour area for motion detection.
        
        Args:
            width: Current frame width
            height: Current frame height
            hotel_box: Hotel box configuration (optional)
            
        Returns:
            Scaled minimum contour area in square pixels
        """
        # Area scales with both x and y dimensions
        x_scale = width / self.reference_width
        y_scale = height / self.reference_height
        
        if hotel_box is not None:
            distance_scale = hotel_box.get_scale_factor()
            x_scale *= distance_scale
            y_scale *= distance_scale
        
        # Area scales quadratically
        area_scale = x_scale * y_scale
        return self.min_contour_area_base * area_scale


@dataclass
class ProcessingConfig:
    """Processing configuration with hotel-aware scaling."""
    
    # Reference resolution (baseline for parameter tuning)
    reference_width: int = 1920
    reference_height: int = 1080
    
    # Trajectory parameters
    min_trajectory_length: int = 5
    
    # Window sizes (frame-based, don't scale)
    entry_window_size: int = 6
    exit_window_size: int = 3
    
    # Base padding values
    entry_padding_base: float = 10.0
    exit_padding_base: float = 20.0
    
    # Base speed thresholds
    start_speed_threshold_base: float = 10.0
    end_speed_threshold_base: float = 10.0
    
    def entry_padding(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled entry padding."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.entry_padding_base * x_scale
    
    def exit_padding(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled exit padding."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.exit_padding_base * x_scale
    
    def start_speed_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled start speed threshold."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.start_speed_threshold_base * x_scale
    
    def end_speed_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled end speed threshold."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.end_speed_threshold_base * x_scale


@dataclass
class OutputConfig:
    """Output generation configuration."""
    base_folder: str = "output"
    save_visualizations: bool = False
    save_intermediate_frames: bool = False
    video_codec: str = "mp4v"
    #video_fps: int = 30 # this should be dynamic based on input video fps
    csv_include_species: bool = True
    csv_columns: Optional[List[str]] = None  # If None, include all columns


@dataclass
class Config:
    """Main configuration with hotel box-aware parameter scaling.
    
    This configuration system automatically scales all pixel-based parameters
    based on both video resolution AND hotel box position/distance.
    
    Attributes:
        video: Video processing settings
        hotel_box: Hotel box position and distance settings
        models: Model file paths
        nest: Nest detection parameters
        tracking: Tracking algorithm parameters
        processing: Event processing parameters
        output: Output generation settings
    
    Example:
        >>> # Create config with custom hotel position
        >>> config = Config.default()
        >>> config.hotel_box.distance_factor = 1.5  # Hotel is farther away
        >>> config.video.res_width = 3840  # 4K video
        >>> 
        >>> # Parameters automatically scale
        >>> width, height = config.video.res_width, config.video.res_height
        >>> nest_width = config.nest.nest_width(width, height, config.hotel_box)
        >>> print(f"Scaled nest width: {nest_width}px")
    """
    
    video: VideoConfig = field(default_factory=VideoConfig)
    hotel_box: HotelBoxConfig = field(default_factory=HotelBoxConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    nest: NestConfig = field(default_factory=NestConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current resolution as tuple.
        
        Returns:
            Tuple of (width, height)
        """
        return (self.video.res_width, self.video.res_height)
    
    def get_nest_params(self, hotel_box: Optional[HotelBoxConfig] = None) -> Dict[str, Any]:
        """Get all scaled nest detection parameters.
        
        Args:
            hotel_box: Optional hotel box config (uses self.hotel_box if None)
            
        Returns:
            Dictionary of scaled parameters
        """
        if hotel_box is None:
            hotel_box = self.hotel_box
        
        width, height = self.resolution
        return {
            'nest_width': self.nest.nest_width(width, height, hotel_box),
            'nest_height': self.nest.nest_height(width, height, hotel_box),
            'padding_x': self.nest.padding_x(width, height, hotel_box),
            'padding_y': self.nest.padding_y(width, height, hotel_box),
            'hotel_padding_x': self.nest.hotel_padding_x(width, height, hotel_box),
            'hotel_padding_y': self.nest.hotel_padding_y(width, height, hotel_box),
            'spacing_tolerance': self.nest.spacing_tolerance(width, height, hotel_box),
            'x_position_tolerance': self.nest.x_position_tolerance(width, height, hotel_box),
            'y_position_tolerance': self.nest.y_position_tolerance(width, height, hotel_box),
        }
    
    def get_tracking_params(self, hotel_box: Optional[HotelBoxConfig] = None) -> Dict[str, Any]:
        """Get all scaled tracking parameters.
        
        Args:
            hotel_box: Optional hotel box config (uses self.hotel_box if None)
            
        Returns:
            Dictionary of scaled parameters
        """
        if hotel_box is None:
            hotel_box = self.hotel_box
        
        width, height = self.resolution
        return {
            'distance_threshold': self.tracking.distance_threshold(width, height, hotel_box),
            'association_threshold': self.tracking.association_threshold(width, height, hotel_box),
            'max_age': self.tracking.max_age,
            'no_motion_frames': self.tracking.no_motion_frames,
        }
    
    def get_processing_params(self, hotel_box: Optional[HotelBoxConfig] = None) -> Dict[str, Any]:
        """Get all scaled processing parameters.
        
        Args:
            hotel_box: Optional hotel box config (uses self.hotel_box if None)
            
        Returns:
            Dictionary of scaled parameters
        """
        if hotel_box is None:
            hotel_box = self.hotel_box
        
        width, height = self.resolution
        return {
            'entry_padding': self.processing.entry_padding(width, height, hotel_box),
            'exit_padding': self.processing.exit_padding(width, height, hotel_box),
            'start_speed_threshold': self.processing.start_speed_threshold(width, height, hotel_box),
            'end_speed_threshold': self.processing.end_speed_threshold(width, height, hotel_box),
            'min_trajectory_length': self.processing.min_trajectory_length,
        }
    
    def print_scaled_values(self) -> None:
        """Print all scaled parameter values for current resolution and hotel box."""
        width, height = self.resolution
        print(f"Configuration for resolution: {width}x{height}")
        print(f"Reference resolution: {self.nest.reference_width}x{self.nest.reference_height}")
        print(f"Hotel box distance factor: {self.hotel_box.distance_factor:.2f}")
        
        x_scale, y_scale = self.nest.get_scale_factor(width, height, self.hotel_box)
        print(f"Scale factors: X={x_scale:.3f}, Y={y_scale:.3f}")
        print()
        
        print("Hotel Box Configuration:")
        bounds = self.hotel_box.get_box_bounds(width, height)
        print(f"  Position: ({self.hotel_box.x_center:.2f}, {self.hotel_box.y_center:.2f})")
        print(f"  Size: {self.hotel_box.width_ratio:.2f} x {self.hotel_box.height_ratio:.2f}")
        print(f"  Bounds: {bounds}")
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
        """Create a Config instance with default values."""
        return cls()
    
    @classmethod
    def high_quality(cls) -> 'Config':
        """Create a Config optimized for high-quality detection."""
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
        """Create a Config optimized for speed."""
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
    def for_distance(cls, distance_factor: float) -> 'Config':
        """Create a Config for a specific camera distance.
        
        Args:
            distance_factor: Scale factor for distance
                1.0 = reference distance
                < 1.0 = closer (objects appear larger)
                > 1.0 = farther (objects appear smaller)
                
        Returns:
            Configured Config instance
        
        Example:
            >>> # Hotel is 50% farther than reference
            >>> config = Config.for_distance(1.5)
        """
        config = cls()
        config.hotel_box.distance_factor = distance_factor
        return config
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create a Config from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            Config instance
        """
        video_config = VideoConfig(**config_dict.get('video', {}))
        hotel_box_config = HotelBoxConfig(**config_dict.get('hotel_box', {}))
        models_config = ModelConfig(**config_dict.get('models', {}))
        nest_config = NestConfig(**config_dict.get('nest', {}))
        tracking_config = TrackingConfig(**config_dict.get('tracking', {}))
        processing_config = ProcessingConfig(**config_dict.get('processing', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        
        return cls(
            video=video_config,
            hotel_box=hotel_box_config,
            models=models_config,
            nest=nest_config,
            tracking=tracking_config,
            processing=processing_config,
            output=output_config
        )
    
    def to_dict(self) -> dict:
        """Convert Config to a dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate the entire configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Individual configs validate in __post_init__
        
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


# Convenience functions
def get_config_for_scenario(scenario: str, **kwargs) -> Config:
    """Get a pre-configured Config for common scenarios.
    
    Args:
        scenario: One of 'default', 'high_quality', 'fast', 'close', 'far'
        **kwargs: Additional parameters (e.g., distance_factor for distance-based configs)
        
    Returns:
        Configured Config instance
        
    Example:
        >>> config = get_config_for_scenario('far', distance_factor=2.0)
    """
    if scenario == 'default':
        return Config.default()
    
    elif scenario == 'high_quality':
        return Config.high_quality()
    
    elif scenario == 'fast':
        return Config.fast()
    
    elif scenario == 'close':
        # Hotel is closer than reference (objects appear larger)
        distance_factor = kwargs.get('distance_factor', 0.7)
        return Config.for_distance(distance_factor)
    
    elif scenario == 'far':
        # Hotel is farther than reference (objects appear smaller)
        distance_factor = kwargs.get('distance_factor', 1.5)
        return Config.for_distance(distance_factor)
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")