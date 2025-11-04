"""Simplified tracking configuration with automatic parameter optimization.

This module provides:
1. Reduced parameter set (4 instead of 10+)
2. Preset configurations for common scenarios
3. Automatic parameter adaptation based on video characteristics
4. Parameter tuning utilities
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedTrackingConfig:
    """Simplified tracking configuration with only essential parameters.
    
    Replaces 10+ parameters with 4 key settings:
    - track_persistence: How long to keep tracks without detection
    - max_association_distance: Maximum distance for matching
    - detection_quality: Overall detection quality preset
    - motion_sensitivity: Motion detection sensitivity
    """
    
    # Core parameters
    track_persistence: int = 30  # frames
    max_association_distance: int = 150  # pixels
    detection_quality: str = 'medium'  # 'low', 'medium', 'high'
    motion_sensitivity: str = 'medium'  # 'low', 'medium', 'high'
    
    # Internal mappings (computed from presets)
    _confidence_threshold: float = 0.25
    _iou_threshold: float = 0.5
    _motion_threshold: int = 5
    _min_contour_area: int = 100
    
    def __post_init__(self):
        """Compute internal parameters from presets."""
        self._apply_quality_preset()
        self._apply_motion_preset()
    
    def _apply_quality_preset(self):
        """Map detection_quality to confidence/IoU thresholds."""
        quality_map = {
            'low': {'confidence': 0.15, 'iou': 0.3},
            'medium': {'confidence': 0.25, 'iou': 0.5},
            'high': {'confidence': 0.35, 'iou': 0.7}
        }
        
        preset = quality_map.get(self.detection_quality, quality_map['medium'])
        self._confidence_threshold = preset['confidence']
        self._iou_threshold = preset['iou']
    
    def _apply_motion_preset(self):
        """Map motion_sensitivity to motion threshold and contour area."""
        motion_map = {
            'low': {'threshold': 10, 'min_area': 200},
            'medium': {'threshold': 5, 'min_area': 100},
            'high': {'threshold': 2, 'min_area': 50}
        }
        
        preset = motion_map.get(self.motion_sensitivity, motion_map['medium'])
        self._motion_threshold = preset['threshold']
        self._min_contour_area = preset['min_area']
    
    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold
    
    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold
    
    @property
    def motion_threshold(self) -> int:
        return self._motion_threshold
    
    @property
    def min_contour_area(self) -> int:
        return self._min_contour_area
    
    @property
    def max_age(self) -> int:
        """Alias for backward compatibility."""
        return self.track_persistence
    
    @property
    def distance_threshold(self) -> int:
        """Use max_association_distance for prediction threshold."""
        return int(self.max_association_distance * 0.7)
    
    @property
    def association_threshold(self) -> int:
        """Use max_association_distance for matching threshold."""
        return self.max_association_distance
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'track_persistence': self.track_persistence,
            'max_association_distance': self.max_association_distance,
            'detection_quality': self.detection_quality,
            'motion_sensitivity': self.motion_sensitivity,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'motion_threshold': self.motion_threshold,
            'min_contour_area': self.min_contour_area
        }
    
    def __repr__(self) -> str:
        return (
            f"SimplifiedTrackingConfig("
            f"persistence={self.track_persistence}, "
            f"distance={self.max_association_distance}, "
            f"quality='{self.detection_quality}', "
            f"motion='{self.motion_sensitivity}')"
        )


class TrackingPresets:
    """Predefined tracking configurations for common scenarios."""
    
    FAST = SimplifiedTrackingConfig(
        track_persistence=15,
        max_association_distance=100,
        detection_quality='low',
        motion_sensitivity='low'
    )
    
    BALANCED = SimplifiedTrackingConfig(
        track_persistence=30,
        max_association_distance=150,
        detection_quality='medium',
        motion_sensitivity='medium'
    )
    
    ACCURATE = SimplifiedTrackingConfig(
        track_persistence=45,
        max_association_distance=200,
        detection_quality='high',
        motion_sensitivity='high'
    )
    
    HIGH_DENSITY = SimplifiedTrackingConfig(
        track_persistence=20,
        max_association_distance=80,
        detection_quality='high',
        motion_sensitivity='high'
    )
    
    LOW_ACTIVITY = SimplifiedTrackingConfig(
        track_persistence=60,
        max_association_distance=200,
        detection_quality='medium',
        motion_sensitivity='low'
    )
    
    @classmethod
    def get(cls, preset_name: str = 'balanced') -> SimplifiedTrackingConfig:
        """Get a preset configuration.
        
        Args:
            preset_name: One of 'fast', 'balanced', 'accurate', 'high_density', 'low_activity'
            
        Returns:
            SimplifiedTrackingConfig object
            
        Example:
            >>> config = TrackingPresets.get('accurate')
            >>> print(config.track_persistence)
            45
        """
        preset_map = {
            'fast': cls.FAST,
            'balanced': cls.BALANCED,
            'accurate': cls.ACCURATE,
            'high_density': cls.HIGH_DENSITY,
            'low_activity': cls.LOW_ACTIVITY
        }
        
        preset = preset_map.get(preset_name.lower())
        if preset is None:
            logger.warning(f"Unknown preset '{preset_name}', using 'balanced'")
            preset = cls.BALANCED
        
        logger.info(f"Using preset: {preset_name}")
        return preset
    
    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """List available presets with descriptions."""
        return {
            'fast': 'Fast processing, lower accuracy - for quick testing',
            'balanced': 'Balanced speed and accuracy - recommended for most cases',
            'accurate': 'Highest accuracy, slower - for important analyses',
            'high_density': 'Optimized for many bees in frame',
            'low_activity': 'Optimized for infrequent bee activity'
        }


class AdaptiveTrackingConfig:
    """Automatically adapt tracking parameters based on video characteristics."""
    
    @staticmethod
    def from_video(
        video_path: str,
        base_preset: str = 'balanced'
    ) -> SimplifiedTrackingConfig:
        """Create config adapted to video characteristics.
        
        Analyzes video to determine optimal parameters.
        
        Args:
            video_path: Path to video file
            base_preset: Starting preset to adapt
            
        Returns:
            Adapted SimplifiedTrackingConfig
            
        Example:
            >>> config = AdaptiveTrackingConfig.from_video("video.mp4")
            >>> print(f"Adapted parameters: {config}")
        """
        # Start with preset
        config = TrackingPresets.get(base_preset)
        
        # Analyze video
        video_stats = AdaptiveTrackingConfig._analyze_video(video_path)
        
        if video_stats is None:
            logger.warning("Could not analyze video, using base preset")
            return config
        
        # Adapt parameters
        config.track_persistence = AdaptiveTrackingConfig._adapt_persistence(
            video_stats['fps']
        )
        
        config.max_association_distance = AdaptiveTrackingConfig._adapt_distance(
            video_stats['width'],
            video_stats['height']
        )
        
        config.detection_quality = AdaptiveTrackingConfig._adapt_quality(
            video_stats['brightness_std']
        )
        
        # Recompute internal parameters
        config.__post_init__()
        
        logger.info(f"Adapted config: {config}")
        return config
    
    @staticmethod
    def _analyze_video(video_path: str) -> Optional[Dict]:
        """Analyze video characteristics.
        
        Returns:
            Dict with keys: fps, width, height, brightness_std, noise_level
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames to assess quality
            n_samples = min(10, total_frames)
            frame_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
            
            brightness_values = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_values.append(gray.mean())
            
            cap.release()
            
            brightness_std = np.std(brightness_values) if brightness_values else 0
            
            return {
                'fps': fps,
                'width': width,
                'height': height,
                'brightness_std': brightness_std,
                'total_frames': total_frames
            }
        
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return None
    
    @staticmethod
    def _adapt_persistence(fps: float) -> int:
        """Adapt track persistence to frame rate.
        
        Rule: Keep tracks for approximately 1 second.
        """
        return max(15, int(fps * 1.0))
    
    @staticmethod
    def _adapt_distance(width: int, height: int) -> int:
        """Adapt association distance to resolution.
        
        Rule: Use 10-12% of frame diagonal.
        """
        diagonal = np.sqrt(width**2 + height**2)
        return int(0.12 * diagonal)
    
    @staticmethod
    def _adapt_quality(brightness_std: float) -> str:
        """Adapt detection quality to lighting conditions.
        
        Higher variation -> lower confidence threshold.
        """
        if brightness_std > 30:
            return 'low'  # Variable lighting, be less strict
        elif brightness_std < 15:
            return 'high'  # Stable lighting, be more strict
        else:
            return 'medium'


class ParameterValidator:
    """Validate tracking parameters for reasonableness."""
    
    RANGES = {
        'track_persistence': (5, 120),
        'max_association_distance': (30, 500),
        'confidence_threshold': (0.05, 0.95),
        'iou_threshold': (0.1, 0.9),
        'motion_threshold': (1, 20),
        'min_contour_area': (10, 1000)
    }
    
    @classmethod
    def validate(cls, config: SimplifiedTrackingConfig) -> Tuple[bool, list]:
        """Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check track_persistence
        min_p, max_p = cls.RANGES['track_persistence']
        if not min_p <= config.track_persistence <= max_p:
            warnings.append(
                f"track_persistence={config.track_persistence} outside "
                f"recommended range [{min_p}, {max_p}]"
            )
        
        # Check max_association_distance
        min_d, max_d = cls.RANGES['max_association_distance']
        if not min_d <= config.max_association_distance <= max_d:
            warnings.append(
                f"max_association_distance={config.max_association_distance} "
                f"outside recommended range [{min_d}, {max_d}]"
            )
        
        # Check confidence_threshold
        min_c, max_c = cls.RANGES['confidence_threshold']
        if not min_c <= config.confidence_threshold <= max_c:
            warnings.append(
                f"confidence_threshold={config.confidence_threshold} outside "
                f"recommended range [{min_c}, {max_c}]"
            )
        
        # Check detection_quality
        if config.detection_quality not in ['low', 'medium', 'high']:
            warnings.append(
                f"detection_quality='{config.detection_quality}' not recognized. "
                "Use 'low', 'medium', or 'high'"
            )
        
        # Check motion_sensitivity
        if config.motion_sensitivity not in ['low', 'medium', 'high']:
            warnings.append(
                f"motion_sensitivity='{config.motion_sensitivity}' not recognized. "
                "Use 'low', 'medium', or 'high'"
            )
        
        is_valid = len(warnings) == 0
        
        if not is_valid:
            logger.warning(f"Configuration validation found {len(warnings)} issues:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("Configuration validated successfully")
        
        return is_valid, warnings


# ============================================================================
# Migration Utilities
# ============================================================================

def migrate_old_config(old_config: Dict) -> SimplifiedTrackingConfig:
    """Migrate old configuration format to new simplified format.
    
    Args:
        old_config: Dictionary with old parameter names
        
    Returns:
        SimplifiedTrackingConfig object
        
    Example:
        >>> old = {
        ...     'max_age': 30,
        ...     'distance_threshold': 100,
        ...     'confidence_threshold': 0.25
        ... }
        >>> new_config = migrate_old_config(old)
    """
    # Map old parameters to new
    track_persistence = old_config.get('max_age', 30)
    
    # Average of old distance parameters
    distance_threshold = old_config.get('distance_threshold', 100)
    association_threshold = old_config.get('association_threshold', 200)
    max_association_distance = int((distance_threshold + association_threshold) / 2)
    
    # Infer quality from confidence
    confidence = old_config.get('confidence_threshold', 0.25)
    if confidence < 0.2:
        quality = 'low'
    elif confidence > 0.3:
        quality = 'high'
    else:
        quality = 'medium'
    
    # Infer motion sensitivity from threshold
    motion_threshold = old_config.get('motion_threshold', 5)
    if motion_threshold > 7:
        motion = 'low'
    elif motion_threshold < 3:
        motion = 'high'
    else:
        motion = 'medium'
    
    new_config = SimplifiedTrackingConfig(
        track_persistence=track_persistence,
        max_association_distance=max_association_distance,
        detection_quality=quality,
        motion_sensitivity=motion
    )
    
    logger.info("Migrated old configuration to new format")
    logger.info(f"  Old: max_age={track_persistence}, "
                f"distance={distance_threshold}, conf={confidence}")
    logger.info(f"  New: {new_config}")
    
    return new_config


def print_parameter_comparison():
    """Print comparison of old vs new parameter sets."""
    print("\n" + "="*80)
    print("TRACKING PARAMETER COMPARISON")
    print("="*80)
    print("\nOLD PARAMETERS (10+):")
    print("  ✗ max_age")
    print("  ✗ distance_threshold")
    print("  ✗ association_threshold")
    print("  ✗ track_start_id")
    print("  ✗ confidence_threshold")
    print("  ✗ iou_threshold")
    print("  ✗ motion_threshold")
    print("  ✗ min_contour_area")
    print("  ✗ aspect_ratio_min")
    print("  ✗ aspect_ratio_max")
    
    print("\nNEW PARAMETERS (4):")
    print("  ✓ track_persistence      - How long to remember tracks (frames)")
    print("  ✓ max_association_distance - Maximum matching distance (pixels)")
    print("  ✓ detection_quality       - 'low', 'medium', or 'high'")
    print("  ✓ motion_sensitivity      - 'low', 'medium', or 'high'")
    
    print("\nPRESET CONFIGURATIONS:")
    for name, description in TrackingPresets.list_presets().items():
        print(f"  • {name:15s} - {description}")
    
    print("\nRECOMMENDATION:")
    print("  Start with: TrackingPresets.get('balanced')")
    print("  Or use:     AdaptiveTrackingConfig.from_video('your_video.mp4')")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Demo usage
    print_parameter_comparison()
    
    print("\nEXAMPLE USAGE:\n")
    
    print("# 1. Use preset")
    print("config = TrackingPresets.get('balanced')")
    config = TrackingPresets.get('balanced')
    print(f"Result: {config}\n")
    
    print("# 2. Customize preset")
    print("config = TrackingPresets.get('accurate')")
    print("config.track_persistence = 60  # Increase for slower bees")
    config = TrackingPresets.get('accurate')
    config.track_persistence = 60
    print(f"Result: {config}\n")
    
    print("# 3. Auto-adapt to video (simulated)")
    print("config = AdaptiveTrackingConfig.from_video('video.mp4')")
    print("# Would analyze video and adapt parameters automatically\n")