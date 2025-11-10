# # """Hybrid motion detection and tracking system with adaptive cold-start handling.

# # Combines background subtraction (efficient) with deep learning (accurate)
# # for multi-insect tracking in bee hotel videos.

# # Key features:
# # - Hybrid BG/DL logic: BG for simple cases, DL for complex
# # - Elliptical search regions oriented by velocity direction
# # - ADAPTIVE cold-start: Large radius for new tracks, narrows as confidence grows
# # - STRICT distance enforcement for DL detections (within search region only)
# # - Fully adaptive thresholds learned from actual bee behavior
# # - Dynamic blob filtering based on actual blob contour areas
# # - AI noise filter (optional fourth filter)
# # - Aggressive duplicate prevention and track resurrection
# # - Solidity filter for shape compactness
# # - IoU-based overlap prevention
# # - Kalman filtering with velocity damping
# # - ROI masking for efficient motion detection
# # """

# # import logging
# # from typing import Dict, List, Tuple, Optional
# # import cv2
# # import numpy as np
# # import pandas as pd
# # import os
# # from dataclasses import dataclass

# # from beemonitor.core.config import Config

# # logger = logging.getLogger(__name__)

# # # Type aliases
# # BBox = Tuple[float, float, float, float]
# # Point = Tuple[float, float]


# # @dataclass
# # class TrackState:
# #     """State for a tracked insect."""
# #     track_id: int
# #     bbox: BBox
# #     centroid: Point
# #     kalman: cv2.KalmanFilter
# #     frames_without_detection: int
# #     label: str
# #     age: int
# #     last_yolo_confirmation: int
# #     trajectory_history: list = None
    
# #     def __post_init__(self):
# #         if self.trajectory_history is None:
# #             self.trajectory_history = []


# # class MotionTracking:
# #     """Standalone hybrid tracker with adaptive cold-start handling.
    
# #     Learns from actual bee behavior instead of using hard-coded values:
# #     - Blob size threshold from actual blob contour areas
# #     - Distance thresholds from observed movement
# #     - Elliptical search regions oriented by velocity direction
# #     - Adaptive cold-start: large radius for new tracks
    
# #     Attributes:
# #         model: YOLO model for detection/confirmation
# #         config: Configuration object
# #         bg_subtractor: Background subtractor (MOG2)
# #         noise_filter: Optional AI noise filter
# #         tracks: Dictionary of active tracks
# #         next_track_id: Next available track ID
# #         d_initial: Initial distance threshold
# #         d_max: Maximum observed distance between frames
# #         recorded_speeds: History of observed speeds
# #         recorded_bee_areas: History of actual blob contour areas
# #         min_blob_area_dynamic: Learned minimum blob area
# #     """
    
# #     def __init__(self, model, config: Optional[Config] = None, use_gpu: Optional[bool] = None):
# #         """Initialize hybrid tracker with adaptive parameters.
        
# #         Args:
# #             model: YOLO model for confirmation
# #             config: Configuration object
# #             use_gpu: Use GPU if available (default: auto-detect)
# #         """
# #         self.model = model
# #         self.config = config if config is not None else Config.default()
        
# #         # Auto-detect GPU
# #         if use_gpu is None:
# #             self.use_gpu = self._detect_gpu()
# #         else:
# #             self.use_gpu = use_gpu
        
# #         # Apply scaling
# #         self._apply_config_scalling(self.config)
        
# #         # Initialize background subtractor
# #         self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
# #             history=500,
# #             varThreshold=16,
# #             detectShadows=False
# #         )
        
# #         # Initialize AI noise filter (optional)
# #         self.noise_filter = None
# #         try:
# #             from beemonitor.ml.bee_noise_filter import BeeNoiseFilter
            
# #             # Try to find noise filter model
# #             noise_filter_path = os.path.join(
# #                 os.path.dirname(__file__), 
# #                 "/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/training_output2/best_model.pth"
# #             )
            
# #             # Also check user-provided path in config
# #             if hasattr(config, 'noise_filter_path') and config.noise_filter_path:
# #                 noise_filter_path = config.noise_filter_path
            
# #             if os.path.exists(noise_filter_path):
# #                 self.noise_filter = BeeNoiseFilter(
# #                     model_path=noise_filter_path,
# #                     device='cpu',
# #                     noise_threshold=0.9,
# #                     image_size=64
# #                 )
# #                 logger.info(f"✅ AI noise filter loaded: {noise_filter_path}")
# #             else:
# #                 logger.warning(f"⚠️ Noise filter model not found: {noise_filter_path}")
# #                 logger.warning("Tracking will continue without AI noise filtering")
# #         except ImportError as e:
# #             logger.warning(f"⚠️ BeeNoiseFilter not available (import error): {e}")
# #             logger.warning("Tracking will continue without AI noise filtering")
# #         except Exception as e:
# #             logger.warning(f"⚠️ Could not initialize noise filter: {e}")
# #             logger.warning("Tracking will continue without AI noise filtering")
        
# #         # Tracking state
# #         self.tracks: Dict[int, TrackState] = {}
# #         self.next_track_id = 0
        
# #         # Adaptive threshold parameters (from paper)
# #         self.d_initial = getattr(self.config.tracking, 'initial_distance_threshold', 30.0)
# #         self.d_max = self.d_initial
# #         self.recorded_speeds = []
        
# #         # Dynamic bee size tracking (from BLOB areas, not boxes)
# #         self.recorded_bee_areas = []
# #         self.min_blob_area_dynamic = 50  # Start conservative, will adapt
        
# #         # Species mapping
# #         self.label_map = self.config.tracking.label_map
# #         self.tracking_classes = self.config.tracking.tracking_classes
        
# #         logger.info(f"✅ Hybrid tracker initialized (GPU: {self.use_gpu})")
# #         logger.info(f"🎯 Fully adaptive parameters ENABLED:")
# #         logger.info(f"  - min_blob_area: starts at {self.min_blob_area_dynamic}px², learns from blob contours")
# #         logger.info(f"  - search_region: elliptical, oriented by velocity direction")
# #         logger.info(f"  - cold-start: adaptive radius (100px→70px→50px→adaptive ellipse)")
# #         logger.info(f"  - distance thresholds: STRICTLY enforced (even for DL)")
# #         logger.info(f"  - AI noise filter: {'ENABLED' if self.noise_filter else 'DISABLED'}")
# #         logger.info(f"  - strict duplicate prevention: ENABLED")
    
# #     def _apply_config_scalling(self, config: Config):
# #         """Apply scale factor to config parameters (set to 1.0 to disable)."""
# #         self.scale_factor = 1.0
        
# #         # Aspect ratio filtering (dimensionless)
# #         self.min_aspect_ratio = self.config.tracking.min_blob_aspect_ratio
# #         self.max_aspect_ratio = self.config.tracking.max_blob_aspect_ratio
        
# #         # Area filtering (will be dynamic)
# #         base_min_area = getattr(self.config.tracking, 'min_blob_area_pixels', 200)
# #         base_max_area = getattr(self.config.tracking, 'max_blob_area_pixels', 5000)
# #         self.min_blob_area_pixels = base_min_area * (self.scale_factor ** 2)
# #         self.max_blob_area_pixels = base_max_area * (self.scale_factor ** 2)
        
# #         logger.info(f"Scale factor: {self.scale_factor:.2f}x")
# #         logger.info(f"Aspect ratio: {self.min_aspect_ratio:.2f}-{self.max_aspect_ratio:.2f}")
    
# #     def _detect_gpu(self) -> bool:
# #         """Detect if GPU is available."""
# #         try:
# #             import torch
# #             if torch.cuda.is_available():
# #                 logger.info(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
# #                 return True
# #         except ImportError:
# #             pass
        
# #         if cv2.cuda.getCudaEnabledDeviceCount() > 0:
# #             logger.info("OpenCV CUDA detected")
# #             return True
        
# #         logger.info("No GPU detected, using CPU")
# #         return False
    
# #     def _initialize_background_from_video(
# #         self,
# #         video_path: str,
# #         max_frames: int = 200,
# #         target_clean_frames: int = 50,
# #         config: Optional[Config] = None
# #     ) -> int:
# #         """Initialize BG model using bee-free frames."""
# #         if config is None:
# #             config = self.config
        
# #         cap = cv2.VideoCapture(video_path)
# #         if not cap.isOpened():
# #             raise ValueError(f"Cannot open video: {video_path}")
        
# #         logger.info("Initializing background model from bee-free frames...")
        
# #         frame_num = 0
# #         clean_frames_used = 0
        
# #         while frame_num < max_frames and clean_frames_used < target_clean_frames:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
            
# #             # Check for bees with YOLO
# #             results = self.model.predict(
# #                 frame,
# #                 conf=0.2,
# #                 iou=config.tracking.iou_threshold,
# #                 verbose=False,
# #                 device='0' if self.use_gpu else 'cpu'
# #             )
            
# #             has_bees = False
# #             if len(results) > 0 and results[0].boxes is not None:
# #                 for cls in results[0].boxes.cls:
# #                     if int(cls.cpu().numpy()) in self.tracking_classes:
# #                         has_bees = True
# #                         break
            
# #             if not has_bees:
# #                 self.bg_subtractor.apply(frame, learningRate=0.1)
# #                 clean_frames_used += 1
            
# #             frame_num += 1
        
# #         cap.release()
        
# #         logger.info(f"Background init: {clean_frames_used}/{frame_num} clean frames")
# #         logger.info(f"Background model is now FROZEN (learningRate=0)")
        
# #         return clean_frames_used
    
# #     # =========================================================================
# #     # MAIN TRACKING METHOD
# #     # =========================================================================
    
# #     def detect_and_track(
# #         self,
# #         video_path: str,
# #         site_roi: BBox,
# #         res_height: int,
# #         res_width: int,
# #         visualize: bool = False,
# #         output_folder: str = "output",
# #         config: Optional[Config] = None,
# #         initialize_background: bool = True
# #     ) -> pd.DataFrame:
# #         """Hybrid tracking with adaptive cold-start handling."""
# #         if config is None:
# #             config = self.config
        
# #         # Initialize BG model
# #         if initialize_background:
# #             self._initialize_background_from_video(video_path, config=config)
        
# #         # Setup
# #         if not os.path.exists(output_folder):
# #             os.makedirs(output_folder)
        
# #         cap = cv2.VideoCapture(video_path)
# #         if not cap.isOpened():
# #             raise ValueError(f"Cannot open video: {video_path}")
        
# #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #         fps = int(cap.get(cv2.CAP_PROP_FPS))
# #         logger.info(f"Processing {total_frames} frames at {fps} fps")
        
# #         # ROI mask
# #         x1_roi, y1_roi, x2_roi, y2_roi = [int(c) for c in site_roi]
# #         roi_mask = self._create_roi_mask(res_height, res_width, site_roi)
# #         logger.info(f"ROI: ({x1_roi}, {y1_roi}) to ({x2_roi}, {y2_roi})")
        
# #         # Visualization
# #         output_video = None
# #         if visualize:
# #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #             output_path = os.path.join(
# #                 output_folder,
# #                 f"tracking_{os.path.basename(video_path).rsplit('.', 1)[0]}.mp4"
# #             )
# #             output_video = cv2.VideoWriter(output_path, fourcc, fps, (res_width, res_height))
        
# #         need_resize = (res_width != config.video.res_width or 
# #                        res_height != config.video.res_height)
        
# #         # State
# #         frame_num = 0
# #         all_detections = []
        
# #         # Statistics
# #         yolo_calls = 0
# #         bg_only_updates = 0
# #         track_resurrections = 0
# #         new_tracks_created = 0
# #         new_tracks_blocked = 0
# #         dl_matches_rejected = 0
        
# #         # Main loop
# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
            
# #             if need_resize:
# #                 frame = cv2.resize(frame, (res_width, res_height))
            
# #             # FG/BG Segmentation
# #             fg_mask_full = self.bg_subtractor.apply(frame, learningRate=0)
# #             fg_mask_full[fg_mask_full == 127] = 0
# #             fg_mask = self._apply_roi_to_fgmask(fg_mask_full, roi_mask)
            
# #             # Extract blobs
# #             blobs = self._extract_blobs(frame, fg_mask, config, x_offset=0, y_offset=0)
            
# #             # Predict tracks
# #             predictions = self._predict_tracks(frame_num)
            
# #             # HYBRID LOGIC
# #             tracks_needing_dl = []
            
# #             for track_id, pred in predictions.items():
# #                 track = pred['track']
# #                 pred_centroid = pred['centroid']
                
# #                 # Get ADAPTIVE search region (larger for young tracks)
# #                 search_region = self._get_search_region(track)
# #                 nearby_blobs = self._count_blobs_in_search_region(blobs, pred_centroid, search_region)
                
# #                 if len(nearby_blobs) == 0:
# #                     tracks_needing_dl.append(track_id)
# #                 elif len(nearby_blobs) == 1:
# #                     self._update_track_with_blob(track, nearby_blobs[0], frame_num)
# #                     bg_only_updates += 1
# #                 elif len(nearby_blobs) <= 3:
# #                     closest = self._find_closest_blob(nearby_blobs, pred_centroid)
# #                     self._update_track_with_blob(track, closest, frame_num)
# #                     bg_only_updates += 1
# #                 else:
# #                     tracks_needing_dl.append(track_id)
            
# #             # Check for new insects
# #             unassociated_blobs = self._find_unassociated_blobs(blobs, predictions)
# #             if unassociated_blobs:
# #                 if not tracks_needing_dl or 'new' not in tracks_needing_dl:
# #                     tracks_needing_dl.append('new')
            
# #             # Run DL when needed
# #             yolo_detections = []
# #             if tracks_needing_dl:
# #                 yolo_calls += 1
# #                 yolo_detections = self._run_yolo(frame, config, x_offset=0, y_offset=0)
                
# #                 stats = self._update_tracks_with_yolo_strict(
# #                     yolo_detections,
# #                     tracks_needing_dl,
# #                     predictions,
# #                     frame_num
# #                 )
# #                 track_resurrections += stats['resurrections']
# #                 new_tracks_created += stats['new_tracks']
# #                 new_tracks_blocked += stats['blocked']
# #                 dl_matches_rejected += stats['rejected_matches']
            
# #             # Age out old tracks
# #             self._age_tracks(frame_num)
            
# #             # Record & visualize
# #             frame_debug_data = {'blobs': blobs, 'yolo_detections': yolo_detections}
# #             detections = self._get_current_detections(frame_num, frame_debug_data)
# #             all_detections.extend(detections)
            
# #             if visualize and output_video:
# #                 viz_frame = self._visualize_frame(frame, frame_num, False, blobs, yolo_detections)
# #                 output_video.write(viz_frame)
            
# #             frame_num += 1
# #             if frame_num % 100 == 0:
# #                 logger.info(f"Progress: {frame_num}/{total_frames} | Tracks: {len(self.tracks)} | "
# #                            f"DL: {yolo_calls} | Resurrected: {track_resurrections} | "
# #                            f"New blocked: {new_tracks_blocked}")
        
# #         # Cleanup
# #         cap.release()
# #         if output_video:
# #             output_video.release()
        
# #         # Statistics
# #         logger.info(f"=== TRACKING COMPLETE ===")
# #         logger.info(f"Total frames: {frame_num}")
# #         logger.info(f"DL calls: {yolo_calls} ({yolo_calls/frame_num*100:.1f}%)")
# #         logger.info(f"BG-only updates: {bg_only_updates}")
# #         logger.info(f"Track resurrections: {track_resurrections}")
# #         logger.info(f"New tracks created: {new_tracks_created}")
# #         logger.info(f"New tracks blocked: {new_tracks_blocked}")
# #         logger.info(f"DL matches rejected (outside region): {dl_matches_rejected}")
# #         logger.info(f"Total detections: {len(all_detections)}")
        
# #         logger.info(f"=== LEARNED PARAMETERS ===")
# #         if self.recorded_speeds:
# #             logger.info(f"Speed: min={min(self.recorded_speeds):.1f}, "
# #                        f"median={np.median(self.recorded_speeds):.1f}, "
# #                        f"max={self.d_max:.1f} px/frame")
# #         if self.recorded_bee_areas:
# #             logger.info(f"Bee area: min={min(self.recorded_bee_areas):.0f}, "
# #                        f"median={np.median(self.recorded_bee_areas):.0f}, "
# #                        f"max={max(self.recorded_bee_areas):.0f} px²")
# #             logger.info(f"Final min_blob_area: {self.min_blob_area_dynamic:.0f}px²")
        
# #         if len(all_detections) == 0:
# #             logger.error("❌ NO DETECTIONS COLLECTED!")
# #             return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
# #         return self._convert_to_grouped_format(all_detections)
    
# #     # =========================================================================
# #     # ADAPTIVE THRESHOLDS
# #     # =========================================================================
    
# #     def _update_adaptive_thresholds(self, track: TrackState):
# #         """Update adaptive thresholds based on observed movement."""
# #         if len(track.trajectory_history) < 2:
# #             return
        
# #         last_pos = track.trajectory_history[-1][1]
# #         prev_pos = track.trajectory_history[-2][1]
        
# #         dist = float(np.sqrt(
# #             (last_pos[0] - prev_pos[0])**2 +
# #             (last_pos[1] - prev_pos[1])**2
# #         ))
        
# #         self.d_max = max(self.d_max, dist)
# #         self.recorded_speeds.append(dist)
# #         if len(self.recorded_speeds) > 1000:
# #             self.recorded_speeds.pop(0)
    
# #     def _update_bee_size_statistics(self, blob_area: float):
# #         """Track actual bee blob sizes (contour areas)."""
# #         self.recorded_bee_areas.append(blob_area)
        
# #         if len(self.recorded_bee_areas) > 100:
# #             self.recorded_bee_areas.pop(0)
        
# #         if len(self.recorded_bee_areas) >= 10:
# #             self.min_blob_area_dynamic = np.percentile(self.recorded_bee_areas, 25) * 0.5
# #             self.min_blob_area_dynamic = max(50, min(500, self.min_blob_area_dynamic))
    
# #     def _get_bg_threshold(self) -> float:
# #         """Threshold for BG matches (MDTBS)."""
# #         return max(self.d_initial, self.d_max)
    
# #     def _get_dl_threshold(self, frames_without_detection: int) -> float:
# #         """Threshold for DL matches (MDTDL)."""
# #         base = self._get_bg_threshold()
# #         tau_star = 10
        
# #         if frames_without_detection <= tau_star:
# #             return 2 * base
        
# #         if len(self.recorded_speeds) > 10:
# #             eta_min = np.percentile(self.recorded_speeds, 25)
# #         else:
# #             eta_min = self.d_initial
        
# #         adaptive_term = min(
# #             eta_min * (frames_without_detection - tau_star) / 100,
# #             0.99 * base
# #         )
        
# #         return 2 * base + adaptive_term
    
# #     def _get_search_region(self, track: TrackState) -> Dict:
# #         """Get elliptical search region with ADAPTIVE COLD-START handling.
        
# #         Young tracks don't have reliable velocity estimates yet, so we use
# #         progressively smaller circular search regions until we have enough
# #         trajectory data to estimate velocity reliably.
        
# #         Search region progression:
# #         - 1 trajectory point: 100px circle (brand new, catch fast bees)
# #         - 2 trajectory points: 70px circle (one velocity sample, still uncertain)
# #         - 3-4 trajectory points: 50px circle (building confidence)
# #         - 5+ trajectory points: Adaptive ellipse (confident velocity estimate)
# #         """
# #         trajectory_points = len(track.trajectory_history)
        
# #         # ✅ ADAPTIVE COLD START based on trajectory confidence
# #         if trajectory_points == 1:
# #             # Brand new track - use VERY large radius to catch fast-moving bees
# #             radius = 100.0
# #             logger.debug(f"Track {track.track_id}: NEW (1 pt), radius={radius:.0f}px")
# #             return {
# #                 'type': 'circle',
# #                 'radius': radius,
# #                 'major_axis': radius,
# #                 'minor_axis': radius,
# #                 'angle': 0.0
# #             }
# #         elif trajectory_points == 2:
# #             # One velocity measurement - still very uncertain
# #             radius = 70.0
# #             logger.debug(f"Track {track.track_id}: YOUNG (2 pts), radius={radius:.0f}px")
# #             return {
# #                 'type': 'circle',
# #                 'radius': radius,
# #                 'major_axis': radius,
# #                 'minor_axis': radius,
# #                 'angle': 0.0
# #             }
# #         elif trajectory_points < 5:
# #             # Building confidence - moderate radius
# #             radius = 50.0
# #             logger.debug(f"Track {track.track_id}: MATURING ({trajectory_points} pts), radius={radius:.0f}px")
# #             return {
# #                 'type': 'circle',
# #                 'radius': radius,
# #                 'major_axis': radius,
# #                 'minor_axis': radius,
# #                 'angle': 0.0
# #             }
        
# #         # ✅ MATURE TRACK: Use velocity-based adaptive ellipse
# #         vx = float(track.kalman.statePost[2])
# #         vy = float(track.kalman.statePost[3])
# #         speed = float(np.sqrt(vx**2 + vy**2))
        
# #         BASE_RADIUS = 20.0
# #         SPEED_SCALE = 2.5
        
# #         # Stationary: small circle
# #         if speed < 2.0:
# #             radius = BASE_RADIUS
# #             return {
# #                 'type': 'circle',
# #                 'radius': radius,
# #                 'major_axis': radius,
# #                 'minor_axis': radius,
# #                 'angle': 0.0
# #             }
        
# #         # Moving: ellipse oriented by velocity
# #         major_axis = BASE_RADIUS + (speed * SPEED_SCALE)
# #         major_axis = max(BASE_RADIUS, min(major_axis, 150.0))
        
# #         # Elongation: faster = more stretched
# #         elongation_ratio = min(3.0, 1.0 + (speed / 10.0))
# #         minor_axis = major_axis / elongation_ratio
# #         minor_axis = max(BASE_RADIUS, minor_axis)
        
# #         angle = float(np.arctan2(vy, vx))
        
# #         return {
# #             'type': 'ellipse',
# #             'major_axis': major_axis,
# #             'minor_axis': minor_axis,
# #             'angle': angle,
# #             'radius': major_axis
# #         }
    
# #     def _get_expanded_search_region_for_resurrection(self, track: TrackState) -> Dict:
# #         """Get EXPANDED search region for resurrection attempts.
        
# #         Uses track's CURRENT position with larger, more permissive region.
# #         This allows for temporary tracking failures but still bounds where we look.
# #         """
# #         vx = float(track.kalman.statePost[2])
# #         vy = float(track.kalman.statePost[3])
# #         speed = float(np.sqrt(vx**2 + vy**2))
        
# #         # For resurrection: use larger base radius
# #         BASE_RADIUS = 40.0  # Doubled from 20
# #         SPEED_SCALE = 3.5   # More generous than 2.5
        
# #         # Stationary: larger circle for resurrection
# #         if speed < 2.0:
# #             radius = BASE_RADIUS * 1.5  # 60px
# #             return {
# #                 'type': 'circle',
# #                 'radius': radius,
# #                 'major_axis': radius,
# #                 'minor_axis': radius,
# #                 'angle': 0.0
# #             }
        
# #         # Moving: expanded ellipse
# #         major_axis = BASE_RADIUS + (speed * SPEED_SCALE)
# #         major_axis = max(BASE_RADIUS, min(major_axis, 200.0))  # Increased max from 150
        
# #         # Less elongation for resurrection (more forgiving)
# #         elongation_ratio = min(2.5, 1.0 + (speed / 15.0))  # Less aggressive than 3.0
# #         minor_axis = major_axis / elongation_ratio
# #         minor_axis = max(BASE_RADIUS, minor_axis)
        
# #         angle = float(np.arctan2(vy, vx))
        
# #         return {
# #             'type': 'ellipse',
# #             'major_axis': major_axis,
# #             'minor_axis': minor_axis,
# #             'angle': angle,
# #             'radius': major_axis
# #         }
    
# #     def _is_point_in_search_region(
# #         self,
# #         point: Point,
# #         center: Point,
# #         search_region: Dict
# #     ) -> bool:
# #         """Check if point is inside search region (circle or ellipse)."""
# #         px, py = point
# #         cx, cy = center
        
# #         dx = px - cx
# #         dy = py - cy
        
# #         if search_region['type'] == 'circle':
# #             dist = float(np.sqrt(dx**2 + dy**2))
# #             return dist <= search_region['radius']
        
# #         # Ellipse: rotate to align with axes
# #         angle = search_region['angle']
# #         cos_a = np.cos(-angle)
# #         sin_a = np.sin(-angle)
        
# #         rx = dx * cos_a - dy * sin_a
# #         ry = dx * sin_a + dy * cos_a
        
# #         a = search_region['major_axis']
# #         b = search_region['minor_axis']
        
# #         ellipse_eq = (rx / a)**2 + (ry / b)**2
        
# #         return ellipse_eq <= 1.0
    
# #     # =========================================================================
# #     # BLOB PROCESSING
# #     # =========================================================================
    
# #     def _extract_blobs(
# #         self,
# #         frame: np.ndarray,
# #         fg_mask: np.ndarray,
# #         config: Config,
# #         x_offset: int,
# #         y_offset: int
# #     ) -> List[Dict]:
# #         """Extract and filter blobs with AI noise filter."""
        
# #         kernel = np.ones((5, 5), np.uint8)
# #         opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
# #         cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
# #         contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
# #         blobs = []
# #         min_area = self.min_blob_area_dynamic
        
# #         rejected_area = 0
# #         rejected_solidity = 0
# #         rejected_shape = 0
# #         rejected_noise = 0
        
# #         for contour in contours:
# #             area = cv2.contourArea(contour)
            
# #             if area < min_area:
# #                 rejected_area += 1
# #                 continue
            
# #             x, y, w, h = cv2.boundingRect(contour)
# #             bbox = (x + x_offset, y + y_offset, x + w + x_offset, y + h + y_offset)
            
# #             if not self._is_valid_blob_shape(bbox):
# #                 rejected_shape += 1
# #                 continue
            
# #             hull = cv2.convexHull(contour)
# #             hull_area = cv2.contourArea(hull)
# #             solidity = area / hull_area if hull_area > 0 else 0
            
# #             if solidity < 0.5:
# #                 rejected_solidity += 1
# #                 continue
            
# #             M = cv2.moments(contour)
# #             if M["m00"] > 0:
# #                 cx = M["m10"] / M["m00"]
# #                 cy = M["m01"] / M["m00"]
# #             else:
# #                 cx = x + w / 2
# #                 cy = y + h / 2
            
# #             centroid = (cx + x_offset, cy + y_offset)
            
# #             blobs.append({
# #                 'bbox': bbox,
# #                 'centroid': centroid,
# #                 'area': area,
# #                 'solidity': solidity,
# #                 'contour': contour
# #             })
        
# #         # AI noise filter
# #         if self.noise_filter and blobs:
# #             filter_blobs = []
# #             for blob in blobs:
# #                 x1, y1, x2, y2 = blob['bbox']
# #                 x = x1 - x_offset
# #                 y = y1 - y_offset
# #                 w = x2 - x1
# #                 h = y2 - y1
# #                 filter_blobs.append((x, y, w, h))
            
# #             roi_height, roi_width = fg_mask.shape
# #             frame_roi = frame[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width]
            
# #             filtered_blobs = self.noise_filter.filter_blobs(frame_roi, filter_blobs)
            
# #             kept_bboxes = set()
# #             for x, y, w, h, prob in filtered_blobs:
# #                 bbox_global = (x + x_offset, y + y_offset, 
# #                               x + w + x_offset, y + h + y_offset)
# #                 kept_bboxes.add(bbox_global)
            
# #             original_count = len(blobs)
# #             blobs = [b for b in blobs if b['bbox'] in kept_bboxes]
# #             rejected_noise = original_count - len(blobs)
        
# #         total = len(contours)
# #         if total > 0:
# #             logger.debug(f"Blobs: {len(blobs)}/{total} (rejected: area={rejected_area}, "
# #                         f"shape={rejected_shape}, solid={rejected_solidity}, noise={rejected_noise})")
        
# #         return blobs
    
# #     def _is_valid_blob_shape(self, bbox: BBox) -> bool:
# #         """Check aspect ratio and size."""
# #         x1, y1, x2, y2 = bbox
# #         width = x2 - x1
# #         height = y2 - y1
        
# #         if height == 0 or width == 0:
# #             return False
        
# #         aspect_ratio = width / height
# #         if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
# #             return False
        
# #         area = width * height
# #         if not (self.min_blob_area_pixels <= area <= self.max_blob_area_pixels):
# #             return False
        
# #         return True
    
# #     def _count_blobs_in_search_region(
# #         self,
# #         blobs: List[Dict],
# #         centroid: Point,
# #         search_region: Dict
# #     ) -> List[Dict]:
# #         """Find blobs within elliptical search region."""
# #         nearby = []
        
# #         for blob in blobs:
# #             if self._is_point_in_search_region(blob['centroid'], centroid, search_region):
# #                 nearby.append(blob)
        
# #         return nearby
    
# #     def _find_closest_blob(self, blobs: List[Dict], centroid: Point) -> Dict:
# #         """Find closest blob to centroid."""
# #         cx, cy = centroid
# #         min_dist = float('inf')
# #         closest = None
        
# #         for blob in blobs:
# #             bx, by = blob['centroid']
# #             dist = float(np.sqrt((bx - cx)**2 + (by - cy)**2))
# #             if dist < min_dist:
# #                 min_dist = dist
# #                 closest = blob
        
# #         return closest
    
# #     def _find_unassociated_blobs(
# #         self,
# #         blobs: List[Dict],
# #         predictions: Dict[int, Dict]
# #     ) -> List[Dict]:
# #         """Find blobs far from all tracks."""
# #         unassociated = []
        
# #         for blob in blobs:
# #             is_near_track = False
            
# #             for pred in predictions.values():
# #                 track = pred['track']
# #                 search_region = self._get_search_region(track)
                
# #                 if self._is_point_in_search_region(blob['centroid'], pred['centroid'], search_region):
# #                     is_near_track = True
# #                     break
            
# #             if not is_near_track:
# #                 unassociated.append(blob)
        
# #         return unassociated
    
# #     # =========================================================================
# #     # PREDICTION & TRACKING
# #     # =========================================================================
    
# #     def _predict_tracks(self, frame_num: int) -> Dict[int, Dict]:
# #         """Predict track positions with velocity damping."""
# #         predictions = {}
        
# #         for track_id, track in self.tracks.items():
# #             self._update_adaptive_thresholds(track)
            
# #             frames_since_dl = frame_num - track.last_yolo_confirmation
# #             if frames_since_dl > 0:
# #                 damping = max(0.2, 1.0 - (frames_since_dl * 0.1))
# #                 track.kalman.statePost[2] *= damping
# #                 track.kalman.statePost[3] *= damping
            
# #             prediction = track.kalman.predict()
# #             pred_x, pred_y = float(prediction[0]), float(prediction[1])
            
# #             x1, y1, x2, y2 = track.bbox
# #             width = x2 - x1
# #             height = y2 - y1
            
# #             pred_bbox = (
# #                 pred_x - width / 2,
# #                 pred_y - height / 2,
# #                 pred_x + width / 2,
# #                 pred_y + height / 2
# #             )
            
# #             predictions[track_id] = {
# #                 'centroid': (pred_x, pred_y),
# #                 'bbox': pred_bbox,
# #                 'track': track
# #             }
        
# #         return predictions
    
# #     def _update_track_with_blob(
# #         self,
# #         track: TrackState,
# #         blob: Dict,
# #         frame_num: int
# #     ):
# #         """Update track using blob (BG result)."""
# #         track.bbox = blob['bbox']
# #         track.centroid = blob['centroid']
# #         track.frames_without_detection = 0
# #         track.age += 1
        
# #         track.trajectory_history.append((frame_num, blob['centroid']))
# #         if len(track.trajectory_history) > 30:
# #             track.trajectory_history.pop(0)
        
# #         measurement = np.array([[blob['centroid'][0]],
# #                                 [blob['centroid'][1]]], dtype=np.float32)
# #         track.kalman.correct(measurement)
        
# #         # Learn from blob area
# #         self._update_bee_size_statistics(blob['area'])
    
# #     def _update_tracks_with_yolo_strict(
# #         self,
# #         yolo_detections: List[Dict],
# #         tracks_needing_dl: List,
# #         predictions: Dict[int, Dict],
# #         frame_num: int
# #     ) -> Dict:
# #         """Update tracks with STRICT search region enforcement."""
# #         stats = {
# #             'resurrections': 0,
# #             'new_tracks': 0,
# #             'blocked': 0,
# #             'rejected_matches': 0
# #         }
        
# #         if not yolo_detections:
# #             for track_id in tracks_needing_dl:
# #                 if track_id != 'new' and track_id in self.tracks:
# #                     self.tracks[track_id].frames_without_detection += 1
# #             return stats
        
# #         used_detections = set()
        
# #         # ================================================================
# #         # STEP 1: Update existing tracks - STRICT search region enforcement
# #         # ================================================================
# #         for track_id in tracks_needing_dl:
# #             if track_id == 'new':
# #                 continue
            
# #             if track_id not in predictions:
# #                 continue
            
# #             pred = predictions[track_id]
# #             track = pred['track']
            
# #             # Get search region for this track
# #             search_region = self._get_search_region(track)
# #             threshold = self._get_dl_threshold(track.frames_without_detection)
            
# #             # STRICT: Only consider YOLO detections WITHIN search region
# #             candidate_detections = []
# #             for idx, det in enumerate(yolo_detections):
# #                 if idx in used_detections:
# #                     continue
                
# #                 # Must be in search region
# #                 if not self._is_point_in_search_region(det['centroid'], pred['centroid'], search_region):
# #                     continue
                
# #                 dist = float(np.sqrt(
# #                     (float(det['centroid'][0]) - float(pred['centroid'][0]))**2 +
# #                     (float(det['centroid'][1]) - float(pred['centroid'][1]))**2
# #                 ))
                
# #                 # Must be within adaptive threshold
# #                 if dist < threshold:
# #                     candidate_detections.append((idx, det, dist))
            
# #             if candidate_detections:
# #                 # Take closest valid candidate
# #                 candidate_detections.sort(key=lambda x: x[2])
# #                 best_idx, best_det, best_dist = candidate_detections[0]
                
# #                 # Update track
# #                 track.bbox = best_det['bbox']
# #                 track.centroid = best_det['centroid']
# #                 track.label = best_det['label']
# #                 track.frames_without_detection = 0
# #                 track.last_yolo_confirmation = frame_num
                
# #                 track.trajectory_history.append((frame_num, best_det['centroid']))
# #                 if len(track.trajectory_history) > 30:
# #                     track.trajectory_history.pop(0)
                
# #                 measurement = np.array([[best_det['centroid'][0]],
# #                                         [best_det['centroid'][1]]], dtype=np.float32)
# #                 track.kalman.correct(measurement)
                
# #                 used_detections.add(best_idx)
# #                 logger.debug(f"✅ Track {track_id} updated (dist={best_dist:.1f}, thresh={threshold:.1f})")
# #             else:
# #                 track.frames_without_detection += 1
# #                 stats['rejected_matches'] += 1
# #                 logger.debug(f"❌ Track {track_id} no valid DL match (outside region or too far)")
        
# #         # ================================================================
# #         # STEP 2: Resurrection - WITH EXPANDED SEARCH REGION CHECK
# #         # ================================================================
# #         for idx, det in enumerate(yolo_detections):
# #             if idx in used_detections:
# #                 continue
            
# #             # Try resurrection with ALL tracks
# #             best_resurrection = None
# #             best_dist = float('inf')
# #             best_track_id = None
            
# #             for track_id, track in self.tracks.items():
# #                 # Create EXPANDED search region for resurrection
# #                 expanded_search_region = self._get_expanded_search_region_for_resurrection(track)
                
# #                 # First check: Must be within expanded search region
# #                 if not self._is_point_in_search_region(det['centroid'], track.centroid, expanded_search_region):
# #                     continue
                
# #                 # Second check: Distance threshold (as safety)
# #                 track_threshold = self._get_dl_threshold(track.frames_without_detection) * 1.5
                
# #                 dist = float(np.sqrt(
# #                     (track.centroid[0] - det['centroid'][0])**2 +
# #                     (track.centroid[1] - det['centroid'][1])**2
# #                 ))
                
# #                 if dist < track_threshold and dist < best_dist:
# #                     best_dist = dist
# #                     best_track_id = track_id
# #                     best_resurrection = track
            
# #             if best_resurrection:
# #                 # Resurrect this track
# #                 logger.info(f"♻️ Resurrecting track {best_track_id} (dist={best_dist:.1f}px)")
# #                 best_resurrection.bbox = det['bbox']
# #                 best_resurrection.centroid = det['centroid']
# #                 best_resurrection.label = det['label']
# #                 best_resurrection.frames_without_detection = 0
# #                 best_resurrection.last_yolo_confirmation = frame_num
                
# #                 best_resurrection.trajectory_history.append((frame_num, det['centroid']))
# #                 if len(best_resurrection.trajectory_history) > 30:
# #                     best_resurrection.trajectory_history.pop(0)
                
# #                 measurement = np.array([[det['centroid'][0]],
# #                                         [det['centroid'][1]]], dtype=np.float32)
# #                 best_resurrection.kalman.correct(measurement)
                
# #                 used_detections.add(idx)
# #                 stats['resurrections'] += 1
# #                 continue
            
# #             # ================================================================
# #             # STEP 3: Create new track - ULTRA STRICT
# #             # ================================================================
# #             can_create = True
# #             min_distance = float('inf')
            
# #             for track_id, track in self.tracks.items():
# #                 # Check expanded search region
# #                 expanded_region = self._get_expanded_search_region_for_resurrection(track)
                
# #                 if self._is_point_in_search_region(det['centroid'], track.centroid, expanded_region):
# #                     can_create = False
# #                     logger.debug(f"🚫 New track blocked - within track {track_id}'s expanded region")
# #                     stats['blocked'] += 1
# #                     break
                
# #                 # Also check distance as safety
# #                 dist = float(np.sqrt(
# #                     (track.centroid[0] - det['centroid'][0])**2 +
# #                     (track.centroid[1] - det['centroid'][1])**2
# #                 ))
                
# #                 min_distance = min(min_distance, dist)
                
# #                 # STRICT: Must be at least 2x DL threshold away
# #                 safety_threshold = self._get_dl_threshold(track.frames_without_detection) * 2.0
                
# #                 if dist < safety_threshold:
# #                     can_create = False
# #                     logger.debug(f"🚫 New track blocked - track {track_id} only {dist:.1f}px away "
# #                                f"(safety={safety_threshold:.1f})")
# #                     stats['blocked'] += 1
# #                     break
            
# #             if can_create:
# #                 logger.info(f"✨ Creating new track (nearest={min_distance:.1f}px)")
# #                 self._create_track(det, frame_num)
# #                 stats['new_tracks'] += 1
        
# #         return stats
    
# #     def _create_track(self, detection: Dict, frame_num: int):
# #         """Create new track from detection."""
# #         kalman = cv2.KalmanFilter(4, 2)
# #         kalman.measurementMatrix = np.array([[1, 0, 0, 0],
# #                                               [0, 1, 0, 0]], dtype=np.float32)
# #         kalman.transitionMatrix = np.array([[1, 0, 1, 0],
# #                                              [0, 1, 0, 1],
# #                                              [0, 0, 1, 0],
# #                                              [0, 0, 0, 1]], dtype=np.float32)
# #         kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
# #         kalman.statePre = np.array([[detection['centroid'][0]],
# #                                      [detection['centroid'][1]],
# #                                      [0],
# #                                      [0]], dtype=np.float32)
        
# #         track = TrackState(
# #             track_id=self.next_track_id,
# #             bbox=detection['bbox'],
# #             centroid=detection['centroid'],
# #             kalman=kalman,
# #             frames_without_detection=0,
# #             label=detection.get('label', 'unknown'),
# #             age=1,
# #             last_yolo_confirmation=frame_num
# #         )
        
# #         # Initialize trajectory with first point
# #         track.trajectory_history.append((frame_num, detection['centroid']))
        
# #         self.tracks[self.next_track_id] = track
# #         self.next_track_id += 1
    
# #     def _age_tracks(self, frame_num: int):
# #         """Remove old tracks."""
# #         max_age = self.config.tracking.max_age
# #         to_remove = []
        
# #         for track_id, track in self.tracks.items():
# #             if track.frames_without_detection > max_age:
# #                 to_remove.append(track_id)
        
# #         for track_id in to_remove:
# #             logger.debug(f"⏰ Removing track {track_id} (age: {self.tracks[track_id].frames_without_detection})")
# #             del self.tracks[track_id]
    
# #     # =========================================================================
# #     # IoU OVERLAP PREVENTION
# #     # =========================================================================
    
# #     def _has_box_overlap(self, bbox: BBox, overlap_threshold: float = 0.3) -> bool:
# #         """Check if bbox overlaps any existing track."""
# #         for track in self.tracks.values():
# #             iou = self._compute_iou(bbox, track.bbox)
# #             if iou > overlap_threshold:
# #                 return True
# #         return False
    
# #     def _find_overlapping_track(self, bbox: BBox) -> Optional[int]:
# #         """Find track with highest overlap."""
# #         best_iou = 0
# #         best_id = None
        
# #         for track_id, track in self.tracks.items():
# #             iou = self._compute_iou(bbox, track.bbox)
# #             if iou > best_iou:
# #                 best_iou = iou
# #                 best_id = track_id
        
# #         return best_id if best_iou > 0.3 else None
    
# #     def _compute_iou(self, bbox1: BBox, bbox2: BBox) -> float:
# #         """Compute IoU of two boxes."""
# #         x1_1, y1_1, x2_1, y2_1 = bbox1
# #         x1_2, y1_2, x2_2, y2_2 = bbox2
        
# #         x1_i = max(x1_1, x1_2)
# #         y1_i = max(y1_1, y1_2)
# #         x2_i = min(x2_1, x2_2)
# #         y2_i = min(y2_1, y2_2)
        
# #         if x2_i < x1_i or y2_i < y1_i:
# #             return 0.0
        
# #         intersection = (x2_i - x1_i) * (y2_i - y1_i)
# #         area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
# #         area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
# #         union = area1 + area2 - intersection
        
# #         return intersection / union if union > 0 else 0.0
    
# #     # =========================================================================
# #     # HELPER METHODS
# #     # =========================================================================
    
# #     def _create_roi_mask(self, h: int, w: int, roi: BBox) -> np.ndarray:
# #         """Create ROI mask."""
# #         x1, y1, x2, y2 = map(int, roi)
# #         mask = np.zeros((h, w), dtype=np.uint8)
# #         if x2 > x1 and y2 > y1:
# #             mask[y1:y2, x1:x2] = 255
# #         return mask
    
# #     def _apply_roi_to_fgmask(self, fg_mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
# #         """Apply ROI mask to FG mask."""
# #         return cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)
    
# #     def _run_yolo(
# #         self,
# #         frame: np.ndarray,
# #         config: Config,
# #         x_offset: int,
# #         y_offset: int
# #     ) -> List[Dict]:
# #         """Run YOLO detection."""
# #         results = self.model.predict(
# #             frame,
# #             conf=config.tracking.confidence_threshold,
# #             iou=config.tracking.iou_threshold,
# #             verbose=False,
# #             device='0' if self.use_gpu else 'cpu'
# #         )
        
# #         detections = []
        
# #         if len(results) > 0 and results[0].boxes is not None:
# #             for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
# #                 class_id = int(cls.cpu().numpy())
                
# #                 if class_id not in self.tracking_classes:
# #                     continue
                
# #                 x1, y1, x2, y2 = box.cpu().numpy()
# #                 cx = (x1 + x2) / 2
# #                 cy = (y1 + y2) / 2
                
# #                 detections.append({
# #                     'bbox': (x1 + x_offset, y1 + y_offset,
# #                             x2 + x_offset, y2 + y_offset),
# #                     'centroid': (cx + x_offset, cy + y_offset),
# #                     'class_id': class_id,
# #                     'label': self.label_map.get(class_id, f'class_{class_id}')
# #                 })
        
# #         return detections
    
# #     def _get_current_detections(
# #         self,
# #         frame_num: int,
# #         debug_data: Optional[Dict] = None
# #     ) -> List[Dict]:
# #         """Get detections from active tracks."""
# #         detections = []
        
# #         for track_id, track in self.tracks.items():
# #             det = {
# #                 'frame': frame_num,
# #                 'track_id': track_id,
# #                 'x1': track.bbox[0],
# #                 'y1': track.bbox[1],
# #                 'x2': track.bbox[2],
# #                 'y2': track.bbox[3],
# #                 'species': track.label
# #             }
            
# #             if debug_data:
# #                 det['debug_blobs'] = debug_data.get('blobs', [])
# #                 det['debug_yolo'] = debug_data.get('yolo_detections', [])
            
# #             detections.append(det)
        
# #         return detections
    
# #     def _visualize_frame(
# #         self,
# #         frame: np.ndarray,
# #         frame_num: int,
# #         is_low_res: bool,
# #         blobs: List[Dict] = None,
# #         yolo_detections: List[Dict] = None
# #     ) -> np.ndarray:
# #         """Visualize with adaptive search regions."""
# #         viz = frame.copy()
        
# #         # Draw blobs (cyan)
# #         if blobs:
# #             for blob in blobs:
# #                 x1, y1, x2, y2 = [int(c) for c in blob['bbox']]
# #                 cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 255, 0), 1)
# #                 cx, cy = [int(c) for c in blob['centroid']]
# #                 cv2.circle(viz, (cx, cy), 2, (255, 255, 0), -1)
        
# #         # Draw YOLO (orange)
# #         if yolo_detections:
# #             for det in yolo_detections:
# #                 x1, y1, x2, y2 = [int(c) for c in det['bbox']]
# #                 cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 165, 255), 2)
# #                 cv2.putText(viz, f"DL:{det['label']}", (x1, y1-5),
# #                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
# #         # Draw tracks with adaptive search regions
# #         for track_id, track in self.tracks.items():
# #             search_region = self._get_search_region(track)
            
# #             cx, cy = [int(c) for c in track.centroid]
            
# #             # Draw search region
# #             if search_region['type'] == 'circle':
# #                 cv2.circle(viz, (cx, cy), int(search_region['radius']), (128, 128, 128), 1)
# #             else:
# #                 axes = (int(search_region['major_axis']), int(search_region['minor_axis']))
# #                 angle_deg = int(np.degrees(search_region['angle']))
# #                 cv2.ellipse(viz, (cx, cy), axes, angle_deg, 0, 360, (128, 128, 128), 1)
                
# #                 # Velocity arrow
# #                 arrow_len = 30
# #                 end_x = int(cx + arrow_len * np.cos(search_region['angle']))
# #                 end_y = int(cy + arrow_len * np.sin(search_region['angle']))
# #                 cv2.arrowedLine(viz, (cx, cy), (end_x, end_y), (255, 128, 0), 2, tipLength=0.3)
            
# #             # Trajectory
# #             if len(track.trajectory_history) > 1:
# #                 for i in range(len(track.trajectory_history) - 1):
# #                     pt1 = tuple(map(int, track.trajectory_history[i][1]))
# #                     pt2 = tuple(map(int, track.trajectory_history[i+1][1]))
# #                     cv2.line(viz, pt1, pt2, (255, 255, 0), 1)
            
# #             # Box color by freshness
# #             frames_since_dl = frame_num - track.last_yolo_confirmation
# #             if frames_since_dl == 0:
# #                 color = (0, 255, 0)  # Green
# #             elif frames_since_dl < 10:
# #                 color = (0, 255, 255)  # Yellow
# #             else:
# #                 color = (255, 0, 0)  # Blue
            
# #             x1, y1, x2, y2 = [int(c) for c in track.bbox]
# #             cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            
# #             # Label with trajectory count
# #             traj_pts = len(track.trajectory_history)
# #             vx = float(track.kalman.statePost[2])
# #             vy = float(track.kalman.statePost[3])
# #             speed = float(np.sqrt(vx**2 + vy**2))
            
# #             if search_region['type'] == 'circle':
# #                 label = f"ID:{track_id} {track.label} (pts={traj_pts}, r={search_region['radius']:.0f})"
# #             else:
# #                 label = f"ID:{track_id} {track.label} (v={speed:.1f}, {search_region['major_axis']:.0f}x{search_region['minor_axis']:.0f})"
# #             cv2.putText(viz, label, (x1, y1-5),
# #                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
# #         # Legend
# #         y = 120
# #         cv2.putText(viz, "Legend:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
# #         cv2.putText(viz, "Green=DL confirmed", (10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
# #         cv2.putText(viz, "Yellow=Recent", (10, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
# #         cv2.putText(viz, "Blue=BG only", (10, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
# #         cv2.putText(viz, "Cyan=Blobs", (10, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
# #         cv2.putText(viz, "Orange=DL", (10, y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)
# #         cv2.putText(viz, "Gray=Search region", (10, y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128), 1)
# #         cv2.putText(viz, "Orange arrow=Direction", (10, y+140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,128,0), 1)
        
# #         cv2.putText(viz, f"min_area: {self.min_blob_area_dynamic:.0f}px2", 
# #                    (10, y+170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
# #         return viz
    
# #     def _convert_to_grouped_format(self, all_detections: List[Dict]) -> pd.DataFrame:
# #         """Convert flat detections to grouped format."""
# #         if not all_detections:
# #             return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
# #         debug_data_by_frame = {}
# #         for det in all_detections:
# #             frame_num = det['frame']
# #             if 'debug_blobs' in det and frame_num not in debug_data_by_frame:
# #                 debug_data_by_frame[frame_num] = {
# #                     'blobs': det.get('debug_blobs', []),
# #                     'yolo': det.get('debug_yolo', [])
# #                 }
        
# #         clean_detections = []
# #         for det in all_detections:
# #             clean = {k: v for k, v in det.items() if k not in ['debug_blobs', 'debug_yolo']}
# #             clean_detections.append(clean)
        
# #         df = pd.DataFrame(clean_detections)
# #         periods = self._split_into_periods(df, gap_threshold=int(self.config.tracking.max_age * 1.1))
        
# #         result_rows = []
# #         for period_df in periods:
# #             track_groups = {}
            
# #             for track_id in period_df['track_id'].unique():
# #                 track_df = period_df[period_df['track_id'] == track_id].sort_values('frame')
# #                 segments = self._split_track_by_gaps(track_df, gap_threshold=self.config.tracking.max_age)
                
# #                 for seg_idx, seg_df in enumerate(segments):
# #                     unique_id = f"{track_id}_{seg_idx}" if len(segments) > 1 else track_id
                    
# #                     centroids = [((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
# #                                 for _, row in seg_df.iterrows()]
# #                     bboxes = [(row['x1'], row['y1'], row['x2'], row['y2'])
# #                              for _, row in seg_df.iterrows()]
# #                     frame_numbers = seg_df['frame'].tolist()
                    
# #                     if len(frame_numbers) >= self.config.tracking.min_track_length:
# #                         track_groups[unique_id] = (unique_id, centroids, bboxes, frame_numbers)
            
# #             if not track_groups:
# #                 continue
            
# #             all_tracks = list(track_groups.values())
# #             min_frame = period_df['frame'].min()
# #             max_frame = period_df['frame'].max()
            
# #             frame_detections = {}
# #             for frame_num in period_df['frame'].unique():
# #                 frame_df = period_df[period_df['frame'] == frame_num]
# #                 frame_debug = debug_data_by_frame.get(int(frame_num), {'blobs': [], 'yolo': []})
                
# #                 frame_detections[int(frame_num)] = {
# #                     'boxes': [(row['x1'], row['y1'], row['x2'], row['y2'])
# #                              for _, row in frame_df.iterrows()],
# #                     'label': frame_df['species'].tolist(),
# #                     'debug_blobs': frame_debug['blobs'],
# #                     'debug_yolo': frame_debug['yolo']
# #                 }
            
# #             result_rows.append({
# #                 'frame_number': (int(min_frame), int(max_frame)),
# #                 'tracks': all_tracks,
# #                 'detections': frame_detections
# #             })
        
# #         return pd.DataFrame(result_rows) if result_rows else pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
    
# #     def _split_into_periods(self, df: pd.DataFrame, gap_threshold: int = 100) -> List[pd.DataFrame]:
# #         """Split detections into activity periods."""
# #         df = df.sort_values('frame')
# #         frames = df['frame'].tolist()
        
# #         periods = []
# #         current_start = 0
        
# #         for i in range(len(frames) - 1):
# #             gap = frames[i + 1] - frames[i]
# #             if gap > gap_threshold:
# #                 periods.append(df.iloc[current_start:i+1].copy())
# #                 current_start = i + 1
        
# #         if current_start < len(df):
# #             periods.append(df.iloc[current_start:].copy())
        
# #         return periods
    
# #     def _split_track_by_gaps(self, track_df: pd.DataFrame, gap_threshold: int = 30) -> List[pd.DataFrame]:
# #         """Split track into segments by gaps."""
# #         frames = track_df['frame'].tolist()
        
# #         segments = []
# #         current_start = 0
        
# #         for i in range(len(frames) - 1):
# #             gap = frames[i + 1] - frames[i]
# #             if gap > gap_threshold:
# #                 segments.append(track_df.iloc[current_start:i+1].copy())
# #                 current_start = i + 1
        
# #         if current_start < len(track_df):
# #             segments.append(track_df.iloc[current_start:].copy())
        
# #         return segments























# """Hybrid motion detection and tracking system with adaptive cold-start handling.

# Combines background subtraction (efficient) with deep learning (accurate)
# for multi-insect tracking in bee hotel videos.

# Key features:
# - Hybrid BG/DL logic: BG for simple cases, DL for complex
# - Elliptical search regions oriented by velocity direction
# - ADAPTIVE cold-start: Large radius for new tracks, narrows as confidence grows
# - FIXED resurrection: Even larger radius for young tracks
# - STRICT distance enforcement for DL detections (within search region only)
# - Fully adaptive thresholds learned from actual bee behavior
# - Dynamic blob filtering based on actual blob contour areas
# - AI noise filter (optional fourth filter)
# - Aggressive duplicate prevention and track resurrection
# - Solidity filter for shape compactness
# - IoU-based overlap prevention
# - Kalman filtering with velocity damping
# - ROI masking for efficient motion detection
# """

# import logging
# from typing import Dict, List, Tuple, Optional
# import cv2
# import numpy as np
# import pandas as pd
# import os
# from dataclasses import dataclass

# from beemonitor.core.config import Config

# logger = logging.getLogger(__name__)

# # Type aliases
# BBox = Tuple[float, float, float, float]
# Point = Tuple[float, float]


# @dataclass
# class TrackState:
#     """State for a tracked insect."""
#     track_id: int
#     bbox: BBox
#     centroid: Point
#     kalman: cv2.KalmanFilter
#     frames_without_detection: int
#     label: str
#     age: int
#     last_yolo_confirmation: int
#     trajectory_history: list = None
    
#     def __post_init__(self):
#         if self.trajectory_history is None:
#             self.trajectory_history = []


# class MotionTracking:
#     """Standalone hybrid tracker with adaptive cold-start handling.
    
#     Learns from actual bee behavior instead of using hard-coded values:
#     - Blob size threshold from actual blob contour areas
#     - Distance thresholds from observed movement
#     - Elliptical search regions oriented by velocity direction
#     - Adaptive cold-start: large radius for new tracks
    
#     Attributes:
#         model: YOLO model for detection/confirmation
#         config: Configuration object
#         bg_subtractor: Background subtractor (MOG2)
#         noise_filter: Optional AI noise filter
#         tracks: Dictionary of active tracks
#         next_track_id: Next available track ID
#         d_initial: Initial distance threshold
#         d_max: Maximum observed distance between frames
#         recorded_speeds: History of observed speeds
#         recorded_bee_areas: History of actual blob contour areas
#         min_blob_area_dynamic: Learned minimum blob area
#     """
    
#     def __init__(self, model, config: Optional[Config] = None, use_gpu: Optional[bool] = None):
#         """Initialize hybrid tracker with adaptive parameters.
        
#         Args:
#             model: YOLO model for confirmation
#             config: Configuration object
#             use_gpu: Use GPU if available (default: auto-detect)
#         """
#         self.model = model
#         self.config = config if config is not None else Config.default()
        
#         # Auto-detect GPU
#         if use_gpu is None:
#             self.use_gpu = self._detect_gpu()
#         else:
#             self.use_gpu = use_gpu
        
#         # Apply scaling
#         self._apply_config_scalling(self.config)
        
#         # Initialize background subtractor
#         self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
#             history=500,
#             varThreshold=16,
#             detectShadows=False
#         )
        
#         # Initialize AI noise filter (optional)
#         self.noise_filter = None
#         try:
#             from beemonitor.ml.bee_noise_filter import BeeNoiseFilter
            
#             # Try to find noise filter model
#             noise_filter_path = os.path.join(
#                 os.path.dirname(__file__), 
#                 "/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/training_output2/best_model.pth"
#             )
            
#             # Also check user-provided path in config
#             if hasattr(config, 'noise_filter_path') and config.noise_filter_path:
#                 noise_filter_path = config.noise_filter_path
            
#             if os.path.exists(noise_filter_path):
#                 self.noise_filter = BeeNoiseFilter(
#                     model_path=noise_filter_path,
#                     device='cpu',
#                     noise_threshold=0.8,
#                     image_size=64
#                 )
#                 logger.info(f"✅ AI noise filter loaded: {noise_filter_path}")
#             else:
#                 logger.warning(f"⚠️ Noise filter model not found: {noise_filter_path}")
#                 logger.warning("Tracking will continue without AI noise filtering")
#         except ImportError as e:
#             logger.warning(f"⚠️ BeeNoiseFilter not available (import error): {e}")
#             logger.warning("Tracking will continue without AI noise filtering")
#         except Exception as e:
#             logger.warning(f"⚠️ Could not initialize noise filter: {e}")
#             logger.warning("Tracking will continue without AI noise filtering")
        
#         # Tracking state
#         self.tracks: Dict[int, TrackState] = {}
#         self.next_track_id = 0
        
#         # Adaptive threshold parameters (from paper)
#         self.d_initial = getattr(self.config.tracking, 'initial_distance_threshold', 30.0)
#         self.d_max = self.d_initial
#         self.recorded_speeds = []
        
#         # Dynamic bee size tracking (from BLOB areas, not boxes)
#         self.recorded_bee_areas = []
#         self.min_blob_area_dynamic = 50  # Start conservative, will adapt
        
#         # Species mapping
#         self.label_map = self.config.tracking.label_map
#         self.tracking_classes = self.config.tracking.tracking_classes
        
#         logger.info(f"✅ Hybrid tracker initialized (GPU: {self.use_gpu})")
#         logger.info(f"🎯 Fully adaptive parameters ENABLED:")
#         logger.info(f"  - min_blob_area: starts at {self.min_blob_area_dynamic}px², learns from blob contours")
#         logger.info(f"  - search_region: elliptical, oriented by velocity direction")
#         logger.info(f"  - cold-start: adaptive radius (100px→70px→50px→adaptive ellipse)")
#         logger.info(f"  - resurrection: larger radius (120px→90px→70px→adaptive ellipse)")
#         logger.info(f"  - distance thresholds: STRICTLY enforced (even for DL)")
#         logger.info(f"  - AI noise filter: {'ENABLED' if self.noise_filter else 'DISABLED'}")
#         logger.info(f"  - strict duplicate prevention: ENABLED")
    
#     def _apply_config_scalling(self, config: Config):
#         """Apply scale factor to config parameters (set to 1.0 to disable)."""
#         self.scale_factor = 1.0
        
#         # Aspect ratio filtering (dimensionless)
#         self.min_aspect_ratio = self.config.tracking.min_blob_aspect_ratio
#         self.max_aspect_ratio = self.config.tracking.max_blob_aspect_ratio
        
#         # Area filtering (will be dynamic)
#         base_min_area = getattr(self.config.tracking, 'min_blob_area_pixels', 200)
#         base_max_area = getattr(self.config.tracking, 'max_blob_area_pixels', 5000)
#         self.min_blob_area_pixels = base_min_area * (self.scale_factor ** 2)
#         self.max_blob_area_pixels = base_max_area * (self.scale_factor ** 2)
        
#         logger.info(f"Scale factor: {self.scale_factor:.2f}x")
#         logger.info(f"Aspect ratio: {self.min_aspect_ratio:.2f}-{self.max_aspect_ratio:.2f}")
    
#     def _detect_gpu(self) -> bool:
#         """Detect if GPU is available."""
#         try:
#             import torch
#             if torch.cuda.is_available():
#                 logger.info(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
#                 return True
#         except ImportError:
#             pass
        
#         if cv2.cuda.getCudaEnabledDeviceCount() > 0:
#             logger.info("OpenCV CUDA detected")
#             return True
        
#         logger.info("No GPU detected, using CPU")
#         return False
    
#     def _initialize_background_from_video(
#         self,
#         video_path: str,
#         max_frames: int = 200,
#         target_clean_frames: int = 50,
#         config: Optional[Config] = None
#     ) -> int:
#         """Initialize BG model using bee-free frames."""
#         if config is None:
#             config = self.config
        
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")
        
#         logger.info("Initializing background model from bee-free frames...")
        
#         frame_num = 0
#         clean_frames_used = 0
        
#         while frame_num < max_frames and clean_frames_used < target_clean_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Check for bees with YOLO
#             results = self.model.predict(
#                 frame,
#                 conf=0.2,
#                 iou=config.tracking.iou_threshold,
#                 verbose=False,
#                 device='0' if self.use_gpu else 'cpu'
#             )
            
#             has_bees = False
#             if len(results) > 0 and results[0].boxes is not None:
#                 for cls in results[0].boxes.cls:
#                     if int(cls.cpu().numpy()) in self.tracking_classes:
#                         has_bees = True
#                         break
            
#             if not has_bees:
#                 self.bg_subtractor.apply(frame, learningRate=0.1)
#                 clean_frames_used += 1
            
#             frame_num += 1
        
#         cap.release()
        
#         logger.info(f"Background init: {clean_frames_used}/{frame_num} clean frames")
#         logger.info(f"Background model is now FROZEN (learningRate=0)")
        
#         return clean_frames_used
    
#     # =========================================================================
#     # MAIN TRACKING METHOD
#     # =========================================================================
    
#     def detect_and_track(
#         self,
#         video_path: str,
#         site_roi: BBox,
#         res_height: int,
#         res_width: int,
#         visualize: bool = False,
#         output_folder: str = "output",
#         config: Optional[Config] = None,
#         initialize_background: bool = True
#     ) -> pd.DataFrame:
#         """Hybrid tracking with adaptive cold-start handling."""
#         if config is None:
#             config = self.config
        
#         # Initialize BG model
#         if initialize_background:
#             self._initialize_background_from_video(video_path, config=config)
        
#         # Setup
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
        
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")
        
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         logger.info(f"Processing {total_frames} frames at {fps} fps")
        
#         # ROI mask
#         x1_roi, y1_roi, x2_roi, y2_roi = [int(c) for c in site_roi]
#         roi_mask = self._create_roi_mask(res_height, res_width, site_roi)
#         logger.info(f"ROI: ({x1_roi}, {y1_roi}) to ({x2_roi}, {y2_roi})")
        
#         # Visualization
#         output_video = None
#         if visualize:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             output_path = os.path.join(
#                 output_folder,
#                 f"tracking_{os.path.basename(video_path).rsplit('.', 1)[0]}.mp4"
#             )
#             output_video = cv2.VideoWriter(output_path, fourcc, fps, (res_width, res_height))
        
#         need_resize = (res_width != config.video.res_width or 
#                        res_height != config.video.res_height)
        
#         # State
#         frame_num = 0
#         all_detections = []
        
#         # Statistics
#         yolo_calls = 0
#         bg_only_updates = 0
#         track_resurrections = 0
#         new_tracks_created = 0
#         new_tracks_blocked = 0
#         dl_matches_rejected = 0
        
#         # Main loop
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             if need_resize:
#                 frame = cv2.resize(frame, (res_width, res_height))
            
#             # FG/BG Segmentation
#             fg_mask_full = self.bg_subtractor.apply(frame, learningRate=0)
#             fg_mask_full[fg_mask_full == 127] = 0
#             fg_mask = self._apply_roi_to_fgmask(fg_mask_full, roi_mask)
            
#             # Extract blobs
#             blobs = self._extract_blobs(frame, fg_mask, config, x_offset=0, y_offset=0)
            
#             # Predict tracks
#             predictions = self._predict_tracks(frame_num)
            
#             # HYBRID LOGIC
#             tracks_needing_dl = []
            
#             for track_id, pred in predictions.items():
#                 track = pred['track']
#                 pred_centroid = pred['centroid']
                
#                 # Get ADAPTIVE search region (larger for young tracks)
#                 search_region = self._get_search_region(track)
#                 nearby_blobs = self._count_blobs_in_search_region(blobs, pred_centroid, search_region)
                
#                 if len(nearby_blobs) == 0:
#                     tracks_needing_dl.append(track_id)
#                 elif len(nearby_blobs) == 1:
#                     self._update_track_with_blob(track, nearby_blobs[0], frame_num)
#                     bg_only_updates += 1
#                 elif len(nearby_blobs) <= 3:
#                     closest = self._find_closest_blob(nearby_blobs, pred_centroid)
#                     self._update_track_with_blob(track, closest, frame_num)
#                     bg_only_updates += 1
#                 else:
#                     tracks_needing_dl.append(track_id)
            
#             # Check for new insects
#             unassociated_blobs = self._find_unassociated_blobs(blobs, predictions)
#             if unassociated_blobs:
#                 if not tracks_needing_dl or 'new' not in tracks_needing_dl:
#                     tracks_needing_dl.append('new')
            
#             # Run DL when needed
#             yolo_detections = []
#             if tracks_needing_dl:
#                 yolo_calls += 1
#                 yolo_detections = self._run_yolo(frame, config, x_offset=0, y_offset=0)
                
#                 stats = self._update_tracks_with_yolo_strict(
#                     yolo_detections,
#                     tracks_needing_dl,
#                     predictions,
#                     frame_num
#                 )
#                 track_resurrections += stats['resurrections']
#                 new_tracks_created += stats['new_tracks']
#                 new_tracks_blocked += stats['blocked']
#                 dl_matches_rejected += stats['rejected_matches']
            
#             # Age out old tracks
#             self._age_tracks(frame_num)
            
#             # Record & visualize
#             frame_debug_data = {'blobs': blobs, 'yolo_detections': yolo_detections}
#             detections = self._get_current_detections(frame_num, frame_debug_data)
#             all_detections.extend(detections)
            
#             if visualize and output_video:
#                 viz_frame = self._visualize_frame(frame, frame_num, False, blobs, yolo_detections)
#                 output_video.write(viz_frame)
            
#             frame_num += 1
#             if frame_num % 100 == 0:
#                 logger.info(f"Progress: {frame_num}/{total_frames} | Tracks: {len(self.tracks)} | "
#                            f"DL: {yolo_calls} | Resurrected: {track_resurrections} | "
#                            f"New blocked: {new_tracks_blocked}")
        
#         # Cleanup
#         cap.release()
#         if output_video:
#             output_video.release()
        
#         # Statistics
#         logger.info(f"=== TRACKING COMPLETE ===")
#         logger.info(f"Total frames: {frame_num}")
#         logger.info(f"DL calls: {yolo_calls} ({yolo_calls/frame_num*100:.1f}%)")
#         logger.info(f"BG-only updates: {bg_only_updates}")
#         logger.info(f"Track resurrections: {track_resurrections}")
#         logger.info(f"New tracks created: {new_tracks_created}")
#         logger.info(f"New tracks blocked: {new_tracks_blocked}")
#         logger.info(f"DL matches rejected (outside region): {dl_matches_rejected}")
#         logger.info(f"Total detections: {len(all_detections)}")
        
#         logger.info(f"=== LEARNED PARAMETERS ===")
#         if self.recorded_speeds:
#             logger.info(f"Speed: min={min(self.recorded_speeds):.1f}, "
#                        f"median={np.median(self.recorded_speeds):.1f}, "
#                        f"max={self.d_max:.1f} px/frame")
#         if self.recorded_bee_areas:
#             logger.info(f"Bee area: min={min(self.recorded_bee_areas):.0f}, "
#                        f"median={np.median(self.recorded_bee_areas):.0f}, "
#                        f"max={max(self.recorded_bee_areas):.0f} px²")
#             logger.info(f"Final min_blob_area: {self.min_blob_area_dynamic:.0f}px²")
        
#         if len(all_detections) == 0:
#             logger.error("❌ NO DETECTIONS COLLECTED!")
#             return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
#         return self._convert_to_grouped_format(all_detections)
    
#     # =========================================================================
#     # ADAPTIVE THRESHOLDS
#     # =========================================================================
    
#     def _update_adaptive_thresholds(self, track: TrackState):
#         """Update adaptive thresholds based on observed movement."""
#         if len(track.trajectory_history) < 2:
#             return
        
#         last_pos = track.trajectory_history[-1][1]
#         prev_pos = track.trajectory_history[-2][1]
        
#         dist = float(np.sqrt(
#             (last_pos[0] - prev_pos[0])**2 +
#             (last_pos[1] - prev_pos[1])**2
#         ))
        
#         self.d_max = max(self.d_max, dist)
#         self.recorded_speeds.append(dist)
#         if len(self.recorded_speeds) > 1000:
#             self.recorded_speeds.pop(0)
    
#     def _update_bee_size_statistics(self, blob_area: float):
#         """Track actual bee blob sizes (contour areas)."""
#         self.recorded_bee_areas.append(blob_area)
        
#         if len(self.recorded_bee_areas) > 100:
#             self.recorded_bee_areas.pop(0)
        
#         if len(self.recorded_bee_areas) >= 10:
#             self.min_blob_area_dynamic = np.percentile(self.recorded_bee_areas, 25) * 0.5
#             self.min_blob_area_dynamic = max(50, min(500, self.min_blob_area_dynamic))
    
#     def _get_bg_threshold(self) -> float:
#         """Threshold for BG matches (MDTBS)."""
#         return max(self.d_initial, self.d_max)
    
#     def _get_dl_threshold(self, frames_without_detection: int) -> float:
#         """Threshold for DL matches (MDTDL)."""
#         base = self._get_bg_threshold()
#         tau_star = 10
        
#         if frames_without_detection <= tau_star:
#             return 2 * base
        
#         if len(self.recorded_speeds) > 10:
#             eta_min = np.percentile(self.recorded_speeds, 25)
#         else:
#             eta_min = self.d_initial
        
#         adaptive_term = min(
#             eta_min * (frames_without_detection - tau_star) / 100,
#             0.99 * base
#         )
        
#         return 2 * base + adaptive_term
    
#     def _get_search_region(self, track: TrackState) -> Dict:
#         """Get elliptical search region with ADAPTIVE COLD-START handling.
        
#         Young tracks don't have reliable velocity estimates yet, so we use
#         progressively smaller circular search regions until we have enough
#         trajectory data to estimate velocity reliably.
        
#         Search region progression:
#         - 1 trajectory point: 100px circle (brand new, catch fast bees)
#         - 2 trajectory points: 70px circle (one velocity sample, still uncertain)
#         - 3-4 trajectory points: 50px circle (building confidence)
#         - 5+ trajectory points: Adaptive ellipse (confident velocity estimate)
#         """
#         trajectory_points = len(track.trajectory_history)
        
#         # ✅ ADAPTIVE COLD START based on trajectory confidence
#         if trajectory_points == 1:
#             # Brand new track - use VERY large radius to catch fast-moving bees
#             radius = 100.0
#             logger.debug(f"Track {track.track_id}: NEW (1 pt), radius={radius:.0f}px")
#             return {
#                 'type': 'circle',
#                 'radius': radius,
#                 'major_axis': radius,
#                 'minor_axis': radius,
#                 'angle': 0.0
#             }
#         elif trajectory_points == 2:
#             # One velocity measurement - still very uncertain
#             radius = 70.0
#             logger.debug(f"Track {track.track_id}: YOUNG (2 pts), radius={radius:.0f}px")
#             return {
#                 'type': 'circle',
#                 'radius': radius,
#                 'major_axis': radius,
#                 'minor_axis': radius,
#                 'angle': 0.0
#             }
#         elif trajectory_points < 5:
#             # Building confidence - moderate radius
#             radius = 50.0
#             logger.debug(f"Track {track.track_id}: MATURING ({trajectory_points} pts), radius={radius:.0f}px")
#             return {
#                 'type': 'circle',
#                 'radius': radius,
#                 'major_axis': radius,
#                 'minor_axis': radius,
#                 'angle': 0.0
#             }
        
#         # ✅ MATURE TRACK: Use velocity-based adaptive ellipse
#         vx = float(track.kalman.statePost[2])
#         vy = float(track.kalman.statePost[3])
#         speed = float(np.sqrt(vx**2 + vy**2))
        
#         BASE_RADIUS = 20.0
#         SPEED_SCALE = 2.5
        
#         # Stationary: small circle
#         if speed < 2.0:
#             radius = BASE_RADIUS
#             return {
#                 'type': 'circle',
#                 'radius': radius,
#                 'major_axis': radius,
#                 'minor_axis': radius,
#                 'angle': 0.0
#             }
        
#         # Moving: ellipse oriented by velocity
#         major_axis = BASE_RADIUS + (speed * SPEED_SCALE)
#         major_axis = max(BASE_RADIUS, min(major_axis, 150.0))
        
#         # Elongation: faster = more stretched
#         elongation_ratio = min(3.0, 1.0 + (speed / 10.0))
#         minor_axis = major_axis / elongation_ratio
#         minor_axis = max(BASE_RADIUS, minor_axis)
        
#         angle = float(np.arctan2(vy, vx))
        
#         return {
#             'type': 'ellipse',
#             'major_axis': major_axis,
#             'minor_axis': minor_axis,
#             'angle': angle,
#             'radius': major_axis
#         }
    
#     def _get_expanded_search_region_for_resurrection(self, track: TrackState) -> Dict:
#         """Get EXPANDED search region for resurrection - LARGER for young tracks.
        
#         For young tracks with unreliable velocity, use even larger circular regions.
#         For mature tracks, use velocity-based expansion.
        
#         Resurrection region progression:
#         - 1 trajectory point: 120px circle (LARGER than 100px normal search)
#         - 2 trajectory points: 90px circle (LARGER than 70px normal search)
#         - 3-4 trajectory points: 70px circle (LARGER than 50px normal search)
#         - 5+ trajectory points: Velocity-based ellipse with expansion
#         """
#         trajectory_points = len(track.trajectory_history)
        
#         # ✅ YOUNG TRACKS: Use LARGER radius than normal search
#         if trajectory_points == 1:
#             # LARGER than 100px normal search
#             radius = 120.0
#             logger.debug(f"Track {track.track_id}: Resurrection NEW, radius={radius:.0f}px")
#             return {
#                 'type': 'circle',
#                 'radius': radius,
#                 'major_axis': radius,
#                 'minor_axis': radius,
#                 'angle': 0.0
#             }
#         elif trajectory_points == 2:
#             # LARGER than 70px normal search
#             radius = 90.0
#             logger.debug(f"Track {track.track_id}: Resurrection YOUNG, radius={radius:.0f}px")
#             return {
#                 'type': 'circle',
#                 'radius': radius,
#                 'major_axis': radius,
#                 'minor_axis': radius,
#                 'angle': 0.0
#             }
#         elif trajectory_points < 5:
#             # LARGER than 50px normal search
#             radius = 70.0
#             logger.debug(f"Track {track.track_id}: Resurrection MATURING, radius={radius:.0f}px")
#             return {
#                 'type': 'circle',
#                 'radius': radius,
#                 'major_axis': radius,
#                 'minor_axis': radius,
#                 'angle': 0.0
#             }
        
#         # ✅ MATURE TRACKS: Use velocity-based expansion
#         vx = float(track.kalman.statePost[2])
#         vy = float(track.kalman.statePost[3])
#         speed = float(np.sqrt(vx**2 + vy**2))
        
#         BASE_RADIUS = 40.0
#         SPEED_SCALE = 3.5
        
#         # Stationary mature tracks
#         if speed < 2.0:
#             radius = BASE_RADIUS * 1.5  # 60px
#             return {
#                 'type': 'circle',
#                 'radius': radius,
#                 'major_axis': radius,
#                 'minor_axis': radius,
#                 'angle': 0.0
#             }
        
#         # Moving mature tracks
#         major_axis = BASE_RADIUS + (speed * SPEED_SCALE)
#         major_axis = max(BASE_RADIUS, min(major_axis, 200.0))
        
#         elongation_ratio = min(2.5, 1.0 + (speed / 15.0))
#         minor_axis = major_axis / elongation_ratio
#         minor_axis = max(BASE_RADIUS, minor_axis)
        
#         angle = float(np.arctan2(vy, vx))
        
#         return {
#             'type': 'ellipse',
#             'major_axis': major_axis,
#             'minor_axis': minor_axis,
#             'angle': angle,
#             'radius': major_axis
#         }
    
#     def _is_point_in_search_region(
#         self,
#         point: Point,
#         center: Point,
#         search_region: Dict
#     ) -> bool:
#         """Check if point is inside search region (circle or ellipse)."""
#         px, py = point
#         cx, cy = center
        
#         dx = px - cx
#         dy = py - cy
        
#         if search_region['type'] == 'circle':
#             dist = float(np.sqrt(dx**2 + dy**2))
#             return dist <= search_region['radius']
        
#         # Ellipse: rotate to align with axes
#         angle = search_region['angle']
#         cos_a = np.cos(-angle)
#         sin_a = np.sin(-angle)
        
#         rx = dx * cos_a - dy * sin_a
#         ry = dx * sin_a + dy * cos_a
        
#         a = search_region['major_axis']
#         b = search_region['minor_axis']
        
#         ellipse_eq = (rx / a)**2 + (ry / b)**2
        
#         return ellipse_eq <= 1.0
    
#     # =========================================================================
#     # BLOB PROCESSING
#     # =========================================================================
    
#     def _extract_blobs(
#         self,
#         frame: np.ndarray,
#         fg_mask: np.ndarray,
#         config: Config,
#         x_offset: int,
#         y_offset: int
#     ) -> List[Dict]:
#         """Extract and filter blobs with AI noise filter."""
        
#         kernel = np.ones((5, 5), np.uint8)
#         opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
#         cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
#         contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         blobs = []
#         min_area = self.min_blob_area_dynamic
        
#         rejected_area = 0
#         rejected_solidity = 0
#         rejected_shape = 0
#         rejected_noise = 0
        
#         for contour in contours:
#             area = cv2.contourArea(contour)
            
#             if area < min_area:
#                 rejected_area += 1
#                 continue
            
#             x, y, w, h = cv2.boundingRect(contour)
#             bbox = (x + x_offset, y + y_offset, x + w + x_offset, y + h + y_offset)
            
#             if not self._is_valid_blob_shape(bbox):
#                 rejected_shape += 1
#                 continue
            
#             hull = cv2.convexHull(contour)
#             hull_area = cv2.contourArea(hull)
#             solidity = area / hull_area if hull_area > 0 else 0
            
#             if solidity < 0.5:
#                 rejected_solidity += 1
#                 continue
            
#             M = cv2.moments(contour)
#             if M["m00"] > 0:
#                 cx = M["m10"] / M["m00"]
#                 cy = M["m01"] / M["m00"]
#             else:
#                 cx = x + w / 2
#                 cy = y + h / 2
            
#             centroid = (cx + x_offset, cy + y_offset)
            
#             blobs.append({
#                 'bbox': bbox,
#                 'centroid': centroid,
#                 'area': area,
#                 'solidity': solidity,
#                 'contour': contour
#             })
        
#         # AI noise filter
#         if self.noise_filter and blobs:
#             filter_blobs = []
#             for blob in blobs:
#                 x1, y1, x2, y2 = blob['bbox']
#                 x = x1 - x_offset
#                 y = y1 - y_offset
#                 w = x2 - x1
#                 h = y2 - y1
#                 filter_blobs.append((x, y, w, h))
            
#             roi_height, roi_width = fg_mask.shape
#             frame_roi = frame[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width]
            
#             filtered_blobs = self.noise_filter.filter_blobs(frame_roi, filter_blobs)
            
#             kept_bboxes = set()
#             for x, y, w, h, prob in filtered_blobs:
#                 bbox_global = (x + x_offset, y + y_offset, 
#                               x + w + x_offset, y + h + y_offset)
#                 kept_bboxes.add(bbox_global)
            
#             original_count = len(blobs)
#             blobs = [b for b in blobs if b['bbox'] in kept_bboxes]
#             rejected_noise = original_count - len(blobs)
        
#         total = len(contours)
#         if total > 0:
#             logger.debug(f"Blobs: {len(blobs)}/{total} (rejected: area={rejected_area}, "
#                         f"shape={rejected_shape}, solid={rejected_solidity}, noise={rejected_noise})")
        
#         return blobs
    
#     def _is_valid_blob_shape(self, bbox: BBox) -> bool:
#         """Check aspect ratio and size."""
#         x1, y1, x2, y2 = bbox
#         width = x2 - x1
#         height = y2 - y1
        
#         if height == 0 or width == 0:
#             return False
        
#         aspect_ratio = width / height
#         if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
#             return False
        
#         area = width * height
#         if not (self.min_blob_area_pixels <= area <= self.max_blob_area_pixels):
#             return False
        
#         return True
    
#     def _count_blobs_in_search_region(
#         self,
#         blobs: List[Dict],
#         centroid: Point,
#         search_region: Dict
#     ) -> List[Dict]:
#         """Find blobs within elliptical search region."""
#         nearby = []
        
#         for blob in blobs:
#             if self._is_point_in_search_region(blob['centroid'], centroid, search_region):
#                 nearby.append(blob)
        
#         return nearby
    
#     def _find_closest_blob(self, blobs: List[Dict], centroid: Point) -> Dict:
#         """Find closest blob to centroid."""
#         cx, cy = centroid
#         min_dist = float('inf')
#         closest = None
        
#         for blob in blobs:
#             bx, by = blob['centroid']
#             dist = float(np.sqrt((bx - cx)**2 + (by - cy)**2))
#             if dist < min_dist:
#                 min_dist = dist
#                 closest = blob
        
#         return closest
    
#     def _find_unassociated_blobs(
#         self,
#         blobs: List[Dict],
#         predictions: Dict[int, Dict]
#     ) -> List[Dict]:
#         """Find blobs far from all tracks."""
#         unassociated = []
        
#         for blob in blobs:
#             is_near_track = False
            
#             for pred in predictions.values():
#                 track = pred['track']
#                 search_region = self._get_search_region(track)
                
#                 if self._is_point_in_search_region(blob['centroid'], pred['centroid'], search_region):
#                     is_near_track = True
#                     break
            
#             if not is_near_track:
#                 unassociated.append(blob)
        
#         return unassociated
    
#     # =========================================================================
#     # PREDICTION & TRACKING
#     # =========================================================================
    
#     def _predict_tracks(self, frame_num: int) -> Dict[int, Dict]:
#         """Predict track positions with velocity damping."""
#         predictions = {}
        
#         for track_id, track in self.tracks.items():
#             self._update_adaptive_thresholds(track)
            
#             frames_since_dl = frame_num - track.last_yolo_confirmation
#             if frames_since_dl > 0:
#                 damping = max(0.2, 1.0 - (frames_since_dl * 0.1))
#                 track.kalman.statePost[2] *= damping
#                 track.kalman.statePost[3] *= damping
            
#             prediction = track.kalman.predict()
#             pred_x, pred_y = float(prediction[0]), float(prediction[1])
            
#             x1, y1, x2, y2 = track.bbox
#             width = x2 - x1
#             height = y2 - y1
            
#             pred_bbox = (
#                 pred_x - width / 2,
#                 pred_y - height / 2,
#                 pred_x + width / 2,
#                 pred_y + height / 2
#             )
            
#             predictions[track_id] = {
#                 'centroid': (pred_x, pred_y),
#                 'bbox': pred_bbox,
#                 'track': track
#             }
        
#         return predictions
    
#     def _update_track_with_blob(
#         self,
#         track: TrackState,
#         blob: Dict,
#         frame_num: int
#     ):
#         """Update track using blob (BG result)."""
#         track.bbox = blob['bbox']
#         track.centroid = blob['centroid']
#         track.frames_without_detection = 0
#         track.age += 1
        
#         track.trajectory_history.append((frame_num, blob['centroid']))
#         if len(track.trajectory_history) > 30:
#             track.trajectory_history.pop(0)
        
#         measurement = np.array([[blob['centroid'][0]],
#                                 [blob['centroid'][1]]], dtype=np.float32)
#         track.kalman.correct(measurement)
        
#         # Learn from blob area
#         self._update_bee_size_statistics(blob['area'])
    
#     def _update_tracks_with_yolo_strict(
#         self,
#         yolo_detections: List[Dict],
#         tracks_needing_dl: List,
#         predictions: Dict[int, Dict],
#         frame_num: int
#     ) -> Dict:
#         """Update tracks with STRICT search region enforcement."""
#         stats = {
#             'resurrections': 0,
#             'new_tracks': 0,
#             'blocked': 0,
#             'rejected_matches': 0
#         }
        
#         if not yolo_detections:
#             for track_id in tracks_needing_dl:
#                 if track_id != 'new' and track_id in self.tracks:
#                     self.tracks[track_id].frames_without_detection += 1
#             return stats
        
#         used_detections = set()
        
#         # ================================================================
#         # STEP 1: Update existing tracks - STRICT search region enforcement
#         # ================================================================
#         for track_id in tracks_needing_dl:
#             if track_id == 'new':
#                 continue
            
#             if track_id not in predictions:
#                 continue
            
#             pred = predictions[track_id]
#             track = pred['track']
            
#             # Get search region for this track
#             search_region = self._get_search_region(track)
#             threshold = self._get_dl_threshold(track.frames_without_detection)
            
#             # STRICT: Only consider YOLO detections WITHIN search region
#             candidate_detections = []
#             for idx, det in enumerate(yolo_detections):
#                 if idx in used_detections:
#                     continue
                
#                 # Must be in search region
#                 if not self._is_point_in_search_region(det['centroid'], pred['centroid'], search_region):
#                     continue
                
#                 dist = float(np.sqrt(
#                     (float(det['centroid'][0]) - float(pred['centroid'][0]))**2 +
#                     (float(det['centroid'][1]) - float(pred['centroid'][1]))**2
#                 ))
                
#                 # Must be within adaptive threshold
#                 if dist < threshold:
#                     candidate_detections.append((idx, det, dist))
            
#             if candidate_detections:
#                 # Take closest valid candidate
#                 candidate_detections.sort(key=lambda x: x[2])
#                 best_idx, best_det, best_dist = candidate_detections[0]
                
#                 # Update track
#                 track.bbox = best_det['bbox']
#                 track.centroid = best_det['centroid']
#                 track.label = best_det['label']
#                 track.frames_without_detection = 0
#                 track.last_yolo_confirmation = frame_num
                
#                 track.trajectory_history.append((frame_num, best_det['centroid']))
#                 if len(track.trajectory_history) > 30:
#                     track.trajectory_history.pop(0)
                
#                 measurement = np.array([[best_det['centroid'][0]],
#                                         [best_det['centroid'][1]]], dtype=np.float32)
#                 track.kalman.correct(measurement)
                
#                 used_detections.add(best_idx)
#                 logger.debug(f"✅ Track {track_id} updated (dist={best_dist:.1f}, thresh={threshold:.1f})")
#             else:
#                 track.frames_without_detection += 1
#                 stats['rejected_matches'] += 1
#                 logger.debug(f"❌ Track {track_id} no valid DL match (outside region or too far)")
        
#         # ================================================================
#         # STEP 2: Resurrection - WITH EXPANDED SEARCH REGION CHECK
#         # ================================================================
#         for idx, det in enumerate(yolo_detections):
#             if idx in used_detections:
#                 continue
            
#             # Try resurrection with ALL tracks
#             best_resurrection = None
#             best_dist = float('inf')
#             best_track_id = None
            
#             for track_id, track in self.tracks.items():
#                 # Create EXPANDED search region for resurrection
#                 expanded_search_region = self._get_expanded_search_region_for_resurrection(track)
                
#                 # First check: Must be within expanded search region
#                 if not self._is_point_in_search_region(det['centroid'], track.centroid, expanded_search_region):
#                     continue
                
#                 # Second check: Distance threshold (as safety)
#                 track_threshold = self._get_dl_threshold(track.frames_without_detection) * 1.5
                
#                 dist = float(np.sqrt(
#                     (track.centroid[0] - det['centroid'][0])**2 +
#                     (track.centroid[1] - det['centroid'][1])**2
#                 ))
                
#                 if dist < track_threshold and dist < best_dist:
#                     best_dist = dist
#                     best_track_id = track_id
#                     best_resurrection = track
            
#             if best_resurrection:
#                 # Resurrect this track
#                 logger.info(f"♻️ Resurrecting track {best_track_id} (dist={best_dist:.1f}px)")
#                 best_resurrection.bbox = det['bbox']
#                 best_resurrection.centroid = det['centroid']
#                 best_resurrection.label = det['label']
#                 best_resurrection.frames_without_detection = 0
#                 best_resurrection.last_yolo_confirmation = frame_num
                
#                 best_resurrection.trajectory_history.append((frame_num, det['centroid']))
#                 if len(best_resurrection.trajectory_history) > 30:
#                     best_resurrection.trajectory_history.pop(0)
                
#                 measurement = np.array([[det['centroid'][0]],
#                                         [det['centroid'][1]]], dtype=np.float32)
#                 best_resurrection.kalman.correct(measurement)
                
#                 used_detections.add(idx)
#                 stats['resurrections'] += 1
#                 continue
            
#             # ================================================================
#             # STEP 3: Create new track - ULTRA STRICT
#             # ================================================================
#             can_create = True
#             min_distance = float('inf')
            
#             for track_id, track in self.tracks.items():
#                 # Check expanded search region
#                 expanded_region = self._get_expanded_search_region_for_resurrection(track)
                
#                 if self._is_point_in_search_region(det['centroid'], track.centroid, expanded_region):
#                     can_create = False
#                     logger.debug(f"🚫 New track blocked - within track {track_id}'s expanded region")
#                     stats['blocked'] += 1
#                     break
                
#                 # Also check distance as safety
#                 dist = float(np.sqrt(
#                     (track.centroid[0] - det['centroid'][0])**2 +
#                     (track.centroid[1] - det['centroid'][1])**2
#                 ))
                
#                 min_distance = min(min_distance, dist)
                
#                 # STRICT: Must be at least 2x DL threshold away
#                 safety_threshold = self._get_dl_threshold(track.frames_without_detection) * 2.0
                
#                 if dist < safety_threshold:
#                     can_create = False
#                     logger.debug(f"🚫 New track blocked - track {track_id} only {dist:.1f}px away "
#                                f"(safety={safety_threshold:.1f})")
#                     stats['blocked'] += 1
#                     break
            
#             if can_create:
#                 logger.info(f"✨ Creating new track (nearest={min_distance:.1f}px)")
#                 self._create_track(det, frame_num)
#                 stats['new_tracks'] += 1
        
#         return stats
    
#     def _create_track(self, detection: Dict, frame_num: int):
#         """Create new track from detection."""
#         kalman = cv2.KalmanFilter(4, 2)
#         kalman.measurementMatrix = np.array([[1, 0, 0, 0],
#                                               [0, 1, 0, 0]], dtype=np.float32)
#         kalman.transitionMatrix = np.array([[1, 0, 1, 0],
#                                              [0, 1, 0, 1],
#                                              [0, 0, 1, 0],
#                                              [0, 0, 0, 1]], dtype=np.float32)
#         kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
#         kalman.statePre = np.array([[detection['centroid'][0]],
#                                      [detection['centroid'][1]],
#                                      [0],
#                                      [0]], dtype=np.float32)
        
#         track = TrackState(
#             track_id=self.next_track_id,
#             bbox=detection['bbox'],
#             centroid=detection['centroid'],
#             kalman=kalman,
#             frames_without_detection=0,
#             label=detection.get('label', 'unknown'),
#             age=1,
#             last_yolo_confirmation=frame_num
#         )
        
#         # Initialize trajectory with first point
#         track.trajectory_history.append((frame_num, detection['centroid']))
        
#         self.tracks[self.next_track_id] = track
#         self.next_track_id += 1
    
#     def _age_tracks(self, frame_num: int):
#         """Remove old tracks."""
#         max_age = self.config.tracking.max_age
#         to_remove = []
        
#         for track_id, track in self.tracks.items():
#             if track.frames_without_detection > max_age:
#                 to_remove.append(track_id)
        
#         for track_id in to_remove:
#             logger.debug(f"⏰ Removing track {track_id} (age: {self.tracks[track_id].frames_without_detection})")
#             del self.tracks[track_id]
    
#     # =========================================================================
#     # IoU OVERLAP PREVENTION
#     # =========================================================================
    
#     def _has_box_overlap(self, bbox: BBox, overlap_threshold: float = 0.3) -> bool:
#         """Check if bbox overlaps any existing track."""
#         for track in self.tracks.values():
#             iou = self._compute_iou(bbox, track.bbox)
#             if iou > overlap_threshold:
#                 return True
#         return False
    
#     def _find_overlapping_track(self, bbox: BBox) -> Optional[int]:
#         """Find track with highest overlap."""
#         best_iou = 0
#         best_id = None
        
#         for track_id, track in self.tracks.items():
#             iou = self._compute_iou(bbox, track.bbox)
#             if iou > best_iou:
#                 best_iou = iou
#                 best_id = track_id
        
#         return best_id if best_iou > 0.3 else None
    
#     def _compute_iou(self, bbox1: BBox, bbox2: BBox) -> float:
#         """Compute IoU of two boxes."""
#         x1_1, y1_1, x2_1, y2_1 = bbox1
#         x1_2, y1_2, x2_2, y2_2 = bbox2
        
#         x1_i = max(x1_1, x1_2)
#         y1_i = max(y1_1, y1_2)
#         x2_i = min(x2_1, x2_2)
#         y2_i = min(y2_1, y2_2)
        
#         if x2_i < x1_i or y2_i < y1_i:
#             return 0.0
        
#         intersection = (x2_i - x1_i) * (y2_i - y1_i)
#         area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
#         area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
#         union = area1 + area2 - intersection
        
#         return intersection / union if union > 0 else 0.0
    
#     # =========================================================================
#     # HELPER METHODS
#     # =========================================================================
    
#     def _create_roi_mask(self, h: int, w: int, roi: BBox) -> np.ndarray:
#         """Create ROI mask."""
#         x1, y1, x2, y2 = map(int, roi)
#         mask = np.zeros((h, w), dtype=np.uint8)
#         if x2 > x1 and y2 > y1:
#             mask[y1:y2, x1:x2] = 255
#         return mask
    
#     def _apply_roi_to_fgmask(self, fg_mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
#         """Apply ROI mask to FG mask."""
#         return cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)
    
#     def _run_yolo(
#         self,
#         frame: np.ndarray,
#         config: Config,
#         x_offset: int,
#         y_offset: int
#     ) -> List[Dict]:
#         """Run YOLO detection."""
#         results = self.model.predict(
#             frame,
#             conf=config.tracking.confidence_threshold,
#             iou=config.tracking.iou_threshold,
#             verbose=False,
#             device='0' if self.use_gpu else 'cpu'
#         )
        
#         detections = []
        
#         if len(results) > 0 and results[0].boxes is not None:
#             for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
#                 class_id = int(cls.cpu().numpy())
                
#                 if class_id not in self.tracking_classes:
#                     continue
                
#                 x1, y1, x2, y2 = box.cpu().numpy()
#                 cx = (x1 + x2) / 2
#                 cy = (y1 + y2) / 2
                
#                 detections.append({
#                     'bbox': (x1 + x_offset, y1 + y_offset,
#                             x2 + x_offset, y2 + y_offset),
#                     'centroid': (cx + x_offset, cy + y_offset),
#                     'class_id': class_id,
#                     'label': self.label_map.get(class_id, f'class_{class_id}')
#                 })
        
#         return detections
    
#     def _get_current_detections(
#         self,
#         frame_num: int,
#         debug_data: Optional[Dict] = None
#     ) -> List[Dict]:
#         """Get detections from active tracks."""
#         detections = []
        
#         for track_id, track in self.tracks.items():
#             det = {
#                 'frame': frame_num,
#                 'track_id': track_id,
#                 'x1': track.bbox[0],
#                 'y1': track.bbox[1],
#                 'x2': track.bbox[2],
#                 'y2': track.bbox[3],
#                 'species': track.label
#             }
            
#             if debug_data:
#                 det['debug_blobs'] = debug_data.get('blobs', [])
#                 det['debug_yolo'] = debug_data.get('yolo_detections', [])
            
#             detections.append(det)
        
#         return detections
    
#     def _visualize_frame(
#         self,
#         frame: np.ndarray,
#         frame_num: int,
#         is_low_res: bool,
#         blobs: List[Dict] = None,
#         yolo_detections: List[Dict] = None
#     ) -> np.ndarray:
#         """Visualize with adaptive search regions."""
#         viz = frame.copy()
        
#         # Draw blobs (cyan)
#         if blobs:
#             for blob in blobs:
#                 x1, y1, x2, y2 = [int(c) for c in blob['bbox']]
#                 cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 255, 0), 1)
#                 cx, cy = [int(c) for c in blob['centroid']]
#                 cv2.circle(viz, (cx, cy), 2, (255, 255, 0), -1)
        
#         # Draw YOLO (orange)
#         if yolo_detections:
#             for det in yolo_detections:
#                 x1, y1, x2, y2 = [int(c) for c in det['bbox']]
#                 cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 165, 255), 2)
#                 cv2.putText(viz, f"DL:{det['label']}", (x1, y1-5),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
#         # Draw tracks with adaptive search regions
#         for track_id, track in self.tracks.items():
#             search_region = self._get_search_region(track)
            
#             cx, cy = [int(c) for c in track.centroid]
            
#             # Draw search region
#             if search_region['type'] == 'circle':
#                 cv2.circle(viz, (cx, cy), int(search_region['radius']), (128, 128, 128), 1)
#             else:
#                 axes = (int(search_region['major_axis']), int(search_region['minor_axis']))
#                 angle_deg = int(np.degrees(search_region['angle']))
#                 cv2.ellipse(viz, (cx, cy), axes, angle_deg, 0, 360, (128, 128, 128), 1)
                
#                 # Velocity arrow
#                 arrow_len = 30
#                 end_x = int(cx + arrow_len * np.cos(search_region['angle']))
#                 end_y = int(cy + arrow_len * np.sin(search_region['angle']))
#                 cv2.arrowedLine(viz, (cx, cy), (end_x, end_y), (255, 128, 0), 2, tipLength=0.3)
            
#             # Trajectory
#             if len(track.trajectory_history) > 1:
#                 for i in range(len(track.trajectory_history) - 1):
#                     pt1 = tuple(map(int, track.trajectory_history[i][1]))
#                     pt2 = tuple(map(int, track.trajectory_history[i+1][1]))
#                     cv2.line(viz, pt1, pt2, (255, 255, 0), 1)
            
#             # Box color by freshness
#             frames_since_dl = frame_num - track.last_yolo_confirmation
#             if frames_since_dl == 0:
#                 color = (0, 255, 0)  # Green
#             elif frames_since_dl < 10:
#                 color = (0, 255, 255)  # Yellow
#             else:
#                 color = (255, 0, 0)  # Blue
            
#             x1, y1, x2, y2 = [int(c) for c in track.bbox]
#             cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            
#             # Label with trajectory count
#             traj_pts = len(track.trajectory_history)
#             vx = float(track.kalman.statePost[2])
#             vy = float(track.kalman.statePost[3])
#             speed = float(np.sqrt(vx**2 + vy**2))
            
#             if search_region['type'] == 'circle':
#                 label = f"ID:{track_id} {track.label} (pts={traj_pts}, r={search_region['radius']:.0f})"
#             else:
#                 label = f"ID:{track_id} {track.label} (v={speed:.1f}, {search_region['major_axis']:.0f}x{search_region['minor_axis']:.0f})"
#             cv2.putText(viz, label, (x1, y1-5),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         # Legend
#         y = 120
#         cv2.putText(viz, "Legend:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
#         cv2.putText(viz, "Green=DL confirmed", (10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
#         cv2.putText(viz, "Yellow=Recent", (10, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
#         cv2.putText(viz, "Blue=BG only", (10, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
#         cv2.putText(viz, "Cyan=Blobs", (10, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
#         cv2.putText(viz, "Orange=DL", (10, y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)
#         cv2.putText(viz, "Gray=Search region", (10, y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128), 1)
#         cv2.putText(viz, "Orange arrow=Direction", (10, y+140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,128,0), 1)
        
#         cv2.putText(viz, f"min_area: {self.min_blob_area_dynamic:.0f}px2", 
#                    (10, y+170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
#         return viz
    
#     def _convert_to_grouped_format(self, all_detections: List[Dict]) -> pd.DataFrame:
#         """Convert flat detections to grouped format."""
#         if not all_detections:
#             return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
#         debug_data_by_frame = {}
#         for det in all_detections:
#             frame_num = det['frame']
#             if 'debug_blobs' in det and frame_num not in debug_data_by_frame:
#                 debug_data_by_frame[frame_num] = {
#                     'blobs': det.get('debug_blobs', []),
#                     'yolo': det.get('debug_yolo', [])
#                 }
        
#         clean_detections = []
#         for det in all_detections:
#             clean = {k: v for k, v in det.items() if k not in ['debug_blobs', 'debug_yolo']}
#             clean_detections.append(clean)
        
#         df = pd.DataFrame(clean_detections)
#         periods = self._split_into_periods(df, gap_threshold=int(self.config.tracking.max_age * 1.1))
        
#         result_rows = []
#         for period_df in periods:
#             track_groups = {}
            
#             for track_id in period_df['track_id'].unique():
#                 track_df = period_df[period_df['track_id'] == track_id].sort_values('frame')
#                 segments = self._split_track_by_gaps(track_df, gap_threshold=self.config.tracking.max_age)
                
#                 for seg_idx, seg_df in enumerate(segments):
#                     unique_id = f"{track_id}_{seg_idx}" if len(segments) > 1 else track_id
                    
#                     centroids = [((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
#                                 for _, row in seg_df.iterrows()]
#                     bboxes = [(row['x1'], row['y1'], row['x2'], row['y2'])
#                              for _, row in seg_df.iterrows()]
#                     frame_numbers = seg_df['frame'].tolist()
                    
#                     if len(frame_numbers) >= self.config.tracking.min_track_length:
#                         track_groups[unique_id] = (unique_id, centroids, bboxes, frame_numbers)
            
#             if not track_groups:
#                 continue
            
#             all_tracks = list(track_groups.values())
#             min_frame = period_df['frame'].min()
#             max_frame = period_df['frame'].max()
            
#             frame_detections = {}
#             for frame_num in period_df['frame'].unique():
#                 frame_df = period_df[period_df['frame'] == frame_num]
#                 frame_debug = debug_data_by_frame.get(int(frame_num), {'blobs': [], 'yolo': []})
                
#                 frame_detections[int(frame_num)] = {
#                     'boxes': [(row['x1'], row['y1'], row['x2'], row['y2'])
#                              for _, row in frame_df.iterrows()],
#                     'label': frame_df['species'].tolist(),
#                     'debug_blobs': frame_debug['blobs'],
#                     'debug_yolo': frame_debug['yolo']
#                 }
            
#             result_rows.append({
#                 'frame_number': (int(min_frame), int(max_frame)),
#                 'tracks': all_tracks,
#                 'detections': frame_detections
#             })
        
#         return pd.DataFrame(result_rows) if result_rows else pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
    
#     def _split_into_periods(self, df: pd.DataFrame, gap_threshold: int = 100) -> List[pd.DataFrame]:
#         """Split detections into activity periods."""
#         df = df.sort_values('frame')
#         frames = df['frame'].tolist()
        
#         periods = []
#         current_start = 0
        
#         for i in range(len(frames) - 1):
#             gap = frames[i + 1] - frames[i]
#             if gap > gap_threshold:
#                 periods.append(df.iloc[current_start:i+1].copy())
#                 current_start = i + 1
        
#         if current_start < len(df):
#             periods.append(df.iloc[current_start:].copy())
        
#         return periods
    
#     def _split_track_by_gaps(self, track_df: pd.DataFrame, gap_threshold: int = 30) -> List[pd.DataFrame]:
#         """Split track into segments by gaps."""
#         frames = track_df['frame'].tolist()
        
#         segments = []
#         current_start = 0
        
#         for i in range(len(frames) - 1):
#             gap = frames[i + 1] - frames[i]
#             if gap > gap_threshold:
#                 segments.append(track_df.iloc[current_start:i+1].copy())
#                 current_start = i + 1
        
#         if current_start < len(track_df):
#             segments.append(track_df.iloc[current_start:].copy())
        
#         return segments



















"""Hybrid motion detection and tracking system with adaptive cold-start handling.

Combines background subtraction (efficient) with deep learning (accurate)
for multi-insect tracking in bee hotel videos.

FIXES in this version:
- ROI expanded by 200px padding to prevent edge fragmentation
- Increased search radii: cold-start 150/100/70px (was 100/70/50)
- Increased resurrection radii: 180/120/90px (was 120/90/70)
- Relaxed resurrection logic (OR instead of AND for region/distance)
- Less aggressive new track blocking (1x threshold instead of 2x)
- Enhanced logging for debugging stationary tracks

Key features:
- Hybrid BG/DL logic: BG for simple cases, DL for complex
- Elliptical search regions oriented by velocity direction
- ADAPTIVE cold-start: Large radius for new tracks, narrows as confidence grows
- FIXED resurrection: Even larger radius for young tracks
- STRICT distance enforcement for DL detections (within search region only)
- Fully adaptive thresholds learned from actual bee behavior
- Dynamic blob filtering based on actual blob contour areas
- AI noise filter (optional fourth filter)
- Aggressive duplicate prevention and track resurrection
- Solidity filter for shape compactness
- IoU-based overlap prevention
- Kalman filtering with velocity damping
- ROI masking for efficient motion detection
"""

import logging
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass

from beemonitor.core.config import Config

logger = logging.getLogger(__name__)

# Type aliases
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class TrackState:
    """State for a tracked insect."""
    track_id: int
    bbox: BBox
    centroid: Point
    kalman: cv2.KalmanFilter
    frames_without_detection: int
    label: str
    age: int
    last_yolo_confirmation: int
    trajectory_history: list = None
    
    def __post_init__(self):
        if self.trajectory_history is None:
            self.trajectory_history = []


class MotionTracking:
    """Standalone hybrid tracker with adaptive cold-start handling.
    
    Learns from actual bee behavior instead of using hard-coded values:
    - Blob size threshold from actual blob contour areas
    - Distance thresholds from observed movement
    - Elliptical search regions oriented by velocity direction
    - Adaptive cold-start: large radius for new tracks
    
    Attributes:
        model: YOLO model for detection/confirmation
        config: Configuration object
        bg_subtractor: Background subtractor (MOG2)
        noise_filter: Optional AI noise filter
        tracks: Dictionary of active tracks
        next_track_id: Next available track ID
        d_initial: Initial distance threshold
        d_max: Maximum observed distance between frames
        recorded_speeds: History of observed speeds
        recorded_bee_areas: History of actual blob contour areas
        min_blob_area_dynamic: Learned minimum blob area
    """
    
    def __init__(self, model, config: Optional[Config] = None, use_gpu: Optional[bool] = None):
        """Initialize hybrid tracker with adaptive parameters.
        
        Args:
            model: YOLO model for confirmation
            config: Configuration object
            use_gpu: Use GPU if available (default: auto-detect)
        """
        self.model = model
        self.config = config if config is not None else Config.default()
        
        # Auto-detect GPU
        if use_gpu is None:
            self.use_gpu = self._detect_gpu()
        else:
            self.use_gpu = use_gpu
        
        # Apply scaling
        self._apply_config_scalling(self.config)
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Initialize AI noise filter (optional)
        self.noise_filter = None
        try:
            from beemonitor.ml.bee_noise_filter import BeeNoiseFilter
            
            # Try to find noise filter model
            noise_filter_path = os.path.join(
                os.path.dirname(__file__), 
                "/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/training_output2/best_model.pth"
            )
            
            # Also check user-provided path in config
            if hasattr(config, 'noise_filter_path') and config.noise_filter_path:
                noise_filter_path = config.noise_filter_path
            
            if os.path.exists(noise_filter_path):
                self.noise_filter = BeeNoiseFilter(
                    model_path=noise_filter_path,
                    device='cpu',
                    noise_threshold=0.9,
                    image_size=64
                )
                logger.info(f"✅ AI noise filter loaded: {noise_filter_path}")
            else:
                logger.warning(f"⚠️ Noise filter model not found: {noise_filter_path}")
                logger.warning("Tracking will continue without AI noise filtering")
        except ImportError as e:
            logger.warning(f"⚠️ BeeNoiseFilter not available (import error): {e}")
            logger.warning("Tracking will continue without AI noise filtering")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize noise filter: {e}")
            logger.warning("Tracking will continue without AI noise filtering")
        
        # Tracking state
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        
        # Adaptive threshold parameters (from paper)
        self.d_initial = getattr(self.config.tracking, 'initial_distance_threshold', 30.0)
        self.d_max = self.d_initial
        self.recorded_speeds = []
        
        # Dynamic bee size tracking (from BLOB areas, not boxes)
        self.recorded_bee_areas = []
        self.min_blob_area_dynamic = 50  # Start conservative, will adapt
        
        # Species mapping
        self.label_map = self.config.tracking.label_map
        self.tracking_classes = self.config.tracking.tracking_classes
        
        logger.info(f"✅ Hybrid tracker initialized (GPU: {self.use_gpu})")
        logger.info(f"🎯 Fully adaptive parameters ENABLED:")
        logger.info(f"  - min_blob_area: starts at {self.min_blob_area_dynamic}px², learns from blob contours")
        logger.info(f"  - search_region: elliptical, oriented by velocity direction")
        logger.info(f"  - cold-start: adaptive radius (150px→100px→70px→adaptive ellipse) [INCREASED]")
        logger.info(f"  - resurrection: larger radius (180px→120px→90px→adaptive ellipse) [INCREASED]")
        logger.info(f"  - distance thresholds: STRICTLY enforced (even for DL)")
        logger.info(f"  - AI noise filter: {'ENABLED' if self.noise_filter else 'DISABLED'}")
        logger.info(f"  - strict duplicate prevention: ENABLED")
        logger.info(f"  - ROI padding: 200px [NEW]")
    
    def _apply_config_scalling(self, config: Config):
        """Apply scale factor to config parameters (set to 1.0 to disable)."""
        self.scale_factor = 1.0
        
        # Aspect ratio filtering (dimensionless)
        self.min_aspect_ratio = self.config.tracking.min_blob_aspect_ratio
        self.max_aspect_ratio = self.config.tracking.max_blob_aspect_ratio
        
        # Area filtering (will be dynamic)
        base_min_area = getattr(self.config.tracking, 'min_blob_area_pixels', 200)
        base_max_area = getattr(self.config.tracking, 'max_blob_area_pixels', 5000)
        self.min_blob_area_pixels = base_min_area * (self.scale_factor ** 2)
        self.max_blob_area_pixels = base_max_area * (self.scale_factor ** 2)
        
        logger.info(f"Scale factor: {self.scale_factor:.2f}x")
        logger.info(f"Aspect ratio: {self.min_aspect_ratio:.2f}-{self.max_aspect_ratio:.2f}")
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
                return True
        except ImportError:
            pass
        
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("OpenCV CUDA detected")
            return True
        
        logger.info("No GPU detected, using CPU")
        return False
    
    def _initialize_background_from_video(
        self,
        video_path: str,
        max_frames: int = 200,
        target_clean_frames: int = 50,
        config: Optional[Config] = None
    ) -> int:
        """Initialize BG model using bee-free frames."""
        if config is None:
            config = self.config
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        logger.info("Initializing background model from bee-free frames...")
        
        frame_num = 0
        clean_frames_used = 0
        
        while frame_num < max_frames and clean_frames_used < target_clean_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check for bees with YOLO
            results = self.model.predict(
                frame,
                conf=0.2,
                iou=config.tracking.iou_threshold,
                verbose=False,
                device='0' if self.use_gpu else 'cpu'
            )
            
            has_bees = False
            if len(results) > 0 and results[0].boxes is not None:
                for cls in results[0].boxes.cls:
                    if int(cls.cpu().numpy()) in self.tracking_classes:
                        has_bees = True
                        break
            
            if not has_bees:
                self.bg_subtractor.apply(frame, learningRate=0.1)
                clean_frames_used += 1
            
            frame_num += 1
        
        cap.release()
        
        logger.info(f"Background init: {clean_frames_used}/{frame_num} clean frames")
        logger.info(f"Background model is now FROZEN (learningRate=0)")
        
        return clean_frames_used
    
    # =========================================================================
    # MAIN TRACKING METHOD
    # =========================================================================
    
    def detect_and_track(
        self,
        video_path: str,
        site_roi: BBox,
        res_height: int,
        res_width: int,
        visualize: bool = False,
        output_folder: str = "output",
        config: Optional[Config] = None,
        initialize_background: bool = True
    ) -> pd.DataFrame:
        """Hybrid tracking with adaptive cold-start handling."""
        if config is None:
            config = self.config
        
        # Initialize BG model
        if initialize_background:
            self._initialize_background_from_video(video_path, config=config)
        
        # Setup
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"Processing {total_frames} frames at {fps} fps")
        
        # ROI mask with padding
        x1_roi, y1_roi, x2_roi, y2_roi = [int(c) for c in site_roi]
        roi_mask = self._create_roi_mask(res_height, res_width, site_roi)
        logger.info(f"ROI (original): ({x1_roi}, {y1_roi}) to ({x2_roi}, {y2_roi})")
        
        # Visualization
        output_video = None
        if visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(
                output_folder,
                f"tracking_{os.path.basename(video_path).rsplit('.', 1)[0]}.mp4"
            )
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (res_width, res_height))
        
        need_resize = (res_width != config.video.res_width or 
                       res_height != config.video.res_height)
        
        # State
        frame_num = 0
        all_detections = []
        
        # Statistics
        yolo_calls = 0
        bg_only_updates = 0
        track_resurrections = 0
        new_tracks_created = 0
        new_tracks_blocked = 0
        dl_matches_rejected = 0
        
        # Main loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if need_resize:
                frame = cv2.resize(frame, (res_width, res_height))
            
            # FG/BG Segmentation
            fg_mask_full = self.bg_subtractor.apply(frame, learningRate=0)
            fg_mask_full[fg_mask_full == 127] = 0
            fg_mask = self._apply_roi_to_fgmask(fg_mask_full, roi_mask)
            
            # Extract blobs
            blobs = self._extract_blobs(frame, fg_mask, config, x_offset=0, y_offset=0)
            
            # Predict tracks
            predictions = self._predict_tracks(frame_num)
            
            # HYBRID LOGIC
            tracks_needing_dl = []
            
            for track_id, pred in predictions.items():
                track = pred['track']
                pred_centroid = pred['centroid']
                
                # Get ADAPTIVE search region (larger for young tracks)
                search_region = self._get_search_region(track)
                nearby_blobs = self._count_blobs_in_search_region(blobs, pred_centroid, search_region)
                
                if len(nearby_blobs) == 0:
                    tracks_needing_dl.append(track_id)
                elif len(nearby_blobs) == 1:
                    self._update_track_with_blob(track, nearby_blobs[0], frame_num)
                    bg_only_updates += 1
                elif len(nearby_blobs) <= 3:
                    closest = self._find_closest_blob(nearby_blobs, pred_centroid)
                    self._update_track_with_blob(track, closest, frame_num)
                    bg_only_updates += 1
                else:
                    tracks_needing_dl.append(track_id)
            
            # Check for new insects
            unassociated_blobs = self._find_unassociated_blobs(blobs, predictions)
            if unassociated_blobs:
                if not tracks_needing_dl or 'new' not in tracks_needing_dl:
                    tracks_needing_dl.append('new')
            
            # Run DL when needed
            yolo_detections = []
            if tracks_needing_dl:
                yolo_calls += 1
                yolo_detections = self._run_yolo(frame, config, x_offset=0, y_offset=0)
                
                stats = self._update_tracks_with_yolo_strict(
                    yolo_detections,
                    tracks_needing_dl,
                    predictions,
                    frame_num
                )
                track_resurrections += stats['resurrections']
                new_tracks_created += stats['new_tracks']
                new_tracks_blocked += stats['blocked']
                dl_matches_rejected += stats['rejected_matches']
            
            # Age out old tracks
            self._age_tracks(frame_num)
            
            # Record & visualize
            frame_debug_data = {'blobs': blobs, 'yolo_detections': yolo_detections}
            detections = self._get_current_detections(frame_num, frame_debug_data)
            all_detections.extend(detections)
            
            if visualize and output_video:
                viz_frame = self._visualize_frame(frame, frame_num, False, blobs, yolo_detections)
                output_video.write(viz_frame)
            
            frame_num += 1
            if frame_num % 100 == 0:
                logger.info(f"Progress: {frame_num}/{total_frames} | Tracks: {len(self.tracks)} | "
                           f"DL: {yolo_calls} | Resurrected: {track_resurrections} | "
                           f"New blocked: {new_tracks_blocked}")
        
        # Cleanup
        cap.release()
        if output_video:
            output_video.release()
        
        # Statistics
        logger.info(f"=== TRACKING COMPLETE ===")
        logger.info(f"Total frames: {frame_num}")
        logger.info(f"DL calls: {yolo_calls} ({yolo_calls/frame_num*100:.1f}%)")
        logger.info(f"BG-only updates: {bg_only_updates}")
        logger.info(f"Track resurrections: {track_resurrections}")
        logger.info(f"New tracks created: {new_tracks_created}")
        logger.info(f"New tracks blocked: {new_tracks_blocked}")
        logger.info(f"DL matches rejected (outside region): {dl_matches_rejected}")
        logger.info(f"Total detections: {len(all_detections)}")
        
        logger.info(f"=== LEARNED PARAMETERS ===")
        if self.recorded_speeds:
            logger.info(f"Speed: min={min(self.recorded_speeds):.1f}, "
                       f"median={np.median(self.recorded_speeds):.1f}, "
                       f"max={self.d_max:.1f} px/frame")
        if self.recorded_bee_areas:
            logger.info(f"Bee area: min={min(self.recorded_bee_areas):.0f}, "
                       f"median={np.median(self.recorded_bee_areas):.0f}, "
                       f"max={max(self.recorded_bee_areas):.0f} px²")
            logger.info(f"Final min_blob_area: {self.min_blob_area_dynamic:.0f}px²")
        
        if len(all_detections) == 0:
            logger.error("❌ NO DETECTIONS COLLECTED!")
            return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
        return self._convert_to_grouped_format(all_detections)
    
    # =========================================================================
    # ADAPTIVE THRESHOLDS
    # =========================================================================
    
    def _update_adaptive_thresholds(self, track: TrackState):
        """Update adaptive thresholds based on observed movement."""
        if len(track.trajectory_history) < 2:
            return
        
        last_pos = track.trajectory_history[-1][1]
        prev_pos = track.trajectory_history[-2][1]
        
        dist = float(np.sqrt(
            (last_pos[0] - prev_pos[0])**2 +
            (last_pos[1] - prev_pos[1])**2
        ))
        
        self.d_max = max(self.d_max, dist)
        self.recorded_speeds.append(dist)
        if len(self.recorded_speeds) > 1000:
            self.recorded_speeds.pop(0)
    
    def _update_bee_size_statistics(self, blob_area: float):
        """Track actual bee blob sizes (contour areas)."""
        self.recorded_bee_areas.append(blob_area)
        
        if len(self.recorded_bee_areas) > 100:
            self.recorded_bee_areas.pop(0)
        
        if len(self.recorded_bee_areas) >= 10:
            self.min_blob_area_dynamic = np.percentile(self.recorded_bee_areas, 25) * 0.5
            self.min_blob_area_dynamic = max(50, min(500, self.min_blob_area_dynamic))
    
    def _get_bg_threshold(self) -> float:
        """Threshold for BG matches (MDTBS)."""
        return max(self.d_initial, self.d_max)
    
    def _get_dl_threshold(self, frames_without_detection: int) -> float:
        """Threshold for DL matches (MDTDL)."""
        base = self._get_bg_threshold()
        tau_star = 10
        
        if frames_without_detection <= tau_star:
            return 2 * base
        
        if len(self.recorded_speeds) > 10:
            eta_min = np.percentile(self.recorded_speeds, 25)
        else:
            eta_min = self.d_initial
        
        adaptive_term = min(
            eta_min * (frames_without_detection - tau_star) / 100,
            0.99 * base
        )
        
        return 2 * base + adaptive_term
    
    def _get_search_region(self, track: TrackState) -> Dict:
        """Get elliptical search region with ADAPTIVE COLD-START handling.
        
        INCREASED radii for better coverage (fixes stationary track issues).
        
        Search region progression:
        - 1 trajectory point: 150px circle (was 100px)
        - 2 trajectory points: 100px circle (was 70px)
        - 3-4 trajectory points: 70px circle (was 50px)
        - 5+ trajectory points: Adaptive ellipse (confident velocity estimate)
        """
        trajectory_points = len(track.trajectory_history)
        
        # ✅ ADAPTIVE COLD START with INCREASED radii
        if trajectory_points == 1:
            radius = 150.0  # INCREASED from 100.0
            logger.debug(f"Track {track.track_id}: NEW (1 pt), radius={radius:.0f}px")
            return {
                'type': 'circle',
                'radius': radius,
                'major_axis': radius,
                'minor_axis': radius,
                'angle': 0.0
            }
        elif trajectory_points == 2:
            radius = 100.0  # INCREASED from 70.0
            logger.debug(f"Track {track.track_id}: YOUNG (2 pts), radius={radius:.0f}px")
            return {
                'type': 'circle',
                'radius': radius,
                'major_axis': radius,
                'minor_axis': radius,
                'angle': 0.0
            }
        elif trajectory_points < 5:
            radius = 70.0  # INCREASED from 50.0
            logger.debug(f"Track {track.track_id}: MATURING ({trajectory_points} pts), radius={radius:.0f}px")
            return {
                'type': 'circle',
                'radius': radius,
                'major_axis': radius,
                'minor_axis': radius,
                'angle': 0.0
            }
        
        # ✅ MATURE TRACK: Use velocity-based adaptive ellipse with INCREASED parameters
        vx = float(track.kalman.statePost[2])
        vy = float(track.kalman.statePost[3])
        speed = float(np.sqrt(vx**2 + vy**2))
        
        BASE_RADIUS = 25.0  # INCREASED from 20.0
        SPEED_SCALE = 3.0   # INCREASED from 2.5
        
        # Stationary: small circle
        if speed < 2.0:
            radius = BASE_RADIUS
            return {
                'type': 'circle',
                'radius': radius,
                'major_axis': radius,
                'minor_axis': radius,
                'angle': 0.0
            }
        
        # Moving: ellipse oriented by velocity
        major_axis = BASE_RADIUS + (speed * SPEED_SCALE)
        major_axis = max(BASE_RADIUS, min(major_axis, 200.0))  # INCREASED max from 150.0
        
        # Elongation: faster = more stretched
        elongation_ratio = min(3.0, 1.0 + (speed / 10.0))
        minor_axis = major_axis / elongation_ratio
        minor_axis = max(BASE_RADIUS, minor_axis)
        
        angle = float(np.arctan2(vy, vx))
        
        return {
            'type': 'ellipse',
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle,
            'radius': major_axis
        }
    
    def _get_expanded_search_region_for_resurrection(self, track: TrackState) -> Dict:
        """Get EXPANDED search region for resurrection - LARGER for young tracks.
        
        INCREASED radii to improve resurrection success rate.
        
        Resurrection region progression:
        - 1 trajectory point: 180px circle (was 120px)
        - 2 trajectory points: 120px circle (was 90px)
        - 3-4 trajectory points: 90px circle (was 70px)
        - 5+ trajectory points: Velocity-based ellipse with expansion
        """
        trajectory_points = len(track.trajectory_history)
        
        # ✅ YOUNG TRACKS: Use LARGER radius than normal search
        if trajectory_points == 1:
            radius = 180.0  # INCREASED from 120.0
            logger.debug(f"Track {track.track_id}: Resurrection NEW, radius={radius:.0f}px")
            return {
                'type': 'circle',
                'radius': radius,
                'major_axis': radius,
                'minor_axis': radius,
                'angle': 0.0
            }
        elif trajectory_points == 2:
            radius = 120.0  # INCREASED from 90.0
            logger.debug(f"Track {track.track_id}: Resurrection YOUNG, radius={radius:.0f}px")
            return {
                'type': 'circle',
                'radius': radius,
                'major_axis': radius,
                'minor_axis': radius,
                'angle': 0.0
            }
        elif trajectory_points < 5:
            radius = 90.0  # INCREASED from 70.0
            logger.debug(f"Track {track.track_id}: Resurrection MATURING, radius={radius:.0f}px")
            return {
                'type': 'circle',
                'radius': radius,
                'major_axis': radius,
                'minor_axis': radius,
                'angle': 0.0
            }
        
        # ✅ MATURE TRACKS: Use velocity-based expansion with INCREASED parameters
        vx = float(track.kalman.statePost[2])
        vy = float(track.kalman.statePost[3])
        speed = float(np.sqrt(vx**2 + vy**2))
        
        BASE_RADIUS = 50.0  # INCREASED from 40.0
        SPEED_SCALE = 4.0   # INCREASED from 3.5
        
        # Stationary mature tracks
        if speed < 2.0:
            radius = BASE_RADIUS * 1.5  # 75px
            return {
                'type': 'circle',
                'radius': radius,
                'major_axis': radius,
                'minor_axis': radius,
                'angle': 0.0
            }
        
        # Moving mature tracks
        major_axis = BASE_RADIUS + (speed * SPEED_SCALE)
        major_axis = max(BASE_RADIUS, min(major_axis, 250.0))  # INCREASED max from 200.0
        
        elongation_ratio = min(2.5, 1.0 + (speed / 15.0))
        minor_axis = major_axis / elongation_ratio
        minor_axis = max(BASE_RADIUS, minor_axis)
        
        angle = float(np.arctan2(vy, vx))
        
        return {
            'type': 'ellipse',
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle,
            'radius': major_axis
        }
    
    def _is_point_in_search_region(
        self,
        point: Point,
        center: Point,
        search_region: Dict
    ) -> bool:
        """Check if point is inside search region (circle or ellipse)."""
        px, py = point
        cx, cy = center
        
        dx = px - cx
        dy = py - cy
        
        if search_region['type'] == 'circle':
            dist = float(np.sqrt(dx**2 + dy**2))
            return dist <= search_region['radius']
        
        # Ellipse: rotate to align with axes
        angle = search_region['angle']
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        a = search_region['major_axis']
        b = search_region['minor_axis']
        
        ellipse_eq = (rx / a)**2 + (ry / b)**2
        
        return ellipse_eq <= 1.0
    
    # =========================================================================
    # BLOB PROCESSING
    # =========================================================================
    
    def _extract_blobs(
        self,
        frame: np.ndarray,
        fg_mask: np.ndarray,
        config: Config,
        x_offset: int,
        y_offset: int
    ) -> List[Dict]:
        """Extract and filter blobs with AI noise filter."""
        
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        min_area = self.min_blob_area_dynamic
        
        rejected_area = 0
        rejected_solidity = 0
        rejected_shape = 0
        rejected_noise = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                rejected_area += 1
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x + x_offset, y + y_offset, x + w + x_offset, y + h + y_offset)
            
            if not self._is_valid_blob_shape(bbox):
                rejected_shape += 1
                continue
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.5:
                rejected_solidity += 1
                continue
            
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2
                cy = y + h / 2
            
            centroid = (cx + x_offset, cy + y_offset)
            
            blobs.append({
                'bbox': bbox,
                'centroid': centroid,
                'area': area,
                'solidity': solidity,
                'contour': contour
            })
        
        # AI noise filter
        if self.noise_filter and blobs:
            filter_blobs = []
            for blob in blobs:
                x1, y1, x2, y2 = blob['bbox']
                x = x1 - x_offset
                y = y1 - y_offset
                w = x2 - x1
                h = y2 - y1
                filter_blobs.append((x, y, w, h))
            
            roi_height, roi_width = fg_mask.shape
            frame_roi = frame[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width]
            
            filtered_blobs = self.noise_filter.filter_blobs(frame_roi, filter_blobs)
            
            kept_bboxes = set()
            for x, y, w, h, prob in filtered_blobs:
                bbox_global = (x + x_offset, y + y_offset, 
                              x + w + x_offset, y + h + y_offset)
                kept_bboxes.add(bbox_global)
            
            original_count = len(blobs)
            blobs = [b for b in blobs if b['bbox'] in kept_bboxes]
            rejected_noise = original_count - len(blobs)
        
        total = len(contours)
        if total > 0:
            logger.debug(f"Blobs: {len(blobs)}/{total} (rejected: area={rejected_area}, "
                        f"shape={rejected_shape}, solid={rejected_solidity}, noise={rejected_noise})")
        
        return blobs
    
    def _is_valid_blob_shape(self, bbox: BBox) -> bool:
        """Check aspect ratio and size."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if height == 0 or width == 0:
            return False
        
        aspect_ratio = width / height
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return False
        
        area = width * height
        if not (self.min_blob_area_pixels <= area <= self.max_blob_area_pixels):
            return False
        
        return True
    
    def _count_blobs_in_search_region(
        self,
        blobs: List[Dict],
        centroid: Point,
        search_region: Dict
    ) -> List[Dict]:
        """Find blobs within elliptical search region."""
        nearby = []
        
        for blob in blobs:
            if self._is_point_in_search_region(blob['centroid'], centroid, search_region):
                nearby.append(blob)
        
        return nearby
    
    def _find_closest_blob(self, blobs: List[Dict], centroid: Point) -> Dict:
        """Find closest blob to centroid."""
        cx, cy = centroid
        min_dist = float('inf')
        closest = None
        
        for blob in blobs:
            bx, by = blob['centroid']
            dist = float(np.sqrt((bx - cx)**2 + (by - cy)**2))
            if dist < min_dist:
                min_dist = dist
                closest = blob
        
        return closest
    
    def _find_unassociated_blobs(
        self,
        blobs: List[Dict],
        predictions: Dict[int, Dict]
    ) -> List[Dict]:
        """Find blobs far from all tracks."""
        unassociated = []
        
        for blob in blobs:
            is_near_track = False
            
            for pred in predictions.values():
                track = pred['track']
                search_region = self._get_search_region(track)
                
                if self._is_point_in_search_region(blob['centroid'], pred['centroid'], search_region):
                    is_near_track = True
                    break
            
            if not is_near_track:
                unassociated.append(blob)
        
        return unassociated
    
    # =========================================================================
    # PREDICTION & TRACKING
    # =========================================================================
    
    def _predict_tracks(self, frame_num: int) -> Dict[int, Dict]:
        """Predict track positions with velocity damping."""
        predictions = {}
        
        for track_id, track in self.tracks.items():
            self._update_adaptive_thresholds(track)
            
            frames_since_dl = frame_num - track.last_yolo_confirmation
            if frames_since_dl > 0:
                damping = max(0.2, 1.0 - (frames_since_dl * 0.1))
                track.kalman.statePost[2] *= damping
                track.kalman.statePost[3] *= damping
            
            prediction = track.kalman.predict()
            pred_x, pred_y = float(prediction[0]), float(prediction[1])
            
            x1, y1, x2, y2 = track.bbox
            width = x2 - x1
            height = y2 - y1
            
            pred_bbox = (
                pred_x - width / 2,
                pred_y - height / 2,
                pred_x + width / 2,
                pred_y + height / 2
            )
            
            predictions[track_id] = {
                'centroid': (pred_x, pred_y),
                'bbox': pred_bbox,
                'track': track
            }
        
        return predictions
    
    def _update_track_with_blob(
        self,
        track: TrackState,
        blob: Dict,
        frame_num: int
    ):
        """Update track using blob (BG result)."""
        track.bbox = blob['bbox']
        track.centroid = blob['centroid']
        track.frames_without_detection = 0
        track.age += 1
        
        track.trajectory_history.append((frame_num, blob['centroid']))
        if len(track.trajectory_history) > 30:
            track.trajectory_history.pop(0)
        
        measurement = np.array([[blob['centroid'][0]],
                                [blob['centroid'][1]]], dtype=np.float32)
        track.kalman.correct(measurement)
        
        # Learn from blob area
        self._update_bee_size_statistics(blob['area'])
    
    def _update_tracks_with_yolo_strict(
        self,
        yolo_detections: List[Dict],
        tracks_needing_dl: List,
        predictions: Dict[int, Dict],
        frame_num: int
    ) -> Dict:
        """Update tracks with STRICT search region enforcement.
        
        FIXED: Relaxed resurrection logic and less aggressive new track blocking.
        """
        stats = {
            'resurrections': 0,
            'new_tracks': 0,
            'blocked': 0,
            'rejected_matches': 0
        }
        
        if not yolo_detections:
            for track_id in tracks_needing_dl:
                if track_id != 'new' and track_id in self.tracks:
                    self.tracks[track_id].frames_without_detection += 1
            return stats
        
        used_detections = set()
        
        # ================================================================
        # STEP 1: Update existing tracks - STRICT search region enforcement
        # ================================================================
        for track_id in tracks_needing_dl:
            if track_id == 'new':
                continue
            
            if track_id not in predictions:
                continue
            
            pred = predictions[track_id]
            track = pred['track']
            
            # Get search region for this track
            search_region = self._get_search_region(track)
            threshold = self._get_dl_threshold(track.frames_without_detection)
            
            # STRICT: Only consider YOLO detections WITHIN search region
            candidate_detections = []
            for idx, det in enumerate(yolo_detections):
                if idx in used_detections:
                    continue
                
                # Must be in search region
                if not self._is_point_in_search_region(det['centroid'], pred['centroid'], search_region):
                    continue
                
                dist = float(np.sqrt(
                    (float(det['centroid'][0]) - float(pred['centroid'][0]))**2 +
                    (float(det['centroid'][1]) - float(pred['centroid'][1]))**2
                ))
                
                # Must be within adaptive threshold
                if dist < threshold:
                    candidate_detections.append((idx, det, dist))
            
            if candidate_detections:
                # Take closest valid candidate
                candidate_detections.sort(key=lambda x: x[2])
                best_idx, best_det, best_dist = candidate_detections[0]
                
                # Update track
                logger.debug(f"✅ Track {track_id} updated with DL (dist={best_dist:.1f}px, thresh={threshold:.1f}px)")
                
                track.bbox = best_det['bbox']
                track.centroid = best_det['centroid']
                track.label = best_det['label']
                track.frames_without_detection = 0
                track.last_yolo_confirmation = frame_num
                
                track.trajectory_history.append((frame_num, best_det['centroid']))
                if len(track.trajectory_history) > 30:
                    track.trajectory_history.pop(0)
                
                measurement = np.array([[best_det['centroid'][0]],
                                        [best_det['centroid'][1]]], dtype=np.float32)
                track.kalman.correct(measurement)
                
                used_detections.add(best_idx)
            else:
                track.frames_without_detection += 1
                stats['rejected_matches'] += 1
                logger.debug(f"❌ Track {track_id} NO valid DL match (outside region or too far)")
        
        # ================================================================
        # STEP 2: RELAXED Resurrection - WITH EXPANDED SEARCH REGION CHECK
        # ================================================================
        for idx, det in enumerate(yolo_detections):
            if idx in used_detections:
                continue
            
            # Try resurrection with ALL tracks
            best_resurrection = None
            best_dist = float('inf')
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                # Create EXPANDED search region for resurrection
                expanded_search_region = self._get_expanded_search_region_for_resurrection(track)
                
                # RELAXED: Accept if EITHER in region OR within distance threshold
                in_region = self._is_point_in_search_region(det['centroid'], track.centroid, expanded_search_region)
                
                dist = float(np.sqrt(
                    (track.centroid[0] - det['centroid'][0])**2 +
                    (track.centroid[1] - det['centroid'][1])**2
                ))
                
                # More lenient threshold for resurrection
                track_threshold = self._get_dl_threshold(track.frames_without_detection) * 2.0
                
                # FIXED: OR instead of AND - more forgiving
                if (in_region or dist < track_threshold) and dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id
                    best_resurrection = track
            
            if best_resurrection:
                # Resurrect this track
                logger.info(f"♻️ RESURRECTED track {best_track_id} (dist={best_dist:.1f}px)")
                best_resurrection.bbox = det['bbox']
                best_resurrection.centroid = det['centroid']
                best_resurrection.label = det['label']
                best_resurrection.frames_without_detection = 0
                best_resurrection.last_yolo_confirmation = frame_num
                
                best_resurrection.trajectory_history.append((frame_num, det['centroid']))
                if len(best_resurrection.trajectory_history) > 30:
                    best_resurrection.trajectory_history.pop(0)
                
                measurement = np.array([[det['centroid'][0]],
                                        [det['centroid'][1]]], dtype=np.float32)
                best_resurrection.kalman.correct(measurement)
                
                used_detections.add(idx)
                stats['resurrections'] += 1
                continue
            
            # ================================================================
            # STEP 3: Create new track - LESS AGGRESSIVE BLOCKING
            # ================================================================
            can_create = True
            min_distance = float('inf')
            blocking_track = None
            
            for track_id, track in self.tracks.items():
                # Check expanded search region
                expanded_region = self._get_expanded_search_region_for_resurrection(track)
                
                in_region = self._is_point_in_search_region(det['centroid'], track.centroid, expanded_region)
                
                dist = float(np.sqrt(
                    (track.centroid[0] - det['centroid'][0])**2 +
                    (track.centroid[1] - det['centroid'][1])**2
                ))
                
                if dist < min_distance:
                    min_distance = dist
                
                # LESS AGGRESSIVE: Only block if BOTH in region AND within 1x threshold (not 2x)
                safety_threshold = self._get_dl_threshold(track.frames_without_detection)
                
                if in_region and dist < safety_threshold:
                    can_create = False
                    blocking_track = track_id
                    logger.debug(f"🚫 New track blocked - track {track_id} {dist:.1f}px away (thresh={safety_threshold:.1f}px)")
                    stats['blocked'] += 1
                    break
            
            if can_create:
                logger.info(f"✨ NEW track created (nearest={min_distance:.1f}px)")
                self._create_track(det, frame_num)
                stats['new_tracks'] += 1
            elif blocking_track is not None:
                # Extra logging for debugging stationary track issues
                logger.warning(f"⚠️ DL detection at {det['centroid']} blocked by track {blocking_track} "
                              f"(dist={min_distance:.1f}px) - possible stationary track issue!")
        
        return stats
    
    def _create_track(self, detection: Dict, frame_num: int):
        """Create new track from detection."""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        kalman.statePre = np.array([[detection['centroid'][0]],
                                     [detection['centroid'][1]],
                                     [0],
                                     [0]], dtype=np.float32)
        
        track = TrackState(
            track_id=self.next_track_id,
            bbox=detection['bbox'],
            centroid=detection['centroid'],
            kalman=kalman,
            frames_without_detection=0,
            label=detection.get('label', 'unknown'),
            age=1,
            last_yolo_confirmation=frame_num
        )
        
        # Initialize trajectory with first point
        track.trajectory_history.append((frame_num, detection['centroid']))
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
    
    def _age_tracks(self, frame_num: int):
        """Remove old tracks."""
        max_age = self.config.tracking.max_age
        to_remove = []
        
        for track_id, track in self.tracks.items():
            if track.frames_without_detection > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            logger.debug(f"⏰ Removing track {track_id} (age: {self.tracks[track_id].frames_without_detection})")
            del self.tracks[track_id]
    
    # =========================================================================
    # IoU OVERLAP PREVENTION
    # =========================================================================
    
    def _has_box_overlap(self, bbox: BBox, overlap_threshold: float = 0.3) -> bool:
        """Check if bbox overlaps any existing track."""
        for track in self.tracks.values():
            iou = self._compute_iou(bbox, track.bbox)
            if iou > overlap_threshold:
                return True
        return False
    
    def _find_overlapping_track(self, bbox: BBox) -> Optional[int]:
        """Find track with highest overlap."""
        best_iou = 0
        best_id = None
        
        for track_id, track in self.tracks.items():
            iou = self._compute_iou(bbox, track.bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = track_id
        
        return best_id if best_iou > 0.3 else None
    
    def _compute_iou(self, bbox1: BBox, bbox2: BBox) -> float:
        """Compute IoU of two boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _create_roi_mask(self, h: int, w: int, roi: BBox) -> np.ndarray:
        """Create ROI mask with 200px padding to prevent edge fragmentation."""
        x1, y1, x2, y2 = map(int, roi)
        
        # Add 200px padding (clip to frame bounds)
        padding = 200
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        
        logger.info(f"✅ ROI with padding: ({x1}, {y1}) to ({x2}, {y2}), padding={padding}px")
        return mask
    
    def _apply_roi_to_fgmask(self, fg_mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """Apply ROI mask to FG mask."""
        return cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)
    
    def _run_yolo(
        self,
        frame: np.ndarray,
        config: Config,
        x_offset: int,
        y_offset: int
    ) -> List[Dict]:
        """Run YOLO detection."""
        results = self.model.predict(
            frame,
            conf=config.tracking.confidence_threshold,
            iou=config.tracking.iou_threshold,
            verbose=False,
            device='0' if self.use_gpu else 'cpu'
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                class_id = int(cls.cpu().numpy())
                
                if class_id not in self.tracking_classes:
                    continue
                
                x1, y1, x2, y2 = box.cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                detections.append({
                    'bbox': (x1 + x_offset, y1 + y_offset,
                            x2 + x_offset, y2 + y_offset),
                    'centroid': (cx + x_offset, cy + y_offset),
                    'class_id': class_id,
                    'label': self.label_map.get(class_id, f'class_{class_id}')
                })
        
        return detections
    
    def _get_current_detections(
        self,
        frame_num: int,
        debug_data: Optional[Dict] = None
    ) -> List[Dict]:
        """Get detections from active tracks."""
        detections = []
        
        for track_id, track in self.tracks.items():
            det = {
                'frame': frame_num,
                'track_id': track_id,
                'x1': track.bbox[0],
                'y1': track.bbox[1],
                'x2': track.bbox[2],
                'y2': track.bbox[3],
                'species': track.label
            }
            
            if debug_data:
                det['debug_blobs'] = debug_data.get('blobs', [])
                det['debug_yolo'] = debug_data.get('yolo_detections', [])
            
            detections.append(det)
        
        return detections
    
    def _visualize_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        is_low_res: bool,
        blobs: List[Dict] = None,
        yolo_detections: List[Dict] = None
    ) -> np.ndarray:
        """Visualize with adaptive search regions."""
        viz = frame.copy()
        
        # Draw blobs (cyan)
        if blobs:
            for blob in blobs:
                x1, y1, x2, y2 = [int(c) for c in blob['bbox']]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cx, cy = [int(c) for c in blob['centroid']]
                cv2.circle(viz, (cx, cy), 2, (255, 255, 0), -1)
        
        # Draw YOLO (orange)
        if yolo_detections:
            for det in yolo_detections:
                x1, y1, x2, y2 = [int(c) for c in det['bbox']]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(viz, f"DL:{det['label']}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        # Draw tracks with adaptive search regions
        for track_id, track in self.tracks.items():
            search_region = self._get_search_region(track)
            
            cx, cy = [int(c) for c in track.centroid]
            
            # Draw search region
            if search_region['type'] == 'circle':
                cv2.circle(viz, (cx, cy), int(search_region['radius']), (128, 128, 128), 1)
            else:
                axes = (int(search_region['major_axis']), int(search_region['minor_axis']))
                angle_deg = int(np.degrees(search_region['angle']))
                cv2.ellipse(viz, (cx, cy), axes, angle_deg, 0, 360, (128, 128, 128), 1)
                
                # Velocity arrow
                arrow_len = 30
                end_x = int(cx + arrow_len * np.cos(search_region['angle']))
                end_y = int(cy + arrow_len * np.sin(search_region['angle']))
                cv2.arrowedLine(viz, (cx, cy), (end_x, end_y), (255, 128, 0), 2, tipLength=0.3)
            
            # Trajectory
            if len(track.trajectory_history) > 1:
                for i in range(len(track.trajectory_history) - 1):
                    pt1 = tuple(map(int, track.trajectory_history[i][1]))
                    pt2 = tuple(map(int, track.trajectory_history[i+1][1]))
                    cv2.line(viz, pt1, pt2, (255, 255, 0), 1)
            
            # Box color by freshness
            frames_since_dl = frame_num - track.last_yolo_confirmation
            if frames_since_dl == 0:
                color = (0, 255, 0)  # Green
            elif frames_since_dl < 10:
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 0, 0)  # Blue
            
            x1, y1, x2, y2 = [int(c) for c in track.bbox]
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            
            # Label with trajectory count
            traj_pts = len(track.trajectory_history)
            vx = float(track.kalman.statePost[2])
            vy = float(track.kalman.statePost[3])
            speed = float(np.sqrt(vx**2 + vy**2))
            
            if search_region['type'] == 'circle':
                label = f"ID:{track_id} {track.label} (pts={traj_pts}, r={search_region['radius']:.0f})"
            else:
                label = f"ID:{track_id} {track.label} (v={speed:.1f}, {search_region['major_axis']:.0f}x{search_region['minor_axis']:.0f})"
            cv2.putText(viz, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Legend
        y = 120
        cv2.putText(viz, "Legend:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(viz, "Green=DL confirmed", (10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        cv2.putText(viz, "Yellow=Recent", (10, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(viz, "Blue=BG only", (10, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
        cv2.putText(viz, "Cyan=Blobs", (10, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        cv2.putText(viz, "Orange=DL", (10, y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)
        cv2.putText(viz, "Gray=Search region", (10, y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128), 1)
        cv2.putText(viz, "Orange arrow=Direction", (10, y+140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,128,0), 1)
        
        cv2.putText(viz, f"min_area: {self.min_blob_area_dynamic:.0f}px2", 
                   (10, y+170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        return viz
    
    def _convert_to_grouped_format(self, all_detections: List[Dict]) -> pd.DataFrame:
        """Convert flat detections to grouped format."""
        if not all_detections:
            return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
        debug_data_by_frame = {}
        for det in all_detections:
            frame_num = det['frame']
            if 'debug_blobs' in det and frame_num not in debug_data_by_frame:
                debug_data_by_frame[frame_num] = {
                    'blobs': det.get('debug_blobs', []),
                    'yolo': det.get('debug_yolo', [])
                }
        
        clean_detections = []
        for det in all_detections:
            clean = {k: v for k, v in det.items() if k not in ['debug_blobs', 'debug_yolo']}
            clean_detections.append(clean)
        
        df = pd.DataFrame(clean_detections)
        periods = self._split_into_periods(df, gap_threshold=int(self.config.tracking.max_age * 1.1))
        
        result_rows = []
        for period_df in periods:
            track_groups = {}
            
            for track_id in period_df['track_id'].unique():
                track_df = period_df[period_df['track_id'] == track_id].sort_values('frame')
                segments = self._split_track_by_gaps(track_df, gap_threshold=self.config.tracking.max_age)
                
                for seg_idx, seg_df in enumerate(segments):
                    unique_id = f"{track_id}_{seg_idx}" if len(segments) > 1 else track_id
                    
                    centroids = [((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
                                for _, row in seg_df.iterrows()]
                    bboxes = [(row['x1'], row['y1'], row['x2'], row['y2'])
                             for _, row in seg_df.iterrows()]
                    frame_numbers = seg_df['frame'].tolist()
                    
                    if len(frame_numbers) >= self.config.tracking.min_track_length:
                        track_groups[unique_id] = (unique_id, centroids, bboxes, frame_numbers)
            
            if not track_groups:
                continue
            
            all_tracks = list(track_groups.values())
            min_frame = period_df['frame'].min()
            max_frame = period_df['frame'].max()
            
            frame_detections = {}
            for frame_num in period_df['frame'].unique():
                frame_df = period_df[period_df['frame'] == frame_num]
                frame_debug = debug_data_by_frame.get(int(frame_num), {'blobs': [], 'yolo': []})
                
                frame_detections[int(frame_num)] = {
                    'boxes': [(row['x1'], row['y1'], row['x2'], row['y2'])
                             for _, row in frame_df.iterrows()],
                    'label': frame_df['species'].tolist(),
                    'debug_blobs': frame_debug['blobs'],
                    'debug_yolo': frame_debug['yolo']
                }
            
            result_rows.append({
                'frame_number': (int(min_frame), int(max_frame)),
                'tracks': all_tracks,
                'detections': frame_detections
            })
        
        return pd.DataFrame(result_rows) if result_rows else pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
    
    def _split_into_periods(self, df: pd.DataFrame, gap_threshold: int = 100) -> List[pd.DataFrame]:
        """Split detections into activity periods."""
        df = df.sort_values('frame')
        frames = df['frame'].tolist()
        
        periods = []
        current_start = 0
        
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap > gap_threshold:
                periods.append(df.iloc[current_start:i+1].copy())
                current_start = i + 1
        
        if current_start < len(df):
            periods.append(df.iloc[current_start:].copy())
        
        return periods
    
    def _split_track_by_gaps(self, track_df: pd.DataFrame, gap_threshold: int = 30) -> List[pd.DataFrame]:
        """Split track into segments by gaps."""
        frames = track_df['frame'].tolist()
        
        segments = []
        current_start = 0
        
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap > gap_threshold:
                segments.append(track_df.iloc[current_start:i+1].copy())
                current_start = i + 1
        
        if current_start < len(track_df):
            segments.append(track_df.iloc[current_start:].copy())
        
        return segments