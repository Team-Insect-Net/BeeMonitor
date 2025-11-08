# """HyDaT-inspired motion detection and tracking.

# This module implements the HyDaT (Hybrid Detection and Tracking) approach:
# 1. Continuous FG/BG segmentation for motion detection
# 2. Track foreground blobs using predictions (Kalman filter)
# 3. Associate blobs to predicted positions
# 4. Run YOLO only when:
#    - New insects detected (blob with no track association)
#    - Missing insects (track with no blob association)
#    - Unassociated detections/predictions

# This dramatically reduces YOLO inference calls while maintaining accuracy.
# """

# import logging
# from typing import Dict, List, Tuple, Optional, Set
# import cv2
# import numpy as np
# import pandas as pd
# import os
# from dataclasses import dataclass
# from scipy.optimize import linear_sum_assignment

# from beemonitor.core.config import Config


# logger = logging.getLogger(__name__)

# # Type aliases
# BBox = Tuple[float, float, float, float]
# Point = Tuple[float, float]


# @dataclass
# class TrackState:
#     """State for a tracked object."""
#     track_id: int
#     bbox: BBox
#     centroid: Point
#     kalman: cv2.KalmanFilter
#     frames_without_detection: int
#     species: str
#     age: int
#     last_yolo_confirmation: int  # Frame number of last YOLO confirmation
#     trajectory_history: list = None  # List of (frame_num, centroid) for visualization
    
#     def __post_init__(self):
#         if self.trajectory_history is None:
#             self.trajectory_history = []


# class HyDaTTracker:
#     """HyDaT-inspired tracker with FG/BG segmentation and selective YOLO.
    
#     Features:
#     - Continuous FG/BG segmentation (MOG2)
#     - Kalman filter prediction for each track
#     - Hungarian algorithm for blob-to-track association
#     - YOLO only when association fails
#     - Low-resolution mode when no activity
    
#     Attributes:
#         model: YOLO model for confirmation
#         config: Configuration object
#         bg_subtractor: Background subtractor (MOG2)
#         bg_subtractor_lowres: Background subtractor for low-res mode
#         tracks: Dictionary of active tracks
#         next_track_id: Next available track ID
#         use_gpu: Whether GPU is available
#     """
    
#     def __init__(self, model, config: Optional[Config] = None, use_gpu: Optional[bool] = None):
#         """Initialize HyDaT tracker.
        
#         Args:
#             model: YOLO model for confirmation
#             config: Configuration object
#             use_gpu: Use GPU if available (default: auto-detect)
#         """
#         self.model = model
#         self.config = config if config is not None else Config.default()
        
#         # Calculate scale factor from hotel/ROI size
#         # This automatically adjusts pixel-based parameters for camera distance
#         self.scale_factor = self._calculate_scale_factor()
        
#         # Read optimized parameters from config (all tuned in config.py)
#         # Apply scaling for pixel-based parameters
        
#         # Aspect ratio filtering for FG blobs (dimensionless - no scaling)
#         self.min_aspect_ratio = self.config.tracking.min_blob_aspect_ratio
#         self.max_aspect_ratio = self.config.tracking.max_blob_aspect_ratio
        
#         # Area filtering (SCALED by scale_factor^2 - area scales quadratically)
#         base_min_area = getattr(self.config.tracking, 'min_blob_area_pixels', 200)
#         base_max_area = getattr(self.config.tracking, 'max_blob_area_pixels', 5000)
#         self.min_blob_area_pixels = base_min_area * (self.scale_factor ** 2)
#         self.max_blob_area_pixels = base_max_area * (self.scale_factor ** 2)
        
#         # Distance thresholds (SCALED linearly - distance scales linearly)
#         base_new_track_dist = getattr(self.config.tracking, 'new_track_distance_threshold', 100)
#         base_proximity_check = getattr(self.config.tracking, 'new_track_proximity_check', 50)
#         self.new_track_distance_threshold = base_new_track_dist * self.scale_factor
#         self.new_track_proximity_check = base_proximity_check * self.scale_factor
        
#         # Scale association threshold base (critical for blob-track matching)
#         base_assoc_threshold = self.config.tracking.association_threshold_base
#         self.config.tracking.association_threshold_base = base_assoc_threshold * self.scale_factor
        
#         # Scale max blob distance from hotel
#         base_max_blob_dist = getattr(self.config.tracking, 'max_blob_distance_from_hotel', 1800.0)
#         if base_max_blob_dist != float('inf'):
#             self.config.tracking.max_blob_distance_from_hotel = base_max_blob_dist * self.scale_factor
        
#         # Scale nest padding (used in event detection and visualization)
#         if hasattr(self.config, 'nest'):
#             base_padding_x = getattr(self.config.nest, 'padding_x_base', 5)
#             base_padding_y = getattr(self.config.nest, 'padding_y_base', 7)
#             # Store scaled padding for later use
#             self.nest_padding_x_scaled = base_padding_x * self.scale_factor
#             self.nest_padding_y_scaled = base_padding_y * self.scale_factor
        
#         # Auto-detect GPU
#         if use_gpu is None:
#             self.use_gpu = self._detect_gpu()
#         else:
#             self.use_gpu = use_gpu
        
#         # Initialize background subtractors
#         self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
#             history=500,
#             varThreshold=16,
#             detectShadows=True
#         )
        
#         self.bg_subtractor_lowres = cv2.createBackgroundSubtractorMOG2(
#             history=500,
#             varThreshold=16,
#             detectShadows=True
#         )
        
#         # Tracking state
#         self.tracks: Dict[int, TrackState] = {}
#         self.next_track_id = 0
        
#         # Species mapping
#         self.label_map = self.config.tracking.label_map
#         self.tracking_classes = self.config.tracking.tracking_classes
        
#         logger.info(f"Initialized HyDaTTracker (GPU: {self.use_gpu})")
#         logger.info(f"=== AUTO-SCALING APPLIED ===")
#         logger.info(f"Scale factor: {self.scale_factor:.2f}x (hotel size relative to reference)")
#         logger.info(f"Scaled parameters:")
#         logger.info(f"  Area filter: {int(self.min_blob_area_pixels)}-{int(self.max_blob_area_pixels)} px²")
#         logger.info(f"  Association threshold: {self.config.tracking.association_threshold_base:.1f} px")
#         logger.info(f"  New track distance: {int(self.new_track_distance_threshold)} px")
#         logger.info(f"  Proximity check: {int(self.new_track_proximity_check)} px")
#         if hasattr(self.config.tracking, 'max_blob_distance_from_hotel'):
#             max_dist = self.config.tracking.max_blob_distance_from_hotel
#             if max_dist != float('inf'):
#                 logger.info(f"  Max blob distance: {int(max_dist)} px")
#         logger.info(f"Unscaled parameters:")
#         logger.info(f"  Max age: {self.config.tracking.max_age} frames")
#         logger.info(f"  Aspect ratio: {self.min_aspect_ratio:.2f}-{self.max_aspect_ratio:.2f}")
#         logger.info(f"  Adaptive association: {'ENABLED' if self.config.tracking.adaptive_association else 'DISABLED'}")
#         logger.info(f"  Tracking classes: {self.tracking_classes}")
    
#     def _calculate_scale_factor(self) -> float:
#         """Calculate scale factor based on hotel/ROI size relative to frame.
        
#         This automatically adjusts pixel-based parameters (area, distance, etc.)
#         based on camera distance. When hotel is far (small in frame), parameters
#         scale down proportionally.
        
#         Reference assumptions (baseline for parameter tuning):
#         - Frame: 1920x1080 (Full HD)
#         - Hotel: ~800x400 pixels (close-up, 60-70% of frame height)
#         - Bee blob size: ~200-5000 pixels²
#         - Association distance: ~100 pixels
        
#         Scaling logic:
#         - Linear dimensions (width, height) scale by factor
#         - Areas scale by factor²
#         - Distances scale by factor
        
#         Returns:
#             Scale factor (1.0 = reference size, 0.5 = half size/far camera, 2.0 = double size/close camera)
#         """
#         # Check if auto-scaling is disabled
#         if getattr(self.config.tracking, 'disable_auto_scaling', False):
#             logger.info("Auto-scaling DISABLED - using raw config parameters")
#             return 1.0
        
#         # Get hotel box dimensions
#         if not hasattr(self.config, 'hotel_box') or self.config.hotel_box is None:
#             logger.warning("No hotel_box in config - using scale factor 1.0 (no scaling)")
#             logger.warning("For auto-scaling to work, set config.hotel_box = (x1, y1, x2, y2)")
#             return 1.0
        
#         x1, y1, x2, y2 = self.config.hotel_box.get_box_bounds(self.config.video.res_width, self.config.video.res_height)
#         hotel_width = x2 - x1
#         hotel_height = y2 - y1
        
#         # Reference dimensions (close-up camera view - what parameters are tuned for)
#         REFERENCE_FRAME_WIDTH = 1920
#         REFERENCE_FRAME_HEIGHT = 1080
#         REFERENCE_HOTEL_WIDTH = 800  # ~42% of frame width
#         REFERENCE_HOTEL_HEIGHT = 400  # ~37% of frame height (60-70% visually due to aspect)
        
#         # Get actual frame dimensions
#         frame_width = getattr(self.config.video, 'res_width', REFERENCE_FRAME_WIDTH)
#         frame_height = getattr(self.config.video, 'res_height', REFERENCE_FRAME_HEIGHT)
        
#         # Calculate scale factors in both dimensions
#         # Compare actual hotel size (as % of frame) to reference hotel size (as % of reference frame)
#         scale_height = (hotel_height / frame_height) / (REFERENCE_HOTEL_HEIGHT / REFERENCE_FRAME_HEIGHT)
#         scale_width = (hotel_width / frame_width) / (REFERENCE_HOTEL_WIDTH / REFERENCE_FRAME_WIDTH)
        
#         # Use average of both (balanced approach)
#         # Could use min for conservative (safer), max for aggressive (riskier)
#         scale_factor = (scale_height + scale_width) / 2
        
#         # Clamp to reasonable range (0.1x to 5.0x)
#         # Below 0.1x: hotel too small, parameters would be unreliable
#         # Above 5.0x: hotel too large, parameters would be excessive
#         scale_factor = max(0.1, min(5.0, scale_factor))
        
#         # Diagnostic logging
#         hotel_pct_width = (hotel_width / frame_width) * 100
#         hotel_pct_height = (hotel_height / frame_height) * 100
#         ref_pct_width = (REFERENCE_HOTEL_WIDTH / REFERENCE_FRAME_WIDTH) * 100
#         ref_pct_height = (REFERENCE_HOTEL_HEIGHT / REFERENCE_FRAME_HEIGHT) * 100
        
#         logger.info(f"=== AUTO-SCALING ANALYSIS ===")
#         logger.info(f"Frame: {int(frame_width)}x{int(frame_height)}")
#         logger.info(f"Hotel: {int(hotel_width)}x{int(hotel_height)}")
#         logger.info(f"Hotel coverage: {hotel_pct_width:.1f}% width, {hotel_pct_height:.1f}% height")
#         logger.info(f"Reference: {ref_pct_width:.1f}% width, {ref_pct_height:.1f}% height")
#         logger.info(f"Scale factors: width={scale_width:.2f}x, height={scale_height:.2f}x")
#         logger.info(f"Final scale factor: {scale_factor:.2f}x")
        
#         if scale_factor < 0.5:
#             logger.warning(f"Scale factor {scale_factor:.2f}x is quite small - hotel may be very far")
#             logger.warning("Consider using a tighter ROI or adjusting camera")
#         elif scale_factor > 2.0:
#             logger.warning(f"Scale factor {scale_factor:.2f}x is quite large - hotel may be very close")
#             logger.warning("Parameters scaled up significantly")
        
#         return scale_factor
    
#     def initialize_background_from_video(
#         self,
#         video_path: str,
#         site_roi: BBox,
#         res_height: int,
#         res_width: int,
#         max_frames: int = 200,
#         target_clean_frames: int = 50,
#         config: Optional[Config] = None
#     ) -> int:
#         """Initialize background model using only bee-free frames.
        
#         This method scans the video and builds the background model using only
#         frames where no bees (tracking classes) are detected by YOLO. This ensures
#         a clean background model without bee interference.
        
#         Args:
#             video_path: Path to video file
#             site_roi: Region of interest (x1, y1, x2, y2)
#             res_height: Target frame height
#             res_width: Target frame width
#             max_frames: Maximum frames to scan (default: 200)
#             target_clean_frames: Target number of clean frames to use (default: 50)
#             config: Optional config override
            
#         Returns:
#             Number of clean frames used to build background model
            
#         Example:
#             >>> tracker = HyDaTTracker(model, config)
#             >>> clean_frames = tracker.initialize_background_from_video(
#             ...     "video.mp4", site_roi, 1080, 1920
#             ... )
#             >>> print(f"Background model built from {clean_frames} bee-free frames")
#         """
#         if config is None:
#             config = self.config
        
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")
        
#         logger.info("Initializing background model from bee-free frames...")
#         logger.info(f"Scanning up to {max_frames} frames, target: {target_clean_frames} clean frames")
        
#         frame_num = 0
#         clean_frames_used = 0
#         frames_with_bees = 0
        
#         while frame_num < max_frames and clean_frames_used < target_clean_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Resize frame
#             frame = cv2.resize(frame, (res_width, res_height))
            
#             # Determine tracking region based on config
#             if getattr(config.tracking, 'track_full_frame', False):
#                 # Use entire frame for background initialization
#                 frame_roi = frame
#             else:
#                 # Use ROI only (default behavior)
#                 x1_roi, y1_roi, x2_roi, y2_roi = [int(c) for c in site_roi]
#                 frame_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
            
#             # Run YOLO detection to check for bees
#             results = self.model.predict(
#                 frame_roi,
#                 conf=config.tracking.confidence_threshold,
#                 iou=config.tracking.iou_threshold,
#                 verbose=False,
#                 device='0' if self.use_gpu else 'cpu'
#             )
            
#             # Check if any tracking class objects detected
#             has_bees = False
#             if len(results) > 0 and results[0].boxes is not None:
#                 for cls in results[0].boxes.cls:
#                     class_id = int(cls.cpu().numpy())
#                     if class_id in self.tracking_classes:
#                         has_bees = True
#                         frames_with_bees += 1
#                         break
            
#             # Only use frame if no bees detected
#             if not has_bees:
#                 # Add to background model (both regular and low-res)
#                 self.bg_subtractor.apply(frame_roi, learningRate=0.1)
                
#                 # Also add to low-res background model
#                 scale = config.tracking.low_res_scale_factor
#                 low_res_roi = cv2.resize(frame_roi, None, fx=scale, fy=scale)
#                 self.bg_subtractor_lowres.apply(low_res_roi, learningRate=0.1)
                
#                 clean_frames_used += 1
                
#                 if clean_frames_used % 10 == 0:
#                     logger.info(f"  Added {clean_frames_used}/{target_clean_frames} clean frames...")
            
#             frame_num += 1
        
#         cap.release()
        
#         logger.info(f"Background initialization complete:")
#         logger.info(f"  Scanned {frame_num} frames")
#         logger.info(f"  Clean frames used: {clean_frames_used}")
#         logger.info(f"  Frames with bees: {frames_with_bees}")
#         logger.info(f"  Success rate: {clean_frames_used/frame_num*100:.1f}%")
        
#         if clean_frames_used < 10:
#             logger.warning(f"Only {clean_frames_used} clean frames found! Background model may be poor.")
#             logger.warning("Consider using more frames or checking if bees are always present.")
        
#         return clean_frames_used
    
#     def _detect_gpu(self) -> bool:
#         """Detect if GPU is available."""
#         try:
#             import torch
#             if torch.cuda.is_available():
#                 logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
#                 return True
#         except ImportError:
#             pass
        
#         if cv2.cuda.getCudaEnabledDeviceCount() > 0:
#             logger.info("OpenCV CUDA support detected")
#             return True
        
#         logger.info("No GPU detected, using CPU")
#         return False
    
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
#         """Detect and track bees using HyDaT approach.
        
#         Args:
#             video_path: Path to video file
#             site_roi: Region of interest (x1, y1, x2, y2)
#             res_height: Target frame height
#             res_width: Target frame width
#             visualize: Whether to save visualization
#             output_folder: Output directory
#             config: Optional config override
#             initialize_background: Whether to initialize BG model from bee-free frames
            
#         Returns:
#             DataFrame with tracking results
#         """
#         if config is None:
#             config = self.config
        
#         # Initialize background model from bee-free frames
#         if initialize_background:
#             logger.info("Initializing background model from bee-free frames...")
#             clean_frames = self.initialize_background_from_video(
#                 video_path, site_roi, res_height, res_width, config=config
#             )
#             if clean_frames < 10:
#                 logger.warning("Background initialization may be insufficient!")
        
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
        
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")
        
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         logger.info(f"Processing {total_frames} frames with HyDaT tracking")
        
#         # Log tracking region mode
#         if getattr(config.tracking, 'track_full_frame', False):
#             logger.info("FULL FRAME tracking mode enabled")
#             logger.info(f"  - Motion detection: entire frame ({res_width}x{res_height})")
#             logger.info(f"  - Tracking region: entire frame ({res_width}x{res_height})")
#             logger.info(f"  - YOLO detection: entire frame")
#         else:
#             logger.info("HYBRID tracking mode (default)")
#             logger.info(f"  - Motion detection: ROI only {site_roi}")
#             logger.info(f"  - Tracking region: FULL FRAME (tracks can leave ROI)")
#             logger.info(f"  - YOLO detection: ROI + full-frame when tracks outside ROI")
        
#         # Initialize output
#         output_video = None
#         if visualize:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             output_path = os.path.join(output_folder, f"tracking_{os.path.basename(video_path).rsplit('.', 1)[0]}.mp4")
#             output_video = cv2.VideoWriter(output_path, fourcc, fps, (res_width, res_height))
        
#         # Processing state
#         frame_num = 0
#         all_detections = []
#         is_low_res = True
#         frames_without_activity = 0
        
#         # Statistics
#         low_res_frames = 0
#         high_res_frames = 0
#         yolo_calls = 0
#         fg_blobs_tracked = 0
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Resize frame
#             frame = cv2.resize(frame, (res_width, res_height))
            
#             # Determine motion detection region based on config
#             # NOTE: In default mode, motion detection is ROI-only but TRACKING extends beyond ROI
#             # DL will run on full frame when tracks predict outside ROI
#             if getattr(config.tracking, 'track_full_frame', False):
#                 # Full frame mode: motion detection on entire frame
#                 frame_roi = frame
#                 x1_roi, y1_roi = 0, 0
#                 x2_roi, y2_roi = res_width, res_height
#             else:
#                 # Hybrid mode (default): motion detection in ROI, tracking beyond ROI
#                 x1_roi, y1_roi, x2_roi, y2_roi = [int(c) for c in site_roi]
#                 frame_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
            
#             # Decide processing mode
#             if is_low_res and config.tracking.enable_low_res_mode:
#                 # Low-resolution check
#                 if frame_num % config.tracking.low_res_check_interval == 0:
#                     fg_detected = self._check_fg_activity_lowres(frame_roi, config)
#                     low_res_frames += 1
                    
#                     if fg_detected:
#                         is_low_res = False
#                         logger.info(f"Frame {frame_num}: Switching to HIGH-RES mode")
#                 else:
#                     low_res_frames += 1
#                     frame_num += 1
#                     continue
            
#             if not is_low_res:
#                 # High-resolution processing with HyDaT
#                 high_res_frames += 1
                
#                 # Step 1: Run FG/BG segmentation
#                 fg_mask = self.bg_subtractor.apply(frame_roi)
#                 fg_mask[fg_mask == 127] = 0  # Remove shadows
                
#                 # Step 2: Extract FG blobs (filtered by aspect ratio to avoid YOLO on noise)
#                 blobs = self._extract_blobs(fg_mask, config, x1_roi, y1_roi)
                
#                 # Step 3: Predict track positions (always, even without blobs)
#                 predictions = self._predict_tracks(frame_num)
                
#                 # Check if any tracks predict outside ROI (need full-frame DL)
#                 tracks_outside_roi = self._check_tracks_outside_roi(
#                     predictions, (x1_roi, y1_roi, x2_roi, y2_roi)
#                 )
                
#                 # Initialize frame debug data
#                 frame_debug_data = {
#                     'blobs': blobs,
#                     'yolo_detections': []
#                 }
                
#                 # Determine if we have activity
#                 has_blobs = len(blobs) > 0
#                 has_active_tracks = len(predictions) > 0
                
#                 if has_blobs or has_active_tracks:
#                     # Activity detected (either blobs OR existing tracks)
#                     frames_without_activity = 0
                    
#                     if has_blobs:
#                         fg_blobs_tracked += len(blobs)
                        
#                         # Step 4: Associate blobs to predictions
#                         matched_blobs, matched_tracks, unmatched_blobs, unmatched_tracks = \
#                             self._associate_blobs_to_tracks(blobs, predictions)
#                     else:
#                         # No blobs but we have active tracks (fast turn scenario!)
#                         # All tracks are unmatched
#                         matched_blobs = []
#                         matched_tracks = []
#                         unmatched_blobs = []
#                         unmatched_tracks = list(predictions.keys())
#                         logger.debug(f"Frame {frame_num}: No blobs detected but {len(unmatched_tracks)} active tracks - running YOLO")
                    
#                     # Step 5: Run YOLO when needed
#                     # CRITICAL: Run YOLO if we have unmatched tracks even without blobs
#                     # This handles fast turns where FG detection fails
#                     yolo_needed = len(unmatched_blobs) > 0 or len(unmatched_tracks) > 0
                    
#                     # ADDITIONAL: Force YOLO check for tracks without recent confirmation
#                     # This catches tracks that may be drifting or need verification
#                     if not yolo_needed and self.config.tracking.max_frames_without_yolo > 0:
#                         stale_tracks = []
#                         for track_id, track in self.tracks.items():
#                             frames_since_yolo = frame_num - track.last_yolo_confirmation
#                             if frames_since_yolo > self.config.tracking.max_frames_without_yolo:
#                                 stale_tracks.append(track_id)
                        
#                         if stale_tracks:
#                             yolo_needed = True
#                             logger.debug(f"Frame {frame_num}: Forcing YOLO check for {len(stale_tracks)} stale tracks")
                    
#                     # NEW: Also run YOLO if tracks predict outside ROI
#                     if tracks_outside_roi:
#                         yolo_needed = True
#                         logger.debug(f"Frame {frame_num}: Tracks outside ROI - running full-frame YOLO")
                    
#                     if yolo_needed:
#                         yolo_calls += 1
                        
#                         # Determine YOLO region based on track positions
#                         if tracks_outside_roi:
#                             # Run YOLO on FULL FRAME (tracks outside ROI)
#                             yolo_detections = self._run_yolo(
#                                 frame, config, 0, 0  # Full frame, no offset
#                             )
#                             logger.debug(f"Frame {frame_num}: Running full-frame YOLO for outside-ROI tracks")
#                         else:
#                             # Run YOLO on ROI only (normal HyDaT)
#                             yolo_detections = self._run_yolo(
#                                 frame_roi, config, x1_roi, y1_roi
#                             )
                        
#                         frame_debug_data['yolo_detections'] = yolo_detections
                        
#                         # Update tracks with YOLO confirmations
#                         self._update_tracks_with_yolo(
#                             yolo_detections, unmatched_blobs, unmatched_tracks, frame_num
#                         )
                    
#                     # Step 6: Update matched tracks with blob positions
#                     if matched_blobs:
#                         self._update_matched_tracks(matched_blobs, matched_tracks, frame_num)
                    
#                     # Record detections with debug data
#                     detections = self._get_current_detections(frame_num, frame_debug_data)
#                     logger.debug(f"Frame {frame_num}: Recording {len(detections)} detections, total so far: {len(all_detections) + len(detections)}")
#                     all_detections.extend(detections)
                    
#                     # Visualization during active tracking
#                     if visualize and output_video:
#                         viz_frame = self._visualize_frame(
#                             frame, frame_num, is_low_res,
#                             blobs=frame_debug_data['blobs'],
#                             yolo_detections=frame_debug_data['yolo_detections']
#                         )
#                         output_video.write(viz_frame)
#                 else:
#                     frames_without_activity += 1
                    
#                     # Age out tracks without detections
#                     self._age_tracks(frame_num)
                
#                 # Switch back to low-res if no activity
#                 if frames_without_activity > 30 and len(self.tracks) == 0:
#                     is_low_res = True
#                     logger.info(f"Frame {frame_num}: Switching to LOW-RES mode")
#             else:
#                 # Low-res mode - skip visualization (no activity to show)
#                 pass
            
#             frame_num += 1
            
#             if frame_num % 100 == 0:
#                 logger.info(f"Processed {frame_num}/{total_frames} frames")
        
#         cap.release()
#         if output_video:
#             output_video.release()
        
#         # Log statistics
#         logger.info(f"Processing complete:")
#         logger.info(f"  Low-res frames: {low_res_frames}")
#         logger.info(f"  High-res frames: {high_res_frames}")
#         logger.info(f"  YOLO calls: {yolo_calls}")
#         logger.info(f"  FG blobs tracked: {fg_blobs_tracked}")
#         logger.info(f"  YOLO reduction: {(1 - yolo_calls/max(high_res_frames, 1))*100:.1f}%")
#         logger.info(f"  Total detections collected: {len(all_detections)}")
        
#         # Convert to grouped track format for event processor
#         logger.info("Converting detections to grouped format...")
#         grouped_tracks = self._convert_to_grouped_format(all_detections)
#         logger.info(f"  Grouped format: {len(grouped_tracks)} periods")
        
#         return grouped_tracks
    
#     def _convert_to_grouped_format(self, all_detections: List[Dict]) -> pd.DataFrame:
#         """Convert flat detection list to grouped track format expected by event processor.
        
#         Args:
#             all_detections: List of detection dictionaries
            
#         Returns:
#             DataFrame with columns: frame_number (tuple), tracks (list), detections (dict)
#         """
#         logger.debug(f"Converting {len(all_detections)} detections to grouped format")
        
#         if not all_detections:
#             logger.warning("No detections to convert - returning empty DataFrame")
#             return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
#         try:
#             # Extract debug data before creating DataFrame (lists cause issues)
#             debug_data_by_frame = {}
#             for det in all_detections:
#                 frame_num = det['frame']
#                 if 'debug_blobs' in det and frame_num not in debug_data_by_frame:
#                     debug_data_by_frame[frame_num] = {
#                         'blobs': det.get('debug_blobs', []),
#                         'yolo': det.get('debug_yolo', [])
#                     }
            
#             # Remove debug data from detections before DataFrame conversion
#             clean_detections = []
#             for det in all_detections:
#                 clean_det = {k: v for k, v in det.items() if k not in ['debug_blobs', 'debug_yolo']}
#                 clean_detections.append(clean_det)
            
#             logger.debug(f"Creating DataFrame from {len(clean_detections)} clean detections")
            
#             # Convert to DataFrame for easier grouping
#             df = pd.DataFrame(clean_detections)
            
#             logger.debug(f"DataFrame created: {len(df)} rows, columns: {list(df.columns)}")
            
#         except Exception as e:
#             logger.error(f"Error creating DataFrame from detections: {e}")
#             logger.error(f"First detection: {all_detections[0] if all_detections else 'None'}")
#             raise
        
#         # Split into activity periods based on temporal gaps
#         # Use slightly longer threshold than max_age for period splitting
#         periods = self._split_into_periods(df, gap_threshold=int(self.config.tracking.max_age * 1.1))
        
#         logger.debug(f"Split into {len(periods)} activity periods")
        
#         result_rows = []
#         for period_df in periods:
#             # Group by track_id within this period
#             track_groups = {}
            
#             for track_id in period_df['track_id'].unique():
#                 track_df = period_df[period_df['track_id'] == track_id].sort_values('frame')
                
#                 # Split track by gaps - creates sub-tracks if large gaps exist
#                 # Use max_age as threshold to match tracking behavior
#                 track_segments = self._split_track_by_gaps(track_df, gap_threshold=self.config.tracking.max_age)
                
#                 for segment_idx, segment_df in enumerate(track_segments):
#                     # Assign unique ID for each segment
#                     unique_id = f"{track_id}_{segment_idx}" if len(track_segments) > 1 else track_id
                    
#                     centroids = [
#                         ((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
#                         for _, row in segment_df.iterrows()
#                     ]
#                     bboxes = [
#                         (row['x1'], row['y1'], row['x2'], row['y2'])
#                         for _, row in segment_df.iterrows()
#                     ]
#                     frame_numbers = segment_df['frame'].tolist()
                    
#                     # Filter out very short tracks (likely noise)
#                     if len(frame_numbers) >= self.config.tracking.min_track_length:
#                         track_groups[unique_id] = (unique_id, centroids, bboxes, frame_numbers)
            
#             if not track_groups:
#                 continue
            
#             # Create row for this period
#             all_tracks = list(track_groups.values())
#             min_frame = period_df['frame'].min()
#             max_frame = period_df['frame'].max()
            
#             # Build detections dict
#             frame_detections = {}
#             for frame_num in period_df['frame'].unique():
#                 frame_df = period_df[period_df['frame'] == frame_num]
                
#                 # Get debug data from extracted dict (if available)
#                 frame_debug = debug_data_by_frame.get(int(frame_num), {'blobs': [], 'yolo': []})
                
#                 frame_detections[int(frame_num)] = {
#                     'boxes': [
#                         (row['x1'], row['y1'], row['x2'], row['y2'])
#                         for _, row in frame_df.iterrows()
#                     ],
#                     'label': frame_df['species'].tolist(),
#                     'debug_blobs': frame_debug['blobs'],
#                     'debug_yolo': frame_debug['yolo']
#                 }
            
#             result_rows.append({
#                 'frame_number': (int(min_frame), int(max_frame)),
#                 'tracks': all_tracks,
#                 'detections': frame_detections
#             })
        
#         logger.debug(f"Created {len(result_rows)} result rows")
#         result_df = pd.DataFrame(result_rows) if result_rows else pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
#         logger.debug(f"Returning DataFrame with {len(result_df)} rows")
#         return result_df
    
#     def _split_into_periods(self, df: pd.DataFrame, gap_threshold: int = 100) -> List[pd.DataFrame]:
#         """Split detections into activity periods based on temporal gaps.
        
#         Args:
#             df: DataFrame of detections
#             gap_threshold: Gap size (frames) to split periods
            
#         Returns:
#             List of DataFrames, one per period
#         """
#         df = df.sort_values('frame')
#         frames = df['frame'].tolist()
        
#         periods = []
#         current_period_start = 0
        
#         for i in range(len(frames) - 1):
#             gap = frames[i + 1] - frames[i]
#             if gap > gap_threshold:
#                 # End current period, start new one
#                 periods.append(df.iloc[current_period_start:i+1].copy())
#                 current_period_start = i + 1
        
#         # Add final period
#         if current_period_start < len(df):
#             periods.append(df.iloc[current_period_start:].copy())
        
#         return periods
    
#     def _split_track_by_gaps(
#         self, 
#         track_df: pd.DataFrame, 
#         gap_threshold: int = 30
#     ) -> List[pd.DataFrame]:
#         """Split a track into segments if it has large gaps.
        
#         Args:
#             track_df: DataFrame for a single track
#             gap_threshold: Gap size (frames) to split track
            
#         Returns:
#             List of DataFrames, one per track segment
#         """
#         frames = track_df['frame'].tolist()
        
#         segments = []
#         current_segment_start = 0
        
#         for i in range(len(frames) - 1):
#             gap = frames[i + 1] - frames[i]
#             if gap > gap_threshold:
#                 # End current segment, start new one
#                 segments.append(track_df.iloc[current_segment_start:i+1].copy())
#                 current_segment_start = i + 1
        
#         # Add final segment
#         if current_segment_start < len(track_df):
#             segments.append(track_df.iloc[current_segment_start:].copy())
        
#         return segments
    
#     def _check_fg_activity_lowres(
#         self,
#         frame_roi: np.ndarray,
#         config: Config
#     ) -> bool:
#         """Check for FG activity in low-resolution mode.
        
#         Args:
#             frame_roi: ROI portion of frame
#             config: Configuration
            
#         Returns:
#             True if activity detected
#         """
#         # Scale down
#         scale = config.tracking.low_res_scale_factor
#         low_res_roi = cv2.resize(frame_roi, None, fx=scale, fy=scale)
        
#         # Apply background subtraction
#         fg_mask = self.bg_subtractor_lowres.apply(low_res_roi)
#         fg_mask[fg_mask == 127] = 0
        
#         # Calculate area
#         if config.tracking.fg_area_method == "pixels":
#             fg_area = np.count_nonzero(fg_mask)
#         else:
#             contours, _ = cv2.findContours(
#                 fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#             )
#             fg_area = sum(cv2.contourArea(c) for c in contours)
        
#         # Check threshold (scaled)
#         min_area = config.tracking.min_fg_area(
#             frame_roi.shape[1], frame_roi.shape[0], config.hotel_box
#         ) * (scale ** 2)
        
#         return fg_area > min_area
    
#     def _extract_blobs(
#         self,
#         fg_mask: np.ndarray,
#         config: Config,
#         x_offset: int,
#         y_offset: int
#     ) -> List[Dict]:
#         """Extract foreground blobs from mask with robust multi-stage filtering.
        
#         Filtering stages:
#         1. Morphological cleanup (remove noise, fill gaps)
#         2. Aspect ratio (bee-like shape, not lines/artifacts)
#         3. Solidity (fill ratio - removes hollow/irregular noise)
#         4. Distance to hotel (removes far-away noise)
        
#         Args:
#             fg_mask: Binary foreground mask
#             config: Configuration
#             x_offset: X offset for ROI
#             y_offset: Y offset for ROI
            
#         Returns:
#             List of blob dictionaries with bbox, centroid, area, solidity
#         """
#         # Stage 1: Morphological cleanup to remove small noise
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
#         fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
        
#         contours, _ = cv2.findContours(
#             fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
        
#         blobs = []
#         min_area = config.tracking.min_contour_area(
#             fg_mask.shape[1], fg_mask.shape[0], config.hotel_box
#         )
        
#         # Get hotel center for distance filtering
#         hotel_center = self._get_hotel_center()
        
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area < min_area:
#                 continue
            
#             # Get bounding box
#             x, y, w, h = cv2.boundingRect(contour)
#             bbox = (x + x_offset, y + y_offset, x + w + x_offset, y + h + y_offset)
            
#             # Stage 2: Filter by aspect ratio (bee-like shapes only)
#             if not self._is_valid_blob_shape(bbox):
#                 continue
            
#             # Stage 3: Solidity check (contour area / bounding box area)
#             # Real bees: 0.5-0.9, Noise: often < 0.3
#             bbox_area = w * h
#             solidity = area / bbox_area if bbox_area > 0 else 0
#             min_solidity = getattr(config.tracking, 'min_blob_solidity', 0.4)
#             if solidity < min_solidity:
#                 continue
            
#             # Calculate centroid
#             M = cv2.moments(contour)
#             if M["m00"] > 0:
#                 cx = M["m10"] / M["m00"]
#                 cy = M["m01"] / M["m00"]
#             else:
#                 cx = x + w / 2
#                 cy = y + h / 2
            
#             centroid = (cx + x_offset, cy + y_offset)
            
#             # Stage 4: Distance filtering - reject blobs far from hotel
#             if hotel_center is not None:
#                 dist_to_hotel = np.sqrt(
#                     (centroid[0] - hotel_center[0])**2 + 
#                     (centroid[1] - hotel_center[1])**2
#                 )
#                 # Calculate max allowed distance (hotel diagonal + margin)
#                 # SCALED by tracker's scale factor
#                 base_max_distance = getattr(config.tracking, 'max_blob_distance_from_hotel', float('inf'))
#                 if base_max_distance != float('inf'):
#                     # Scale distance threshold based on hotel size
#                     scaled_max_distance = base_max_distance * self.scale_factor
#                     if dist_to_hotel > scaled_max_distance:
#                         continue
            
#             blob = {
#                 'bbox': bbox,
#                 'centroid': centroid,
#                 'area': area,
#                 'solidity': solidity
#             }
#             blobs.append(blob)
        
#         return blobs
    
#     def _get_hotel_center(self) -> Optional[Tuple[float, float]]:
#         """Get center point of hotel box for distance filtering.
        
#         Returns:
#             (x, y) tuple of hotel center, or None if no hotel box
#         """
#         if not hasattr(self.config, 'hotel_box') or self.config.hotel_box is None:
#             return None
        
#         x1, y1, x2, y2 = self.config.hotel_box.get_box_bounds(self.config.video.res_width, self.config.video.res_height)
#         return ((x1 + x2) / 2, (y1 + y2) / 2)
    
#     def _check_tracks_outside_roi(
#         self,
#         predictions: Dict[int, Dict],
#         roi: Tuple[int, int, int, int]
#     ) -> bool:
#         """Check if any predicted track positions are outside ROI.
        
#         This determines if we need to run DL on full frame to track bees
#         that have moved beyond the ROI boundaries.
        
#         Args:
#             predictions: Dictionary of track predictions
#             roi: ROI boundaries (x1, y1, x2, y2)
            
#         Returns:
#             True if any tracks predict outside ROI
#         """
#         if not predictions:
#             return False
        
#         x1_roi, y1_roi, x2_roi, y2_roi = roi
        
#         for track_id, pred in predictions.items():
#             cx, cy = pred['centroid']
            
#             # Check if predicted position is outside ROI (with small margin)
#             margin = 20  # pixels margin to catch near-boundary tracks
#             if (cx < x1_roi - margin or cx > x2_roi + margin or
#                 cy < y1_roi - margin or cy > y2_roi + margin):
#                 return True
        
#         return False
    
#     def _is_valid_blob_shape(self, bbox: Tuple[float, float, float, float]) -> bool:
#         """Check if blob has bee-like aspect ratio and size.
        
#         Args:
#             bbox: Bounding box (x1, y1, x2, y2)
            
#         Returns:
#             True if aspect ratio and size are within valid range
#         """
#         x1, y1, x2, y2 = bbox
#         width = x2 - x1
#         height = y2 - y1
        
#         if height == 0 or width == 0:
#             return False
        
#         # Check aspect ratio (bee-like shape)
#         aspect_ratio = width / height
#         if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
#             return False
        
#         # Check area (reasonable bee size)
#         area = width * height
#         if not (self.min_blob_area_pixels <= area <= self.max_blob_area_pixels):
#             return False
        
#         return True
    
#     def _predict_tracks(self, frame_num: int) -> Dict[int, Dict]:
#         """Predict positions of active tracks using Kalman filter.
        
#         When tracks haven't been updated by DL recently, apply velocity damping
#         to prevent unrealistic fast-moving predictions during direction changes.
        
#         Args:
#             frame_num: Current frame number
            
#         Returns:
#             Dictionary of track_id -> prediction
#         """
#         predictions = {}
        
#         for track_id, track in self.tracks.items():
#             # Check if track was recently confirmed by DL
#             frames_since_dl = frame_num - track.last_yolo_confirmation
            
#             # Apply velocity damping if no recent DL confirmation
#             # This prevents predictions from continuing at full velocity during direction changes
#             if frames_since_dl > 0:
#                 # Damping factor increases with time since DL confirmation
#                 # 0 frames: no damping (1.0), 5 frames: 50% damping (0.5), 10+ frames: 80% damping (0.2)
#                 damping_factor = max(0.2, 1.0 - (frames_since_dl * 0.1))
                
#                 # Apply damping to velocity components in Kalman state
#                 track.kalman.statePost[2] *= damping_factor  # vx
#                 track.kalman.statePost[3] *= damping_factor  # vy
            
#             # Predict next position
#             prediction = track.kalman.predict()
#             pred_x, pred_y = prediction[0], prediction[1]
            
#             # Estimate bbox around prediction (use last bbox size)
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

#     def _associate_blobs_to_tracks(
#         self,
#         blobs: List[Dict],
#         predictions: Dict[int, Dict]
#     ) -> Tuple[List, List, List, List]:
#         """Associate detected blobs to predicted track positions with strict safety checks.
        
#         Safety constraints prevent unrealistic associations:
#         - Hard distance limit (prevents cross-hotel jumps)
#         - Velocity limit (bees don't teleport)
#         - Cost threshold (basic association quality)
        
#         Args:
#             blobs: List of detected blobs
#             predictions: Dictionary of predicted track positions
            
#         Returns:
#             Tuple of (matched_blobs, matched_tracks, unmatched_blobs, unmatched_tracks)
#         """
#         if not blobs or not predictions:
#             return [], [], blobs, list(predictions.keys())
        
#         # Build adaptive cost matrix
#         blob_centroids = [b['centroid'] for b in blobs]
#         track_ids = list(predictions.keys())
        
#         cost_matrix = np.zeros((len(blobs), len(track_ids)))
        
#         # Base threshold (scaled by tracker's scale factor)
#         base_threshold = self.config.tracking.association_threshold(
#             1920, 1080, self.config.hotel_box
#         ) * self.scale_factor
        
#         for i, blob in enumerate(blobs):
#             blob_cent = blob['centroid']
#             blob_area = blob.get('area', 0)
            
#             for j, track_id in enumerate(track_ids):
#                 pred = predictions[track_id]
#                 track = pred['track']
#                 pred_cent = pred['centroid']
                
#                 # Get velocity from Kalman state [x, y, vx, vy]
#                 vx = float(track.kalman.statePost[2])
#                 vy = float(track.kalman.statePost[3])
#                 speed = np.sqrt(vx**2 + vy**2)
                
#                 # Calculate position difference
#                 dx = blob_cent[0] - pred_cent[0]
#                 dy = blob_cent[1] - pred_cent[1]
#                 euclidean_dist = np.sqrt(dx**2 + dy**2)
                
#                 # Compute adaptive directional cost
#                 cost = self._compute_adaptive_cost(
#                     dx, dy, vx, vy, speed, euclidean_dist, 
#                     blob_area, track, base_threshold
#                 )
                
#                 cost_matrix[i, j] = cost
        
#         # Hungarian algorithm
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
#         matched_blobs = []
#         matched_tracks = []
#         matched_indices = set()
        
#         # STRICT FILTERING - prevent unrealistic associations
#         # These are HARD limits that override cost-based matching
#         MAX_DISTANCE = getattr(self.config.tracking, 'max_association_distance', 200) * self.scale_factor
#         MAX_VELOCITY = getattr(self.config.tracking, 'max_bee_velocity', 50) * self.scale_factor
        
#         logger.debug(f"Association safety limits: max_dist={MAX_DISTANCE:.1f}px, max_vel={MAX_VELOCITY:.1f}px/frame")
        
#         # Filter matches by safety constraints
#         for i, j in zip(row_ind, col_ind):
#             cost = cost_matrix[i, j]
            
#             # Check 1: Cost threshold
#             if cost >= base_threshold:
#                 continue  # Cost too high
            
#             blob = blobs[i]
#             track_id = track_ids[j]
#             track = predictions[track_id]['track']
            
#             # Check 2: HARD DISTANCE LIMIT (prevents cross-hotel jumps)
#             blob_cent = blob['centroid']
#             track_cent = track.centroid  # Use ACTUAL position, not predicted
#             distance = np.sqrt(
#                 (blob_cent[0] - track_cent[0])**2 + 
#                 (blob_cent[1] - track_cent[1])**2
#             )
            
#             if distance > MAX_DISTANCE:
#                 logger.debug(f"REJECTED blob-track match: distance {distance:.1f} > {MAX_DISTANCE:.1f}px (track {track_id})")
#                 continue
            
#             # Check 3: VELOCITY SANITY CHECK (bees don't teleport)
#             # Implied velocity = distance traveled in 1 frame
#             implied_velocity = distance  # pixels/frame (assuming 1 frame delta)
            
#             if implied_velocity > MAX_VELOCITY:
#                 logger.debug(f"REJECTED blob-track match: velocity {implied_velocity:.1f} > {MAX_VELOCITY:.1f}px/frame (track {track_id})")
#                 continue
            
#             # Optional Check 4: ACCELERATION CHECK (prevent sudden speed changes)
#             # Current velocity from Kalman
#             current_speed = np.sqrt(
#                 float(track.kalman.statePost[2])**2 + 
#                 float(track.kalman.statePost[3])**2
#             )
            
#             # If current speed is low and implied velocity is high = suspicious
#             MAX_ACCELERATION = getattr(self.config.tracking, 'max_bee_acceleration', 30) * self.scale_factor
#             speed_change = abs(implied_velocity - current_speed)
            
#             if speed_change > MAX_ACCELERATION:
#                 logger.debug(f"REJECTED blob-track match: acceleration {speed_change:.1f} > {MAX_ACCELERATION:.1f}px/frame² (track {track_id})")
#                 continue
            
#             # All checks passed - valid match
#             logger.debug(f"ACCEPTED blob-track match: dist={distance:.1f}px, vel={implied_velocity:.1f}px/frame, cost={cost:.1f} (track {track_id})")
#             matched_blobs.append(blob)
#             matched_tracks.append(track_id)
#             matched_indices.add(i)
        
#         # Unmatched
#         unmatched_blobs = [b for i, b in enumerate(blobs) if i not in matched_indices]
#         matched_track_ids = set(matched_tracks)
#         unmatched_tracks = [tid for tid in track_ids if tid not in matched_track_ids]
        
#         # Log summary
#         logger.debug(f"Association: {len(matched_blobs)} matched, {len(unmatched_blobs)} unmatched blobs, {len(unmatched_tracks)} unmatched tracks")
        
#         return matched_blobs, matched_tracks, unmatched_blobs, unmatched_tracks
    
#     def _compute_adaptive_cost(
#         self,
#         dx: float,
#         dy: float,
#         vx: float,
#         vy: float,
#         speed: float,
#         euclidean_dist: float,
#         blob_area: float,
#         track: 'TrackState',
#         base_threshold: float
#     ) -> float:
#         """Compute adaptive directional cost for blob-to-track association.
        
#         Creates an elliptical search area based on velocity:
#         - Major axis along velocity direction (allows forward motion)
#         - Minor axis perpendicular (allows lateral drift)
#         - Extended reverse search (catches 180° turns)
#         - Speed-adaptive threshold (faster = larger search)
        
#         Args:
#             dx, dy: Position difference (blob - prediction)
#             vx, vy: Velocity components
#             speed: Magnitude of velocity
#             euclidean_dist: Simple Euclidean distance
#             blob_area: Area of the blob
#             track: Track state object
#             base_threshold: Base association threshold
            
#         Returns:
#             Weighted cost (lower = better match)
#         """
#         # Check if adaptive association is enabled
#         if not self.config.tracking.adaptive_association:
#             return euclidean_dist
        
#         # Speed threshold for "moving" vs "stationary"
#         speed_threshold = self.config.tracking.stationary_speed_threshold
        
#         if speed < speed_threshold:
#             # Stationary or slow-moving: use circular search
#             return euclidean_dist
        
#         # Normalize velocity direction
#         vx_norm = vx / speed
#         vy_norm = vy / speed
        
#         # Decompose position difference into parallel and perpendicular components
#         # Parallel: projection onto velocity direction
#         parallel = dx * vx_norm + dy * vy_norm
        
#         # Perpendicular: orthogonal distance
#         perpendicular = abs(dx * vy_norm - dy * vx_norm)
        
#         # Adaptive threshold based on speed
#         # Faster motion → larger forward search area
#         speed_factor = min(speed / 50.0, self.config.tracking.max_speed_factor)
        
#         # Elliptical search parameters from config
#         forward_threshold = base_threshold * (1.0 + speed_factor)  # Major axis (forward)
#         lateral_threshold = base_threshold * self.config.tracking.lateral_search_ratio  # Minor axis
#         reverse_threshold = base_threshold * self.config.tracking.reverse_search_ratio  # Reverse
        
#         # Check direction of displacement relative to velocity
#         if parallel > 0:
#             # Forward direction (expected)
#             # Use elliptical distance: (parallel/a)² + (perpendicular/b)²
#             normalized_dist = np.sqrt(
#                 (parallel / forward_threshold) ** 2 +
#                 (perpendicular / lateral_threshold) ** 2
#             )
#             cost = normalized_dist * base_threshold
            
#         elif parallel < 0:
#             # Reverse direction (180° turn)
#             # Use reverse cone - more lenient than forward but not as much
#             reverse_parallel = abs(parallel)
#             normalized_dist = np.sqrt(
#                 (reverse_parallel / reverse_threshold) ** 2 +
#                 (perpendicular / lateral_threshold) ** 2
#             )
#             # Add penalty for reverse motion (prefer forward matches)
#             cost = normalized_dist * base_threshold * self.config.tracking.reverse_motion_penalty
            
#         else:
#             # Pure lateral motion
#             cost = perpendicular
        
#         # Feature similarity bonus (if available)
#         if self.config.tracking.area_similarity_weight > 0 and hasattr(track, 'bbox') and blob_area > 0:
#             track_area = (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])
#             if track_area > 0:
#                 area_ratio = min(blob_area, track_area) / max(blob_area, track_area)
#                 # Apply area similarity bonus (reduce cost)
#                 if area_ratio > 0.5:  # Similar size
#                     similarity_bonus = 1.0 - (self.config.tracking.area_similarity_weight * (area_ratio - 0.5) * 2)
#                     cost *= similarity_bonus
        
#         return cost
    
#     def _run_yolo(
#         self,
#         frame_roi: np.ndarray,
#         config: Config,
#         x_offset: int,
#         y_offset: int
#     ) -> List[Dict]:
#         """Run YOLO detection on frame.
        
#         Args:
#             frame_roi: ROI portion of frame
#             config: Configuration
#             x_offset: X offset for ROI
#             y_offset: Y offset for ROI
            
#         Returns:
#             List of YOLO detections
#         """
#         results = self.model.predict(
#             frame_roi,
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
                
#                 detection = {
#                     'bbox': (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset),
#                     'centroid': (cx + x_offset, cy + y_offset),
#                     'class_id': class_id,
#                     'species': self.label_map.get(class_id, f'class_{class_id}')
#                 }
#                 detections.append(detection)
        
#         return detections


#     def _update_tracks_with_yolo(
#         self,
#         yolo_detections: List[Dict],
#         unmatched_blobs: List[Dict],
#         unmatched_tracks: List[int],
#         frame_num: int
#     ):
#         """Update tracks with YOLO confirmations with strict safety checks.
        
#         Three-stage process:
#         1. Match unmatched blobs to YOLO detections → create new tracks
#         2. Match unmatched tracks to YOLO detections → update existing tracks
#         3. Resurrect struggling tracks OR create new tracks for remaining YOLO detections
        
#         Safety checks prevent:
#         - Cross-hotel track jumps (distance + velocity limits)
#         - Track fragmentation (resurrection before new track creation)
#         - Unrealistic teleportation (velocity constraints)
        
#         Args:
#             yolo_detections: YOLO detection results
#             unmatched_blobs: Blobs without track association
#             unmatched_tracks: Tracks without blob association
#             frame_num: Current frame number
#         """
#         # Track which YOLO detections have been used to prevent double-assignment
#         used_yolo_detections = set()
        
#         # =================================================================
#         # CASE 1: For unmatched blobs, try to match with YOLO detections
#         # =================================================================
#         for blob in unmatched_blobs:
#             # Only consider blobs with valid aspect ratio
#             if not self._is_valid_blob_shape(blob['bbox']):
#                 continue
            
#             best_match = None
#             best_dist = float('inf')
#             best_idx = None
            
#             # Find closest unused YOLO detection
#             for idx, det in enumerate(yolo_detections):
#                 if idx in used_yolo_detections:
#                     continue
                
#                 dist = np.sqrt(
#                     (blob['centroid'][0] - det['centroid'][0]) ** 2 +
#                     (blob['centroid'][1] - det['centroid'][1]) ** 2
#                 )
                
#                 if dist < best_dist:
#                     best_dist = dist
#                     best_match = det
#                     best_idx = idx
            
#             # STRICT validation for NEW track creation (using scaled thresholds)
#             if best_match and best_dist < self.new_track_distance_threshold:
#                 # Check if there's already a track nearby (prevent duplicates)
#                 has_nearby_track = False
#                 for track_id, track in self.tracks.items():
#                     track_dist = np.sqrt(
#                         (track.centroid[0] - best_match['centroid'][0]) ** 2 +
#                         (track.centroid[1] - best_match['centroid'][1]) ** 2
#                     )
#                     if track_dist < self.new_track_proximity_check:
#                         has_nearby_track = True
#                         logger.debug(f"Frame {frame_num}: Blob+YOLO match skipped - track {track_id} nearby ({track_dist:.1f}px)")
#                         break
                
#                 # Only create new track if no nearby tracks exist
#                 if not has_nearby_track:
#                     logger.debug(f"Frame {frame_num}: Creating new track (blob+YOLO) at {best_match['centroid']}")
#                     self._create_track(best_match, frame_num)
#                     used_yolo_detections.add(best_idx)
#                 else:
#                     logger.debug(f"Frame {frame_num}: Skipping new track creation - nearby track exists")
        
#         # =================================================================
#         # CASE 2: For unmatched tracks, try to find with YOLO
#         # =================================================================
#         # CRITICAL: This handles fast turns when FG detection fails
#         for track_id in unmatched_tracks:
#             track = self.tracks[track_id]
#             best_match = None
#             best_cost = float('inf')
#             best_idx = None
            
#             # Use adaptive association if enabled
#             track_centroid = track.centroid
            
#             # Get velocity from track
#             vx = float(track.kalman.statePost[2])
#             vy = float(track.kalman.statePost[3])
#             speed = np.sqrt(vx**2 + vy**2)
            
#             # Find best unused YOLO detection for this track
#             for idx, det in enumerate(yolo_detections):
#                 if idx in used_yolo_detections:
#                     continue
                
#                 # Calculate position difference
#                 dx = det['centroid'][0] - track_centroid[0]
#                 dy = det['centroid'][1] - track_centroid[1]
#                 euclidean_dist = np.sqrt(dx**2 + dy**2)
                
#                 # Use adaptive cost computation
#                 cost = self._compute_adaptive_cost(
#                     dx, dy, vx, vy, speed, euclidean_dist,
#                     0, track, self.config.tracking.association_threshold_base
#                 )
                
#                 if cost < best_cost:
#                     best_cost = cost
#                     best_match = det
#                     best_idx = idx
            
#             # Use association threshold (scaled by tracker's scale factor)
#             max_threshold = self.config.tracking.association_threshold(
#                 1920, 1080, self.config.hotel_box
#             ) * self.scale_factor
            
#             if best_match and best_cost < max_threshold:
#                 # Update track with YOLO confirmation
#                 track.bbox = best_match['bbox']
#                 track.centroid = best_match['centroid']
#                 track.species = best_match['species']
#                 track.frames_without_detection = 0
#                 track.last_yolo_confirmation = frame_num
                
#                 # Add to trajectory history (keep last 30 positions)
#                 track.trajectory_history.append((frame_num, best_match['centroid']))
#                 if len(track.trajectory_history) > 30:
#                     track.trajectory_history.pop(0)
                
#                 # Update Kalman filter
#                 measurement = np.array([[best_match['centroid'][0]], [best_match['centroid'][1]]], dtype=np.float32)
#                 track.kalman.correct(measurement)
                
#                 used_yolo_detections.add(best_idx)
#                 logger.debug(f"Frame {frame_num}: YOLO confirmed track {track_id} at {best_match['centroid']} (cost={best_cost:.1f})")
#             else:
#                 # Track not found, increment age
#                 track.frames_without_detection += 1
#                 if best_match:
#                     logger.debug(f"Frame {frame_num}: Track {track_id} - YOLO detection too far (cost={best_cost:.1f} > {max_threshold:.1f})")
#                 else:
#                     logger.debug(f"Frame {frame_num}: Track {track_id} - no YOLO detections available")
        
#         # =================================================================
#         # CASE 3: Handle remaining YOLO detections with SAFE RESURRECTION
#         # =================================================================
#         # For YOLO detections that haven't been matched to blobs or tracks,
#         # try to resurrect a nearby struggling track BEFORE creating a new one
#         # BUT with strict safety checks to prevent cross-hotel jumps
        
#         # Safety thresholds (same as blob association)
#         MAX_RESURRECTION_DISTANCE = getattr(self.config.tracking, 'max_association_distance', 200) * self.scale_factor
#         MAX_VELOCITY = getattr(self.config.tracking, 'max_bee_velocity', 50) * self.scale_factor
        
#         for idx, det in enumerate(yolo_detections):
#             if idx in used_yolo_detections:
#                 continue
            
#             # Find the closest existing track (regardless of state)
#             best_existing_track = None
#             best_existing_track_id = None
#             best_existing_dist = float('inf')
            
#             for track_id, track in self.tracks.items():
#                 track_dist = np.sqrt(
#                     (track.centroid[0] - det['centroid'][0]) ** 2 +
#                     (track.centroid[1] - det['centroid'][1]) ** 2
#                 )
                
#                 if track_dist < best_existing_dist:
#                     best_existing_dist = track_dist
#                     best_existing_track = track
#                     best_existing_track_id = track_id
            
#             # Initial check: Is there a track within resurrection range?
#             RESURRECTION_SEARCH_DISTANCE = self.new_track_proximity_check * 2.5  # Wide search
            
#             if best_existing_track and best_existing_dist < RESURRECTION_SEARCH_DISTANCE:
#                 # Found a nearby track - now apply STRICT SAFETY CHECKS
                
#                 # SAFETY CHECK 1: Hard distance limit (prevent cross-hotel jumps)
#                 if best_existing_dist > MAX_RESURRECTION_DISTANCE:
#                     logger.debug(f"Frame {frame_num}: Resurrection REJECTED for track {best_existing_track_id} - "
#                             f"distance {best_existing_dist:.1f}px > {MAX_RESURRECTION_DISTANCE:.1f}px (creating new track)")
#                     # Too far - create new track instead
#                     logger.debug(f"Frame {frame_num}: Creating new track (resurrection blocked by distance)")
#                     self._create_track(det, frame_num)
#                     used_yolo_detections.add(idx)
#                     continue
                
#                 # SAFETY CHECK 2: Velocity sanity (prevent teleportation)
#                 implied_velocity = best_existing_dist  # pixels/frame (1 frame delta)
                
#                 if implied_velocity > MAX_VELOCITY:
#                     logger.debug(f"Frame {frame_num}: Resurrection REJECTED for track {best_existing_track_id} - "
#                             f"velocity {implied_velocity:.1f}px/frame > {MAX_VELOCITY:.1f}px/frame (creating new track)")
#                     # Too fast - create new track instead
#                     logger.debug(f"Frame {frame_num}: Creating new track (resurrection blocked by velocity)")
#                     self._create_track(det, frame_num)
#                     used_yolo_detections.add(idx)
#                     continue
                
#                 # SAFETY CHECK 3: Acceleration check (prevent sudden speed changes)
#                 current_speed = np.sqrt(
#                     float(best_existing_track.kalman.statePost[2])**2 + 
#                     float(best_existing_track.kalman.statePost[3])**2
#                 )
#                 speed_change = abs(implied_velocity - current_speed)
#                 MAX_ACCELERATION = getattr(self.config.tracking, 'max_bee_acceleration', 30) * self.scale_factor
                
#                 if speed_change > MAX_ACCELERATION:
#                     logger.debug(f"Frame {frame_num}: Resurrection REJECTED for track {best_existing_track_id} - "
#                             f"acceleration {speed_change:.1f}px/frame² > {MAX_ACCELERATION:.1f}px/frame² (creating new track)")
#                     # Acceleration too high - create new track instead
#                     logger.debug(f"Frame {frame_num}: Creating new track (resurrection blocked by acceleration)")
#                     self._create_track(det, frame_num)
#                     used_yolo_detections.add(idx)
#                     continue
                
#                 # ALL SAFETY CHECKS PASSED - Safe to resurrect!
#                 logger.info(f"Frame {frame_num}: 🔄 RESURRECTING track {best_existing_track_id} "
#                         f"(dist={best_existing_dist:.1f}px, vel={implied_velocity:.1f}px/frame, "
#                         f"aged {best_existing_track.frames_without_detection} frames)")
                
#                 # Update track state
#                 best_existing_track.bbox = det['bbox']
#                 best_existing_track.centroid = det['centroid']
#                 best_existing_track.species = det['species']
#                 best_existing_track.frames_without_detection = 0  # Reset age counter
#                 best_existing_track.last_yolo_confirmation = frame_num
                
#                 # Update trajectory history
#                 best_existing_track.trajectory_history.append((frame_num, det['centroid']))
#                 if len(best_existing_track.trajectory_history) > 30:
#                     best_existing_track.trajectory_history.pop(0)
                
#                 # Update Kalman filter with new measurement
#                 measurement = np.array([[det['centroid'][0]], [det['centroid'][1]]], dtype=np.float32)
#                 best_existing_track.kalman.correct(measurement)
                
#                 # Mark this YOLO detection as used
#                 used_yolo_detections.add(idx)
                
#             else:
#                 # No nearby track found - safe to create a NEW track
#                 # This YOLO detection represents a genuinely new bee
                
#                 if best_existing_track:
#                     logger.debug(f"Frame {frame_num}: Creating new track (YOLO-only) at {det['centroid']} "
#                             f"(nearest track {best_existing_track_id} is {best_existing_dist:.1f}px away)")
#                 else:
#                     logger.debug(f"Frame {frame_num}: Creating new track (YOLO-only) at {det['centroid']} "
#                             f"(no existing tracks)")
                
#                 self._create_track(det, frame_num)
#                 used_yolo_detections.add(idx)
        
#         # Log summary statistics
#         total_yolo = len(yolo_detections)
#         used_yolo = len(used_yolo_detections)
#         unused_yolo = total_yolo - used_yolo
        
#         if unused_yolo > 0:
#             logger.warning(f"Frame {frame_num}: {unused_yolo}/{total_yolo} YOLO detections unused (possible issue?)")








    
#     def _update_matched_tracks(
#         self,
#         matched_blobs: List[Dict],
#         matched_tracks: List[int],
#         frame_num: int
#     ):
#         """Update tracks that were matched to blobs.
        
#         Args:
#             matched_blobs: Blobs that were matched
#             matched_tracks: Track IDs that were matched
#             frame_num: Current frame number
#         """
#         for blob, track_id in zip(matched_blobs, matched_tracks):
#             track = self.tracks[track_id]
            
#             # Update track
#             track.bbox = blob['bbox']
#             track.centroid = blob['centroid']
#             track.frames_without_detection = 0
#             track.age += 1
            
#             # Add to trajectory history (keep last 30 positions)
#             track.trajectory_history.append((frame_num, blob['centroid']))
#             if len(track.trajectory_history) > 30:
#                 track.trajectory_history.pop(0)
            
#             # Update Kalman filter
#             measurement = np.array([[blob['centroid'][0]], [blob['centroid'][1]]], dtype=np.float32)
#             track.kalman.correct(measurement)
    
#     def _create_track(self, detection: Dict, frame_num: int):
#         """Create new track from detection.
        
#         Args:
#             detection: Detection dictionary
#             frame_num: Current frame number
#         """
#         # Initialize Kalman filter
#         kalman = cv2.KalmanFilter(4, 2)  # 4 state vars (x, y, vx, vy), 2 measurements (x, y)
#         kalman.measurementMatrix = np.array([[1, 0, 0, 0],
#                                               [0, 1, 0, 0]], dtype=np.float32)
#         kalman.transitionMatrix = np.array([[1, 0, 1, 0],
#                                              [0, 1, 0, 1],
#                                              [0, 0, 1, 0],
#                                              [0, 0, 0, 1]], dtype=np.float32)
#         kalman.processNoiseCov = np.eye(4, dtype=np.float32) * self.config.tracking.kalman_process_noise
        
#         # Initialize state
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
#             species=detection.get('species', 'unknown'),
#             age=1,
#             last_yolo_confirmation=frame_num
#         )
        
#         self.tracks[self.next_track_id] = track
#         self.next_track_id += 1
    
#     def _age_tracks(self, frame_num: int):
#         """Age out tracks without recent detections.
        
#         Args:
#             frame_num: Current frame number
#         """
#         to_remove = []
        
#         for track_id, track in self.tracks.items():
#             track.frames_without_detection += 1
            
#             if track.frames_without_detection > self.config.tracking.max_age:
#                 to_remove.append(track_id)
        
#         for track_id in to_remove:
#             del self.tracks[track_id]
    
#     def _get_current_detections(self, frame_num: int, debug_data: Optional[Dict] = None) -> List[Dict]:
#         """Get current detections from active tracks with optional debug data.
        
#         Args:
#             frame_num: Current frame number
#             debug_data: Optional dict with 'blobs' and 'yolo_detections' for visualization
            
#         Returns:
#             List of detection dictionaries
#         """
#         detections = []
        
#         for track_id, track in self.tracks.items():
#             detection = {
#                 'frame': frame_num,
#                 'track_id': track_id,
#                 'x1': track.bbox[0],
#                 'y1': track.bbox[1],
#                 'x2': track.bbox[2],
#                 'y2': track.bbox[3],
#                 'species': track.species
#             }
            
#             # Add debug data if provided (for visualization)
#             if debug_data is not None:
#                 detection['debug_blobs'] = debug_data.get('blobs', [])
#                 detection['debug_yolo'] = debug_data.get('yolo_detections', [])
            
#             detections.append(detection)
        
#         return detections
    
#     def _visualize_frame(
#         self,
#         frame: np.ndarray,
#         frame_num: int,
#         is_low_res: bool,
#         blobs: List[Dict] = None,
#         yolo_detections: List[Dict] = None
#     ) -> np.ndarray:
#         """Visualize frame with tracks, blobs, YOLO detections, and trajectories.
        
#         Args:
#             frame: Frame to annotate
#             frame_num: Frame number
#             is_low_res: Whether in low-res mode
#             blobs: Optional list of FG/BG blobs to visualize
#             yolo_detections: Optional list of YOLO detections to visualize
            
#         Returns:
#             Annotated frame
#         """
#         viz_frame = frame.copy()
        
#         # Draw FG/BG blobs (cyan boxes)
#         if blobs:
#             for blob in blobs:
#                 x1, y1, x2, y2 = [int(c) for c in blob['bbox']]
#                 cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Cyan
#                 # Draw blob centroid
#                 cx, cy = [int(c) for c in blob['centroid']]
#                 cv2.circle(viz_frame, (cx, cy), 2, (255, 255, 0), -1)
        
#         # Draw YOLO detections (orange boxes)
#         if yolo_detections:
#             for det in yolo_detections:
#                 x1, y1, x2, y2 = [int(c) for c in det['bbox']]
#                 cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
#                 label = f"DL:{det.get('species', 'unknown')}"
#                 cv2.putText(
#                     viz_frame, label, (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1
#                 )
        
#         # Draw tracks with trajectory lines
#         for track_id, track in self.tracks.items():
#             x1, y1, x2, y2 = [int(c) for c in track.bbox]
            
#             # Draw trajectory line (last 30 positions)  #  I think we should only draw active tracks not all tracks
#             if len(track.trajectory_history) > 1:
#                 for i in range(len(track.trajectory_history) - 1):
#                     pt1 = (int(track.trajectory_history[i][1][0]), int(track.trajectory_history[i][1][1]))
#                     pt2 = (int(track.trajectory_history[i+1][1][0]), int(track.trajectory_history[i+1][1][1]))
#                     cv2.line(viz_frame, pt1, pt2, (255, 255, 0), 1)  # Cyan trajectory
            
#             # Color based on frames since YOLO confirmation
#             frames_since_yolo = frame_num - track.last_yolo_confirmation
#             if frames_since_yolo == 0:
#                 color = (0, 255, 0)  # Green = YOLO confirmed
#             elif frames_since_yolo < 10:
#                 color = (0, 255, 255)  # Yellow = Recent YOLO
#             else:
#                 color = (255, 0, 0)  # Blue = FG tracking only
            
#             cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
            
#             label = f"ID:{track_id} {track.species}"
#             cv2.putText(
#                 viz_frame, label, (x1, y1 - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
#             )
        
#         # Draw mode indicator
#         # mode_text = "LOW-RES" if is_low_res else "HIGH-RES (HyDaT)"
#         # mode_color = (255, 0, 0) if is_low_res else (0, 255, 0)
        
#         # cv2.putText(
#         #     viz_frame, f"Mode: {mode_text}", (10, 30),
#         #     cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2
#         # )
        
#         # cv2.putText(
#         #     viz_frame, f"Frame: {frame_num}", (10, 60),
#         #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
#         # )
        
#         # cv2.putText(
#         #     viz_frame, f"Tracks: {len(self.tracks)}", (10, 90),
#         #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
#         # )
        
#         # Draw legend
#         legend_y = 120
#         cv2.putText(viz_frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#         cv2.putText(viz_frame, "Green=DL confirmed", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
#         cv2.putText(viz_frame, "Yellow=Recent DL", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#         cv2.putText(viz_frame, "Blue=FG only", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
#         cv2.putText(viz_frame, "Cyan=Blobs/Traj", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
#         cv2.putText(viz_frame, "Orange=DL detections", (10, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
#         return viz_frame


# # Backward compatibility aliases
# AdaptiveMotionDetector = HyDaTTracker
# HybridMotionDetector = HyDaTTracker
# MotionDetector = HyDaTTracker















"""HyDaT-inspired motion detection and tracking.

This module implements the HyDaT (Hybrid Detection and Tracking) approach:
1. Continuous FG/BG segmentation for motion detection
2. Track foreground blobs using predictions (Kalman filter)
3. Associate blobs to predicted positions
4. Run YOLO only when:
   - New insects detected (blob with no track association)
   - Missing insects (track with no blob association)
   - Unassociated detections/predictions

This dramatically reduces YOLO inference calls while maintaining accuracy.
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
import cv2
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from beemonitor.core.config import Config


logger = logging.getLogger(__name__)

# Type aliases
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class TrackState:
    """State for a tracked object."""
    track_id: int
    bbox: BBox
    centroid: Point
    kalman: cv2.KalmanFilter
    frames_without_detection: int
    species: str
    age: int
    last_yolo_confirmation: int  # Frame number of last YOLO confirmation
    trajectory_history: list = None  # List of (frame_num, centroid) for visualization
    
    def __post_init__(self):
        if self.trajectory_history is None:
            self.trajectory_history = []


class HyDaTTracker:
    """HyDaT-inspired tracker with FG/BG segmentation and selective YOLO.
    
    Features:
    - Continuous FG/BG segmentation (MOG2)
    - Kalman filter prediction for each track
    - Hungarian algorithm for blob-to-track association
    - YOLO only when association fails
    - Low-resolution mode when no activity
    
    Attributes:
        model: YOLO model for confirmation
        config: Configuration object
        bg_subtractor: Background subtractor (MOG2)
        bg_subtractor_lowres: Background subtractor for low-res mode
        tracks: Dictionary of active tracks
        next_track_id: Next available track ID
        use_gpu: Whether GPU is available
    """
    
    def __init__(self, model, config: Optional[Config] = None, use_gpu: Optional[bool] = None):
        """Initialize HyDaT tracker.
        
        Args:
            model: YOLO model for confirmation
            config: Configuration object
            use_gpu: Use GPU if available (default: auto-detect)
        """
        self.model = model
        self.config = config if config is not None else Config.default()
        
        # Calculate scale factor from hotel/ROI size
        # This automatically adjusts pixel-based parameters for camera distance
        self.scale_factor = self._calculate_scale_factor()
        
        # Read optimized parameters from config (all tuned in config.py)
        # Apply scaling for pixel-based parameters
        
        # Aspect ratio filtering for FG blobs (dimensionless - no scaling)
        self.min_aspect_ratio = self.config.tracking.min_blob_aspect_ratio
        self.max_aspect_ratio = self.config.tracking.max_blob_aspect_ratio
        
        # Area filtering (SCALED by scale_factor^2 - area scales quadratically)
        base_min_area = getattr(self.config.tracking, 'min_blob_area_pixels', 200)
        base_max_area = getattr(self.config.tracking, 'max_blob_area_pixels', 5000)
        self.min_blob_area_pixels = base_min_area * (self.scale_factor ** 2)
        self.max_blob_area_pixels = base_max_area * (self.scale_factor ** 2)
        
        # Distance thresholds (SCALED linearly - distance scales linearly)
        base_new_track_dist = getattr(self.config.tracking, 'new_track_distance_threshold', 100)
        base_proximity_check = getattr(self.config.tracking, 'new_track_proximity_check', 50)
        self.new_track_distance_threshold = base_new_track_dist * self.scale_factor
        self.new_track_proximity_check = base_proximity_check * self.scale_factor
        
        # Scale association threshold base (critical for blob-track matching)
        base_assoc_threshold = self.config.tracking.association_threshold_base
        self.config.tracking.association_threshold_base = base_assoc_threshold * self.scale_factor
        
        # Scale max blob distance from hotel
        base_max_blob_dist = getattr(self.config.tracking, 'max_blob_distance_from_hotel', 1800.0)
        if base_max_blob_dist != float('inf'):
            self.config.tracking.max_blob_distance_from_hotel = base_max_blob_dist * self.scale_factor
        
        # Scale nest padding (used in event detection and visualization)
        if hasattr(self.config, 'nest'):
            base_padding_x = getattr(self.config.nest, 'padding_x_base', 5)
            base_padding_y = getattr(self.config.nest, 'padding_y_base', 7)
            # Store scaled padding for later use
            self.nest_padding_x_scaled = base_padding_x * self.scale_factor
            self.nest_padding_y_scaled = base_padding_y * self.scale_factor
        
        # Auto-detect GPU
        if use_gpu is None:
            self.use_gpu = self._detect_gpu()
        else:
            self.use_gpu = use_gpu
        
        # Initialize background subtractors
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        self.bg_subtractor_lowres = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Tracking state
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        
        # Species mapping
        self.label_map = self.config.tracking.label_map
        self.tracking_classes = self.config.tracking.tracking_classes
        
        logger.info(f"Initialized HyDaTTracker (GPU: {self.use_gpu})")
        logger.info(f"=== AUTO-SCALING APPLIED ===")
        logger.info(f"Scale factor: {self.scale_factor:.2f}x (hotel size relative to reference)")
        logger.info(f"Scaled parameters:")
        logger.info(f"  Area filter: {int(self.min_blob_area_pixels)}-{int(self.max_blob_area_pixels)} px²")
        logger.info(f"  Association threshold: {self.config.tracking.association_threshold_base:.1f} px")
        logger.info(f"  New track distance: {int(self.new_track_distance_threshold)} px")
        logger.info(f"  Proximity check: {int(self.new_track_proximity_check)} px")
        if hasattr(self.config.tracking, 'max_blob_distance_from_hotel'):
            max_dist = self.config.tracking.max_blob_distance_from_hotel
            if max_dist != float('inf'):
                logger.info(f"  Max blob distance: {int(max_dist)} px")
        logger.info(f"Unscaled parameters:")
        logger.info(f"  Max age: {self.config.tracking.max_age} frames")
        logger.info(f"  Aspect ratio: {self.min_aspect_ratio:.2f}-{self.max_aspect_ratio:.2f}")
        logger.info(f"  Adaptive association: {'ENABLED' if self.config.tracking.adaptive_association else 'DISABLED'}")
        logger.info(f"  Tracking classes: {self.tracking_classes}")
    
    def _calculate_scale_factor(self) -> float:
        """Calculate scale factor based on hotel/ROI size relative to frame.
        
        This automatically adjusts pixel-based parameters (area, distance, etc.)
        based on camera distance. When hotel is far (small in frame), parameters
        scale down proportionally.
        
        Reference assumptions (baseline for parameter tuning):
        - Frame: 1920x1080 (Full HD)
        - Hotel: ~800x400 pixels (close-up, 60-70% of frame height)
        - Bee blob size: ~200-5000 pixels²
        - Association distance: ~100 pixels
        
        Scaling logic:
        - Linear dimensions (width, height) scale by factor
        - Areas scale by factor²
        - Distances scale by factor
        
        Returns:
            Scale factor (1.0 = reference size, 0.5 = half size/far camera, 2.0 = double size/close camera)
        """
        # Check if auto-scaling is disabled
        if getattr(self.config.tracking, 'disable_auto_scaling', False):
            logger.info("Auto-scaling DISABLED - using raw config parameters")
            return 1.0
        
        # Get hotel box dimensions
        if not hasattr(self.config, 'hotel_box') or self.config.hotel_box is None:
            logger.warning("No hotel_box in config - using scale factor 1.0 (no scaling)")
            logger.warning("For auto-scaling to work, set config.hotel_box = (x1, y1, x2, y2)")
            return 1.0
        
        x1, y1, x2, y2 = self.config.hotel_box.get_box_bounds(self.config.video.res_width, self.config.video.res_height)
        hotel_width = x2 - x1
        hotel_height = y2 - y1
        
        # Reference dimensions (close-up camera view - what parameters are tuned for)
        REFERENCE_FRAME_WIDTH = 1920
        REFERENCE_FRAME_HEIGHT = 1080
        REFERENCE_HOTEL_WIDTH = 800  # ~42% of frame width
        REFERENCE_HOTEL_HEIGHT = 400  # ~37% of frame height (60-70% visually due to aspect)
        
        # Get actual frame dimensions
        frame_width = getattr(self.config.video, 'res_width', REFERENCE_FRAME_WIDTH)
        frame_height = getattr(self.config.video, 'res_height', REFERENCE_FRAME_HEIGHT)
        
        # Calculate scale factors in both dimensions
        # Compare actual hotel size (as % of frame) to reference hotel size (as % of reference frame)
        scale_height = (hotel_height / frame_height) / (REFERENCE_HOTEL_HEIGHT / REFERENCE_FRAME_HEIGHT)
        scale_width = (hotel_width / frame_width) / (REFERENCE_HOTEL_WIDTH / REFERENCE_FRAME_WIDTH)
        
        # Use average of both (balanced approach)
        # Could use min for conservative (safer), max for aggressive (riskier)
        scale_factor = (scale_height + scale_width) / 2
        
        # Clamp to reasonable range (0.1x to 5.0x)
        # Below 0.1x: hotel too small, parameters would be unreliable
        # Above 5.0x: hotel too large, parameters would be excessive
        scale_factor = max(0.1, min(5.0, scale_factor))
        
        # Diagnostic logging
        hotel_pct_width = (hotel_width / frame_width) * 100
        hotel_pct_height = (hotel_height / frame_height) * 100
        ref_pct_width = (REFERENCE_HOTEL_WIDTH / REFERENCE_FRAME_WIDTH) * 100
        ref_pct_height = (REFERENCE_HOTEL_HEIGHT / REFERENCE_FRAME_HEIGHT) * 100
        
        logger.info(f"=== AUTO-SCALING ANALYSIS ===")
        logger.info(f"Frame: {int(frame_width)}x{int(frame_height)}")
        logger.info(f"Hotel: {int(hotel_width)}x{int(hotel_height)}")
        logger.info(f"Hotel coverage: {hotel_pct_width:.1f}% width, {hotel_pct_height:.1f}% height")
        logger.info(f"Reference: {ref_pct_width:.1f}% width, {ref_pct_height:.1f}% height")
        logger.info(f"Scale factors: width={scale_width:.2f}x, height={scale_height:.2f}x")
        logger.info(f"Final scale factor: {scale_factor:.2f}x")
        
        if scale_factor < 0.5:
            logger.warning(f"Scale factor {scale_factor:.2f}x is quite small - hotel may be very far")
            logger.warning("Consider using a tighter ROI or adjusting camera")
        elif scale_factor > 2.0:
            logger.warning(f"Scale factor {scale_factor:.2f}x is quite large - hotel may be very close")
            logger.warning("Parameters scaled up significantly")
        
        return scale_factor
    
    def initialize_background_from_video(
        self,
        video_path: str,
        site_roi: BBox,
        res_height: int,
        res_width: int,
        max_frames: int = 200,
        target_clean_frames: int = 50,
        config: Optional[Config] = None
    ) -> int:
        """Initialize background model using only bee-free frames.
        
        This method scans the video and builds the background model using only
        frames where no bees (tracking classes) are detected by YOLO. This ensures
        a clean background model without bee interference.
        
        Args:
            video_path: Path to video file
            site_roi: Region of interest (x1, y1, x2, y2)
            res_height: Target frame height
            res_width: Target frame width
            max_frames: Maximum frames to scan (default: 200)
            target_clean_frames: Target number of clean frames to use (default: 50)
            config: Optional config override
            
        Returns:
            Number of clean frames used to build background model
            
        Example:
            >>> tracker = HyDaTTracker(model, config)
            >>> clean_frames = tracker.initialize_background_from_video(
            ...     "video.mp4", site_roi, 1080, 1920
            ... )
            >>> print(f"Background model built from {clean_frames} bee-free frames")
        """
        if config is None:
            config = self.config
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        logger.info("Initializing background model from bee-free frames...")
        logger.info(f"Scanning up to {max_frames} frames, target: {target_clean_frames} clean frames")
        
        frame_num = 0
        clean_frames_used = 0
        frames_with_bees = 0
        
        while frame_num < max_frames and clean_frames_used < target_clean_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (res_width, res_height))
            
            # Determine tracking region based on config
            if getattr(config.tracking, 'track_full_frame', False):
                # Use entire frame for background initialization
                frame_roi = frame
            else:
                # Use ROI only (default behavior)
                x1_roi, y1_roi, x2_roi, y2_roi = [int(c) for c in site_roi]
                frame_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
            
            # Run YOLO detection to check for bees
            results = self.model.predict(
                frame_roi,
                conf=config.tracking.confidence_threshold,
                iou=config.tracking.iou_threshold,
                verbose=False,
                device='0' if self.use_gpu else 'cpu'
            )
            
            # Check if any tracking class objects detected
            has_bees = False
            if len(results) > 0 and results[0].boxes is not None:
                for cls in results[0].boxes.cls:
                    class_id = int(cls.cpu().numpy())
                    if class_id in self.tracking_classes:
                        has_bees = True
                        frames_with_bees += 1
                        break
            
            # Only use frame if no bees detected
            if not has_bees:
                # Add to background model (both regular and low-res)
                self.bg_subtractor.apply(frame_roi, learningRate=0.1)
                
                # Also add to low-res background model
                scale = config.tracking.low_res_scale_factor
                low_res_roi = cv2.resize(frame_roi, None, fx=scale, fy=scale)
                self.bg_subtractor_lowres.apply(low_res_roi, learningRate=0.1)
                
                clean_frames_used += 1
                
                if clean_frames_used % 10 == 0:
                    logger.info(f"  Added {clean_frames_used}/{target_clean_frames} clean frames...")
            
            frame_num += 1
        
        cap.release()
        
        logger.info(f"Background initialization complete:")
        logger.info(f"  Scanned {frame_num} frames")
        logger.info(f"  Clean frames used: {clean_frames_used}")
        logger.info(f"  Frames with bees: {frames_with_bees}")
        logger.info(f"  Success rate: {clean_frames_used/frame_num*100:.1f}%")
        
        if clean_frames_used < 10:
            logger.warning(f"Only {clean_frames_used} clean frames found! Background model may be poor.")
            logger.warning("Consider using more frames or checking if bees are always present.")
        
        return clean_frames_used
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
                return True
        except ImportError:
            pass
        
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("OpenCV CUDA support detected")
            return True
        
        logger.info("No GPU detected, using CPU")
        return False
    
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
        """Detect and track bees using HyDaT approach.
        
        Args:
            video_path: Path to video file
            site_roi: Region of interest (x1, y1, x2, y2)
            res_height: Target frame height
            res_width: Target frame width
            visualize: Whether to save visualization
            output_folder: Output directory
            config: Optional config override
            initialize_background: Whether to initialize BG model from bee-free frames
            
        Returns:
            DataFrame with tracking results
        """
        if config is None:
            config = self.config
        
        # Initialize background model from bee-free frames
        if initialize_background:
            logger.info("Initializing background model from bee-free frames...")
            clean_frames = self.initialize_background_from_video(
                video_path, site_roi, res_height, res_width, config=config
            )
            if clean_frames < 10:
                logger.warning("Background initialization may be insufficient!")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"Processing {total_frames} frames with HyDaT tracking")
        
        # Log tracking region mode
        if getattr(config.tracking, 'track_full_frame', False):
            logger.info("FULL FRAME tracking mode enabled")
            logger.info(f"  - Motion detection: entire frame ({res_width}x{res_height})")
            logger.info(f"  - Tracking region: entire frame ({res_width}x{res_height})")
            logger.info(f"  - YOLO detection: entire frame")
        else:
            logger.info("HYBRID tracking mode (default)")
            logger.info(f"  - Motion detection: ROI only {site_roi}")
            logger.info(f"  - Tracking region: FULL FRAME (tracks can leave ROI)")
            logger.info(f"  - YOLO detection: ROI + full-frame when tracks outside ROI")
        
        # Initialize output
        output_video = None
        if visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(output_folder, f"tracking_{os.path.basename(video_path).rsplit('.', 1)[0]}.mp4")
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (res_width, res_height))
        
        # Visualization mode: debug = all frames, normal = activity periods with padding
        debug_mode = getattr(config.tracking, 'debug_mode', False)
        viz_padding_frames = getattr(config.tracking, 'viz_padding_frames', 15)  # Padding for smooth transitions
        
        if visualize:
            if debug_mode:
                logger.info("DEBUG MODE: Visualizing ALL frames")
            else:
                logger.info(f"NORMAL MODE: Visualizing activity periods with {viz_padding_frames} frame padding")
        
        # Processing state
        frame_num = 0
        all_detections = []
        is_low_res = True
        frames_without_activity = 0
        frames_since_last_activity = 0  # Track frames since last activity for padding
        
        # Statistics
        low_res_frames = 0
        high_res_frames = 0
        yolo_calls = 0
        fg_blobs_tracked = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (res_width, res_height))
            
            # Determine motion detection region based on config
            # NOTE: In default mode, motion detection is ROI-only but TRACKING extends beyond ROI
            # DL will run on full frame when tracks predict outside ROI
            if getattr(config.tracking, 'track_full_frame', False):
                # Full frame mode: motion detection on entire frame
                frame_roi = frame
                x1_roi, y1_roi = 0, 0
                x2_roi, y2_roi = res_width, res_height
            else:
                # Hybrid mode (default): motion detection in ROI, tracking beyond ROI
                x1_roi, y1_roi, x2_roi, y2_roi = [int(c) for c in site_roi]
                frame_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
            
            # Decide processing mode
            if is_low_res and config.tracking.enable_low_res_mode:
                # Low-resolution check
                if frame_num % config.tracking.low_res_check_interval == 0:
                    fg_detected = self._check_fg_activity_lowres(frame_roi, config)
                    low_res_frames += 1
                    
                    if fg_detected:
                        is_low_res = False
                        logger.info(f"Frame {frame_num}: Switching to HIGH-RES mode")
                    else:
                        # Debug mode: visualize even in low-res
                        if visualize and output_video and debug_mode:
                            viz_frame = self._visualize_frame(
                                frame, frame_num, is_low_res,
                                blobs=[],
                                yolo_detections=[]
                            )
                            output_video.write(viz_frame)
                else:
                    low_res_frames += 1
                    
                    # Debug mode: visualize even in low-res skip frames
                    if visualize and output_video and debug_mode:
                        viz_frame = self._visualize_frame(
                            frame, frame_num, is_low_res,
                            blobs=[],
                            yolo_detections=[]
                        )
                        output_video.write(viz_frame)
                    
                    frame_num += 1
                    continue
            
            if not is_low_res:
                # High-resolution processing with HyDaT
                high_res_frames += 1
                
                # Step 1: Run FG/BG segmentation
                fg_mask = self.bg_subtractor.apply(frame_roi)
                fg_mask[fg_mask == 127] = 0  # Remove shadows
                
                # Step 2: Extract FG blobs (filtered by aspect ratio to avoid YOLO on noise)
                blobs = self._extract_blobs(fg_mask, config, x1_roi, y1_roi)
                
                # Step 3: Predict track positions (always, even without blobs)
                predictions = self._predict_tracks(frame_num)
                
                # Check if any tracks predict outside ROI (need full-frame DL)
                tracks_outside_roi = self._check_tracks_outside_roi(
                    predictions, (x1_roi, y1_roi, x2_roi, y2_roi)
                )
                
                # Initialize frame debug data
                frame_debug_data = {
                    'blobs': blobs,
                    'yolo_detections': []
                }
                
                # Determine if we have activity
                has_blobs = len(blobs) > 0
                has_active_tracks = len(predictions) > 0
                
                if has_blobs or has_active_tracks:
                    # Activity detected (either blobs OR existing tracks)
                    frames_without_activity = 0
                    frames_since_last_activity = 0  # Reset padding counter
                    
                    if has_blobs:
                        fg_blobs_tracked += len(blobs)
                        
                        # Step 4: Associate blobs to predictions
                        matched_blobs, matched_tracks, unmatched_blobs, unmatched_tracks = \
                            self._associate_blobs_to_tracks(blobs, predictions)
                    else:
                        # No blobs but we have active tracks (fast turn scenario!)
                        # All tracks are unmatched
                        matched_blobs = []
                        matched_tracks = []
                        unmatched_blobs = []
                        unmatched_tracks = list(predictions.keys())
                        logger.debug(f"Frame {frame_num}: No blobs detected but {len(unmatched_tracks)} active tracks - running YOLO")
                    
                    # Step 5: Run YOLO when needed
                    # CRITICAL: Run YOLO if we have unmatched tracks even without blobs
                    # This handles fast turns where FG detection fails
                    yolo_needed = len(unmatched_blobs) > 0 or len(unmatched_tracks) > 0
                    
                    # ADDITIONAL: Force YOLO check for tracks without recent confirmation
                    # This catches tracks that may be drifting or need verification
                    if not yolo_needed and self.config.tracking.max_frames_without_yolo > 0:
                        stale_tracks = []
                        for track_id, track in self.tracks.items():
                            frames_since_yolo = frame_num - track.last_yolo_confirmation
                            if frames_since_yolo > self.config.tracking.max_frames_without_yolo:
                                stale_tracks.append(track_id)
                        
                        if stale_tracks:
                            yolo_needed = True
                            logger.debug(f"Frame {frame_num}: Forcing YOLO check for {len(stale_tracks)} stale tracks")
                    
                    # NEW: Also run YOLO if tracks predict outside ROI
                    if tracks_outside_roi:
                        yolo_needed = True
                        logger.debug(f"Frame {frame_num}: Tracks outside ROI - running full-frame YOLO")
                    
                    if yolo_needed:
                        yolo_calls += 1
                        
                        # Determine YOLO region based on track positions
                        if tracks_outside_roi:
                            # Run YOLO on FULL FRAME (tracks outside ROI)
                            yolo_detections = self._run_yolo(
                                frame, config, 0, 0  # Full frame, no offset
                            )
                            logger.debug(f"Frame {frame_num}: Running full-frame YOLO for outside-ROI tracks")
                        else:
                            # Run YOLO on ROI only (normal HyDaT)
                            yolo_detections = self._run_yolo(
                                frame_roi, config, x1_roi, y1_roi
                            )
                        
                        frame_debug_data['yolo_detections'] = yolo_detections
                        
                        # Update tracks with YOLO confirmations
                        self._update_tracks_with_yolo(
                            yolo_detections, unmatched_blobs, unmatched_tracks, frame_num
                        )
                    
                    # Step 6: Update matched tracks with blob positions
                    if matched_blobs:
                        self._update_matched_tracks(matched_blobs, matched_tracks, frame_num)
                    
                    # Record detections with debug data
                    detections = self._get_current_detections(frame_num, frame_debug_data)
                    logger.debug(f"Frame {frame_num}: Recording {len(detections)} detections, total so far: {len(all_detections) + len(detections)}")
                    all_detections.extend(detections)
                    
                    # Visualization during active tracking
                    if visualize and output_video:
                        viz_frame = self._visualize_frame(
                            frame, frame_num, is_low_res,
                            blobs=frame_debug_data['blobs'],
                            yolo_detections=frame_debug_data['yolo_detections']
                        )
                        output_video.write(viz_frame)
                else:
                    frames_without_activity += 1
                    frames_since_last_activity += 1
                    
                    # Age out tracks without detections
                    self._age_tracks(frame_num)
                    
                    # Visualization during padding period (normal mode only)
                    # In normal mode, visualize for padding_frames after activity stops
                    if visualize and output_video and not debug_mode:
                        if frames_since_last_activity <= viz_padding_frames:
                            viz_frame = self._visualize_frame(
                                frame, frame_num, is_low_res,
                                blobs=[],
                                yolo_detections=[]
                            )
                            output_video.write(viz_frame)
                
                # Switch back to low-res if no activity
                if frames_without_activity > 30 and len(self.tracks) == 0:
                    is_low_res = True
                    logger.info(f"Frame {frame_num}: Switching to LOW-RES mode")
            else:
                # Low-res mode - skip visualization (no activity to show)
                pass
            
            frame_num += 1
            
            if frame_num % 100 == 0:
                logger.info(f"Processed {frame_num}/{total_frames} frames")
        
        cap.release()
        if output_video:
            output_video.release()
        
        # Log statistics
        logger.info(f"Processing complete:")
        logger.info(f"  Low-res frames: {low_res_frames}")
        logger.info(f"  High-res frames: {high_res_frames}")
        logger.info(f"  YOLO calls: {yolo_calls}")
        logger.info(f"  FG blobs tracked: {fg_blobs_tracked}")
        logger.info(f"  YOLO reduction: {(1 - yolo_calls/max(high_res_frames, 1))*100:.1f}%")
        logger.info(f"  Total detections collected: {len(all_detections)}")
        
        # Convert to grouped track format for event processor
        logger.info("Converting detections to grouped format...")
        grouped_tracks = self._convert_to_grouped_format(all_detections)
        logger.info(f"  Grouped format: {len(grouped_tracks)} periods")
        
        return grouped_tracks
    
    def _convert_to_grouped_format(self, all_detections: List[Dict]) -> pd.DataFrame:
        """Convert flat detection list to grouped track format expected by event processor.
        
        Args:
            all_detections: List of detection dictionaries
            
        Returns:
            DataFrame with columns: frame_number (tuple), tracks (list), detections (dict)
        """
        logger.debug(f"Converting {len(all_detections)} detections to grouped format")
        
        if not all_detections:
            logger.warning("No detections to convert - returning empty DataFrame")
            return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
        try:
            # Extract debug data before creating DataFrame (lists cause issues)
            debug_data_by_frame = {}
            for det in all_detections:
                frame_num = det['frame']
                if 'debug_blobs' in det and frame_num not in debug_data_by_frame:
                    debug_data_by_frame[frame_num] = {
                        'blobs': det.get('debug_blobs', []),
                        'yolo': det.get('debug_yolo', [])
                    }
            
            # Remove debug data from detections before DataFrame conversion
            clean_detections = []
            for det in all_detections:
                clean_det = {k: v for k, v in det.items() if k not in ['debug_blobs', 'debug_yolo']}
                clean_detections.append(clean_det)
            
            logger.debug(f"Creating DataFrame from {len(clean_detections)} clean detections")
            
            # Convert to DataFrame for easier grouping
            df = pd.DataFrame(clean_detections)
            
            logger.debug(f"DataFrame created: {len(df)} rows, columns: {list(df.columns)}")
            
        except Exception as e:
            logger.error(f"Error creating DataFrame from detections: {e}")
            logger.error(f"First detection: {all_detections[0] if all_detections else 'None'}")
            raise
        
        # Split into activity periods based on temporal gaps
        # Use slightly longer threshold than max_age for period splitting
        periods = self._split_into_periods(df, gap_threshold=int(self.config.tracking.max_age * 1.1))
        
        logger.debug(f"Split into {len(periods)} activity periods")
        
        result_rows = []
        for period_df in periods:
            # Group by track_id within this period
            track_groups = {}
            
            for track_id in period_df['track_id'].unique():
                track_df = period_df[period_df['track_id'] == track_id].sort_values('frame')
                
                # Split track by gaps - creates sub-tracks if large gaps exist
                # Use max_age as threshold to match tracking behavior
                track_segments = self._split_track_by_gaps(track_df, gap_threshold=self.config.tracking.max_age)
                
                for segment_idx, segment_df in enumerate(track_segments):
                    # Assign unique ID for each segment
                    unique_id = f"{track_id}_{segment_idx}" if len(track_segments) > 1 else track_id
                    
                    centroids = [
                        ((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
                        for _, row in segment_df.iterrows()
                    ]
                    bboxes = [
                        (row['x1'], row['y1'], row['x2'], row['y2'])
                        for _, row in segment_df.iterrows()
                    ]
                    frame_numbers = segment_df['frame'].tolist()
                    
                    # Filter out very short tracks (likely noise)
                    if len(frame_numbers) >= self.config.tracking.min_track_length:
                        track_groups[unique_id] = (unique_id, centroids, bboxes, frame_numbers)
            
            if not track_groups:
                continue
            
            # Create row for this period
            all_tracks = list(track_groups.values())
            min_frame = period_df['frame'].min()
            max_frame = period_df['frame'].max()
            
            # Build detections dict
            frame_detections = {}
            for frame_num in period_df['frame'].unique():
                frame_df = period_df[period_df['frame'] == frame_num]
                
                # Get debug data from extracted dict (if available)
                frame_debug = debug_data_by_frame.get(int(frame_num), {'blobs': [], 'yolo': []})
                
                frame_detections[int(frame_num)] = {
                    'boxes': [
                        (row['x1'], row['y1'], row['x2'], row['y2'])
                        for _, row in frame_df.iterrows()
                    ],
                    'label': frame_df['species'].tolist(),
                    'debug_blobs': frame_debug['blobs'],
                    'debug_yolo': frame_debug['yolo']
                }
            
            result_rows.append({
                'frame_number': (int(min_frame), int(max_frame)),
                'tracks': all_tracks,
                'detections': frame_detections
            })
        
        logger.debug(f"Created {len(result_rows)} result rows")
        result_df = pd.DataFrame(result_rows) if result_rows else pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        logger.debug(f"Returning DataFrame with {len(result_df)} rows")
        return result_df
    
    def _split_into_periods(self, df: pd.DataFrame, gap_threshold: int = 100) -> List[pd.DataFrame]:
        """Split detections into activity periods based on temporal gaps.
        
        Args:
            df: DataFrame of detections
            gap_threshold: Gap size (frames) to split periods
            
        Returns:
            List of DataFrames, one per period
        """
        df = df.sort_values('frame')
        frames = df['frame'].tolist()
        
        periods = []
        current_period_start = 0
        
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap > gap_threshold:
                # End current period, start new one
                periods.append(df.iloc[current_period_start:i+1].copy())
                current_period_start = i + 1
        
        # Add final period
        if current_period_start < len(df):
            periods.append(df.iloc[current_period_start:].copy())
        
        return periods
    
    def _split_track_by_gaps(
        self, 
        track_df: pd.DataFrame, 
        gap_threshold: int = 30
    ) -> List[pd.DataFrame]:
        """Split a track into segments if it has large gaps.
        
        Args:
            track_df: DataFrame for a single track
            gap_threshold: Gap size (frames) to split track
            
        Returns:
            List of DataFrames, one per track segment
        """
        frames = track_df['frame'].tolist()
        
        segments = []
        current_segment_start = 0
        
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap > gap_threshold:
                # End current segment, start new one
                segments.append(track_df.iloc[current_segment_start:i+1].copy())
                current_segment_start = i + 1
        
        # Add final segment
        if current_segment_start < len(track_df):
            segments.append(track_df.iloc[current_segment_start:].copy())
        
        return segments
    
    def _check_fg_activity_lowres(
        self,
        frame_roi: np.ndarray,
        config: Config
    ) -> bool:
        """Check for FG activity in low-resolution mode.
        
        Args:
            frame_roi: ROI portion of frame
            config: Configuration
            
        Returns:
            True if activity detected
        """
        # Scale down
        scale = config.tracking.low_res_scale_factor
        low_res_roi = cv2.resize(frame_roi, None, fx=scale, fy=scale)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor_lowres.apply(low_res_roi)
        fg_mask[fg_mask == 127] = 0
        
        # Calculate area
        if config.tracking.fg_area_method == "pixels":
            fg_area = np.count_nonzero(fg_mask)
        else:
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            fg_area = sum(cv2.contourArea(c) for c in contours)
        
        # Check threshold (scaled)
        min_area = config.tracking.min_fg_area(
            frame_roi.shape[1], frame_roi.shape[0], config.hotel_box
        ) * (scale ** 2)
        
        return fg_area > min_area
    
    def _extract_blobs(
        self,
        fg_mask: np.ndarray,
        config: Config,
        x_offset: int,
        y_offset: int
    ) -> List[Dict]:
        """Extract foreground blobs from mask with robust multi-stage filtering.
        
        Filtering stages:
        1. Morphological cleanup (remove noise, fill gaps)
        2. Aspect ratio (bee-like shape, not lines/artifacts)
        3. Solidity (fill ratio - removes hollow/irregular noise)
        4. Distance to hotel (removes far-away noise)
        
        Args:
            fg_mask: Binary foreground mask
            config: Configuration
            x_offset: X offset for ROI
            y_offset: Y offset for ROI
            
        Returns:
            List of blob dictionaries with bbox, centroid, area, solidity
        """
        # Stage 1: Morphological cleanup to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
        
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        blobs = []
        min_area = config.tracking.min_contour_area(
            fg_mask.shape[1], fg_mask.shape[0], config.hotel_box
        )
        
        # Get hotel center for distance filtering
        hotel_center = self._get_hotel_center()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x + x_offset, y + y_offset, x + w + x_offset, y + h + y_offset)
            
            # Stage 2: Filter by aspect ratio (bee-like shapes only)
            if not self._is_valid_blob_shape(bbox):
                continue
            
            # Stage 3: Solidity check (contour area / bounding box area)
            # Real bees: 0.5-0.9, Noise: often < 0.3
            bbox_area = w * h
            solidity = area / bbox_area if bbox_area > 0 else 0
            min_solidity = getattr(config.tracking, 'min_blob_solidity', 0.4)
            if solidity < min_solidity:
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2
                cy = y + h / 2
            
            centroid = (cx + x_offset, cy + y_offset)
            
            # Stage 4: Distance filtering - reject blobs far from hotel
            if hotel_center is not None:
                dist_to_hotel = np.sqrt(
                    (centroid[0] - hotel_center[0])**2 + 
                    (centroid[1] - hotel_center[1])**2
                )
                # Calculate max allowed distance (hotel diagonal + margin)
                # SCALED by tracker's scale factor
                base_max_distance = getattr(config.tracking, 'max_blob_distance_from_hotel', float('inf'))
                if base_max_distance != float('inf'):
                    # Scale distance threshold based on hotel size
                    scaled_max_distance = base_max_distance * self.scale_factor
                    if dist_to_hotel > scaled_max_distance:
                        continue
            
            blob = {
                'bbox': bbox,
                'centroid': centroid,
                'area': area,
                'solidity': solidity
            }
            blobs.append(blob)
        
        return blobs
    
    def _get_hotel_center(self) -> Optional[Tuple[float, float]]:
        """Get center point of hotel box for distance filtering.
        
        Returns:
            (x, y) tuple of hotel center, or None if no hotel box
        """
        if not hasattr(self.config, 'hotel_box') or self.config.hotel_box is None:
            return None
        
        x1, y1, x2, y2 = self.config.hotel_box.get_box_bounds(self.config.video.res_width, self.config.video.res_height)
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _check_tracks_outside_roi(
        self,
        predictions: Dict[int, Dict],
        roi: Tuple[int, int, int, int]
    ) -> bool:
        """Check if any predicted track positions are outside ROI.
        
        This determines if we need to run DL on full frame to track bees
        that have moved beyond the ROI boundaries.
        
        Args:
            predictions: Dictionary of track predictions
            roi: ROI boundaries (x1, y1, x2, y2)
            
        Returns:
            True if any tracks predict outside ROI
        """
        if not predictions:
            return False
        
        x1_roi, y1_roi, x2_roi, y2_roi = roi
        
        for track_id, pred in predictions.items():
            cx, cy = pred['centroid']
            
            # Check if predicted position is outside ROI (with small margin)
            margin = 20  # pixels margin to catch near-boundary tracks
            if (cx < x1_roi - margin or cx > x2_roi + margin or
                cy < y1_roi - margin or cy > y2_roi + margin):
                return True
        
        return False
    
    def _is_valid_blob_shape(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if blob has bee-like aspect ratio and size.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            True if aspect ratio and size are within valid range
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if height == 0 or width == 0:
            return False
        
        # Check aspect ratio (bee-like shape)
        aspect_ratio = width / height
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return False
        
        # Check area (reasonable bee size)
        area = width * height
        if not (self.min_blob_area_pixels <= area <= self.max_blob_area_pixels):
            return False
        
        return True
    
    def _predict_tracks(self, frame_num: int) -> Dict[int, Dict]:
        """Predict positions of active tracks using Kalman filter.
        
        When tracks haven't been updated by DL recently, apply velocity damping
        to prevent unrealistic fast-moving predictions during direction changes.
        
        Args:
            frame_num: Current frame number
            
        Returns:
            Dictionary of track_id -> prediction
        """
        predictions = {}
        
        for track_id, track in self.tracks.items():
            # Check if track was recently confirmed by DL
            frames_since_dl = frame_num - track.last_yolo_confirmation
            
            # Apply velocity damping if no recent DL confirmation
            # This prevents predictions from continuing at full velocity during direction changes
            if frames_since_dl > 0:
                # Damping factor increases with time since DL confirmation
                # 0 frames: no damping (1.0), 5 frames: 50% damping (0.5), 10+ frames: 80% damping (0.2)
                damping_factor = max(0.2, 1.0 - (frames_since_dl * 0.1))
                
                # Apply damping to velocity components in Kalman state
                track.kalman.statePost[2] *= damping_factor  # vx
                track.kalman.statePost[3] *= damping_factor  # vy
            
            # Predict next position
            prediction = track.kalman.predict()
            pred_x, pred_y = prediction[0], prediction[1]
            
            # Estimate bbox around prediction (use last bbox size)
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

    def _associate_blobs_to_tracks(
        self,
        blobs: List[Dict],
        predictions: Dict[int, Dict]
    ) -> Tuple[List, List, List, List]:
        """Associate detected blobs to predicted track positions with strict safety checks.
        
        Safety constraints prevent unrealistic associations:
        - Hard distance limit (prevents cross-hotel jumps)
        - Velocity limit (bees don't teleport)
        - Cost threshold (basic association quality)
        
        Args:
            blobs: List of detected blobs
            predictions: Dictionary of predicted track positions
            
        Returns:
            Tuple of (matched_blobs, matched_tracks, unmatched_blobs, unmatched_tracks)
        """
        if not blobs or not predictions:
            return [], [], blobs, list(predictions.keys())
        
        # Build adaptive cost matrix
        blob_centroids = [b['centroid'] for b in blobs]
        track_ids = list(predictions.keys())
        
        cost_matrix = np.zeros((len(blobs), len(track_ids)))
        
        # Base threshold (scaled by tracker's scale factor)
        base_threshold = self.config.tracking.association_threshold(
            1920, 1080, self.config.hotel_box
        ) * self.scale_factor
        
        for i, blob in enumerate(blobs):
            blob_cent = blob['centroid']
            blob_area = blob.get('area', 0)
            
            for j, track_id in enumerate(track_ids):
                pred = predictions[track_id]
                track = pred['track']
                pred_cent = pred['centroid']
                
                # Get velocity from Kalman state [x, y, vx, vy]
                vx = float(track.kalman.statePost[2])
                vy = float(track.kalman.statePost[3])
                speed = np.sqrt(vx**2 + vy**2)
                
                # Calculate position difference
                dx = blob_cent[0] - pred_cent[0]
                dy = blob_cent[1] - pred_cent[1]
                euclidean_dist = np.sqrt(dx**2 + dy**2)
                
                # Compute adaptive directional cost
                cost = self._compute_adaptive_cost(
                    dx, dy, vx, vy, speed, euclidean_dist, 
                    blob_area, track, base_threshold
                )
                
                cost_matrix[i, j] = cost
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_blobs = []
        matched_tracks = []
        matched_indices = set()
        
        # STRICT FILTERING - prevent unrealistic associations
        # These are HARD limits that override cost-based matching
        MAX_DISTANCE = getattr(self.config.tracking, 'max_association_distance', 200) * self.scale_factor
        MAX_VELOCITY = getattr(self.config.tracking, 'max_bee_velocity', 50) * self.scale_factor
        
        logger.debug(f"Association safety limits: max_dist={MAX_DISTANCE:.1f}px, max_vel={MAX_VELOCITY:.1f}px/frame")
        
        # Filter matches by safety constraints
        for i, j in zip(row_ind, col_ind):
            cost = cost_matrix[i, j]
            
            # Check 1: Cost threshold
            if cost >= base_threshold:
                continue  # Cost too high
            
            blob = blobs[i]
            track_id = track_ids[j]
            track = predictions[track_id]['track']
            
            # Check 2: HARD DISTANCE LIMIT (prevents cross-hotel jumps)
            blob_cent = blob['centroid']
            track_cent = track.centroid  # Use ACTUAL position, not predicted
            distance = np.sqrt(
                (blob_cent[0] - track_cent[0])**2 + 
                (blob_cent[1] - track_cent[1])**2
            )
            
            if distance > MAX_DISTANCE:
                logger.debug(f"REJECTED blob-track match: distance {distance:.1f} > {MAX_DISTANCE:.1f}px (track {track_id})")
                continue
            
            # Check 3: VELOCITY SANITY CHECK (bees don't teleport)
            # Implied velocity = distance traveled in 1 frame
            implied_velocity = distance  # pixels/frame (assuming 1 frame delta)
            
            if implied_velocity > MAX_VELOCITY:
                logger.debug(f"REJECTED blob-track match: velocity {implied_velocity:.1f} > {MAX_VELOCITY:.1f}px/frame (track {track_id})")
                continue
            
            # Optional Check 4: ACCELERATION CHECK (prevent sudden speed changes)
            # Current velocity from Kalman
            current_speed = np.sqrt(
                float(track.kalman.statePost[2])**2 + 
                float(track.kalman.statePost[3])**2
            )
            
            # If current speed is low and implied velocity is high = suspicious
            MAX_ACCELERATION = getattr(self.config.tracking, 'max_bee_acceleration', 30) * self.scale_factor
            speed_change = abs(implied_velocity - current_speed)
            
            if speed_change > MAX_ACCELERATION:
                logger.debug(f"REJECTED blob-track match: acceleration {speed_change:.1f} > {MAX_ACCELERATION:.1f}px/frame² (track {track_id})")
                continue
            
            # All checks passed - valid match
            logger.debug(f"ACCEPTED blob-track match: dist={distance:.1f}px, vel={implied_velocity:.1f}px/frame, cost={cost:.1f} (track {track_id})")
            matched_blobs.append(blob)
            matched_tracks.append(track_id)
            matched_indices.add(i)
        
        # Unmatched
        unmatched_blobs = [b for i, b in enumerate(blobs) if i not in matched_indices]
        matched_track_ids = set(matched_tracks)
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_track_ids]
        
        # Log summary
        logger.debug(f"Association: {len(matched_blobs)} matched, {len(unmatched_blobs)} unmatched blobs, {len(unmatched_tracks)} unmatched tracks")
        
        return matched_blobs, matched_tracks, unmatched_blobs, unmatched_tracks
    
    def _compute_adaptive_cost(
        self,
        dx: float,
        dy: float,
        vx: float,
        vy: float,
        speed: float,
        euclidean_dist: float,
        blob_area: float,
        track: 'TrackState',
        base_threshold: float
    ) -> float:
        """Compute adaptive directional cost for blob-to-track association.
        
        Creates an elliptical search area based on velocity:
        - Major axis along velocity direction (allows forward motion)
        - Minor axis perpendicular (allows lateral drift)
        - Extended reverse search (catches 180° turns)
        - Speed-adaptive threshold (faster = larger search)
        
        Args:
            dx, dy: Position difference (blob - prediction)
            vx, vy: Velocity components
            speed: Magnitude of velocity
            euclidean_dist: Simple Euclidean distance
            blob_area: Area of the blob
            track: Track state object
            base_threshold: Base association threshold
            
        Returns:
            Weighted cost (lower = better match)
        """
        # Check if adaptive association is enabled
        if not self.config.tracking.adaptive_association:
            return euclidean_dist
        
        # Speed threshold for "moving" vs "stationary"
        speed_threshold = self.config.tracking.stationary_speed_threshold
        
        if speed < speed_threshold:
            # Stationary or slow-moving: use circular search
            return euclidean_dist
        
        # Normalize velocity direction
        vx_norm = vx / speed
        vy_norm = vy / speed
        
        # Decompose position difference into parallel and perpendicular components
        # Parallel: projection onto velocity direction
        parallel = dx * vx_norm + dy * vy_norm
        
        # Perpendicular: orthogonal distance
        perpendicular = abs(dx * vy_norm - dy * vx_norm)
        
        # Adaptive threshold based on speed
        # Faster motion → larger forward search area
        speed_factor = min(speed / 50.0, self.config.tracking.max_speed_factor)
        
        # Elliptical search parameters from config
        forward_threshold = base_threshold * (1.0 + speed_factor)  # Major axis (forward)
        lateral_threshold = base_threshold * self.config.tracking.lateral_search_ratio  # Minor axis
        reverse_threshold = base_threshold * self.config.tracking.reverse_search_ratio  # Reverse
        
        # Check direction of displacement relative to velocity
        if parallel > 0:
            # Forward direction (expected)
            # Use elliptical distance: (parallel/a)² + (perpendicular/b)²
            normalized_dist = np.sqrt(
                (parallel / forward_threshold) ** 2 +
                (perpendicular / lateral_threshold) ** 2
            )
            cost = normalized_dist * base_threshold
            
        elif parallel < 0:
            # Reverse direction (180° turn)
            # Use reverse cone - more lenient than forward but not as much
            reverse_parallel = abs(parallel)
            normalized_dist = np.sqrt(
                (reverse_parallel / reverse_threshold) ** 2 +
                (perpendicular / lateral_threshold) ** 2
            )
            # Add penalty for reverse motion (prefer forward matches)
            cost = normalized_dist * base_threshold * self.config.tracking.reverse_motion_penalty
            
        else:
            # Pure lateral motion
            cost = perpendicular
        
        # Feature similarity bonus (if available)
        if self.config.tracking.area_similarity_weight > 0 and hasattr(track, 'bbox') and blob_area > 0:
            track_area = (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])
            if track_area > 0:
                area_ratio = min(blob_area, track_area) / max(blob_area, track_area)
                # Apply area similarity bonus (reduce cost)
                if area_ratio > 0.5:  # Similar size
                    similarity_bonus = 1.0 - (self.config.tracking.area_similarity_weight * (area_ratio - 0.5) * 2)
                    cost *= similarity_bonus
        
        return cost
    
    def _run_yolo(
        self,
        frame_roi: np.ndarray,
        config: Config,
        x_offset: int,
        y_offset: int
    ) -> List[Dict]:
        """Run YOLO detection on frame.
        
        Args:
            frame_roi: ROI portion of frame
            config: Configuration
            x_offset: X offset for ROI
            y_offset: Y offset for ROI
            
        Returns:
            List of YOLO detections
        """
        results = self.model.predict(
            frame_roi,
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
                
                detection = {
                    'bbox': (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset),
                    'centroid': (cx + x_offset, cy + y_offset),
                    'class_id': class_id,
                    'species': self.label_map.get(class_id, f'class_{class_id}')
                }
                detections.append(detection)
        
        return detections


    def _update_tracks_with_yolo(
        self,
        yolo_detections: List[Dict],
        unmatched_blobs: List[Dict],
        unmatched_tracks: List[int],
        frame_num: int
    ):
        """Update tracks with YOLO confirmations with strict safety checks.
        
        Three-stage process:
        1. Match unmatched blobs to YOLO detections → create new tracks
        2. Match unmatched tracks to YOLO detections → update existing tracks
        3. Resurrect struggling tracks OR create new tracks for remaining YOLO detections
        
        Safety checks prevent:
        - Cross-hotel track jumps (distance + velocity limits)
        - Track fragmentation (resurrection before new track creation)
        - Unrealistic teleportation (velocity constraints)
        
        Args:
            yolo_detections: YOLO detection results
            unmatched_blobs: Blobs without track association
            unmatched_tracks: Tracks without blob association
            frame_num: Current frame number
        """
        # Track which YOLO detections have been used to prevent double-assignment
        used_yolo_detections = set()
        
        # =================================================================
        # CASE 1: For unmatched blobs, try to match with YOLO detections
        # =================================================================
        for blob in unmatched_blobs:
            # Only consider blobs with valid aspect ratio
            if not self._is_valid_blob_shape(blob['bbox']):
                continue
            
            best_match = None
            best_dist = float('inf')
            best_idx = None
            
            # Find closest unused YOLO detection
            for idx, det in enumerate(yolo_detections):
                if idx in used_yolo_detections:
                    continue
                
                dist = np.sqrt(
                    (blob['centroid'][0] - det['centroid'][0]) ** 2 +
                    (blob['centroid'][1] - det['centroid'][1]) ** 2
                )
                
                if dist < best_dist:
                    best_dist = dist
                    best_match = det
                    best_idx = idx
            
            # STRICT validation for NEW track creation (using scaled thresholds)
            if best_match and best_dist < self.new_track_distance_threshold:
                # Check if there's already a track nearby (prevent duplicates)
                has_nearby_track = False
                for track_id, track in self.tracks.items():
                    track_dist = np.sqrt(
                        (track.centroid[0] - best_match['centroid'][0]) ** 2 +
                        (track.centroid[1] - best_match['centroid'][1]) ** 2
                    )
                    if track_dist < self.new_track_proximity_check:
                        has_nearby_track = True
                        logger.debug(f"Frame {frame_num}: Blob+YOLO match skipped - track {track_id} nearby ({track_dist:.1f}px)")
                        break
                
                # Only create new track if no nearby tracks exist
                if not has_nearby_track:
                    logger.debug(f"Frame {frame_num}: Creating new track (blob+YOLO) at {best_match['centroid']}")
                    self._create_track(best_match, frame_num)
                    used_yolo_detections.add(best_idx)
                else:
                    logger.debug(f"Frame {frame_num}: Skipping new track creation - nearby track exists")
        
        # =================================================================
        # CASE 2: For unmatched tracks, try to find with YOLO
        # =================================================================
        # CRITICAL: This handles fast turns when FG detection fails
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            best_match = None
            best_cost = float('inf')
            best_idx = None
            
            # Use adaptive association if enabled
            track_centroid = track.centroid
            
            # Get velocity from track
            vx = float(track.kalman.statePost[2])
            vy = float(track.kalman.statePost[3])
            speed = np.sqrt(vx**2 + vy**2)
            
            # Find best unused YOLO detection for this track
            for idx, det in enumerate(yolo_detections):
                if idx in used_yolo_detections:
                    continue
                
                # Calculate position difference
                dx = det['centroid'][0] - track_centroid[0]
                dy = det['centroid'][1] - track_centroid[1]
                euclidean_dist = np.sqrt(dx**2 + dy**2)
                
                # Use adaptive cost computation
                cost = self._compute_adaptive_cost(
                    dx, dy, vx, vy, speed, euclidean_dist,
                    0, track, self.config.tracking.association_threshold_base
                )
                
                if cost < best_cost:
                    best_cost = cost
                    best_match = det
                    best_idx = idx
            
            # Use association threshold (scaled by tracker's scale factor)
            max_threshold = self.config.tracking.association_threshold(
                1920, 1080, self.config.hotel_box
            ) * self.scale_factor
            
            if best_match and best_cost < max_threshold:
                # Update track with YOLO confirmation
                track.bbox = best_match['bbox']
                track.centroid = best_match['centroid']
                track.species = best_match['species']
                track.frames_without_detection = 0
                track.last_yolo_confirmation = frame_num
                
                # Add to trajectory history (keep last 30 positions)
                track.trajectory_history.append((frame_num, best_match['centroid']))
                if len(track.trajectory_history) > 30:
                    track.trajectory_history.pop(0)
                
                # Update Kalman filter
                measurement = np.array([[best_match['centroid'][0]], [best_match['centroid'][1]]], dtype=np.float32)
                track.kalman.correct(measurement)
                
                used_yolo_detections.add(best_idx)
                logger.debug(f"Frame {frame_num}: YOLO confirmed track {track_id} at {best_match['centroid']} (cost={best_cost:.1f})")
            else:
                # Track not found, increment age
                track.frames_without_detection += 1
                if best_match:
                    logger.debug(f"Frame {frame_num}: Track {track_id} - YOLO detection too far (cost={best_cost:.1f} > {max_threshold:.1f})")
                else:
                    logger.debug(f"Frame {frame_num}: Track {track_id} - no YOLO detections available")
        
        # =================================================================
        # CASE 3: Handle remaining YOLO detections with SAFE RESURRECTION
        # =================================================================
        # For YOLO detections that haven't been matched to blobs or tracks,
        # try to resurrect a nearby struggling track BEFORE creating a new one
        # BUT with strict safety checks to prevent cross-hotel jumps
        
        # Safety thresholds (same as blob association)
        MAX_RESURRECTION_DISTANCE = getattr(self.config.tracking, 'max_association_distance', 200) * self.scale_factor
        MAX_VELOCITY = getattr(self.config.tracking, 'max_bee_velocity', 50) * self.scale_factor
        
        for idx, det in enumerate(yolo_detections):
            if idx in used_yolo_detections:
                continue
            
            # Find the closest existing track (regardless of state)
            best_existing_track = None
            best_existing_track_id = None
            best_existing_dist = float('inf')
            
            for track_id, track in self.tracks.items():
                track_dist = np.sqrt(
                    (track.centroid[0] - det['centroid'][0]) ** 2 +
                    (track.centroid[1] - det['centroid'][1]) ** 2
                )
                
                if track_dist < best_existing_dist:
                    best_existing_dist = track_dist
                    best_existing_track = track
                    best_existing_track_id = track_id
            
            # Initial check: Is there a track within resurrection range?
            RESURRECTION_SEARCH_DISTANCE = self.new_track_proximity_check * 2.5  # Wide search
            
            if best_existing_track and best_existing_dist < RESURRECTION_SEARCH_DISTANCE:
                # Found a nearby track - now apply STRICT SAFETY CHECKS
                
                # SAFETY CHECK 1: Hard distance limit (prevent cross-hotel jumps)
                if best_existing_dist > MAX_RESURRECTION_DISTANCE:
                    logger.debug(f"Frame {frame_num}: Resurrection REJECTED for track {best_existing_track_id} - "
                            f"distance {best_existing_dist:.1f}px > {MAX_RESURRECTION_DISTANCE:.1f}px (creating new track)")
                    # Too far - create new track instead
                    logger.debug(f"Frame {frame_num}: Creating new track (resurrection blocked by distance)")
                    self._create_track(det, frame_num)
                    used_yolo_detections.add(idx)
                    continue
                
                # SAFETY CHECK 2: Velocity sanity (prevent teleportation)
                implied_velocity = best_existing_dist  # pixels/frame (1 frame delta)
                
                if implied_velocity > MAX_VELOCITY:
                    logger.debug(f"Frame {frame_num}: Resurrection REJECTED for track {best_existing_track_id} - "
                            f"velocity {implied_velocity:.1f}px/frame > {MAX_VELOCITY:.1f}px/frame (creating new track)")
                    # Too fast - create new track instead
                    logger.debug(f"Frame {frame_num}: Creating new track (resurrection blocked by velocity)")
                    self._create_track(det, frame_num)
                    used_yolo_detections.add(idx)
                    continue
                
                # SAFETY CHECK 3: Acceleration check (prevent sudden speed changes)
                current_speed = np.sqrt(
                    float(best_existing_track.kalman.statePost[2])**2 + 
                    float(best_existing_track.kalman.statePost[3])**2
                )
                speed_change = abs(implied_velocity - current_speed)
                MAX_ACCELERATION = getattr(self.config.tracking, 'max_bee_acceleration', 30) * self.scale_factor
                
                if speed_change > MAX_ACCELERATION:
                    logger.debug(f"Frame {frame_num}: Resurrection REJECTED for track {best_existing_track_id} - "
                            f"acceleration {speed_change:.1f}px/frame² > {MAX_ACCELERATION:.1f}px/frame² (creating new track)")
                    # Acceleration too high - create new track instead
                    logger.debug(f"Frame {frame_num}: Creating new track (resurrection blocked by acceleration)")
                    self._create_track(det, frame_num)
                    used_yolo_detections.add(idx)
                    continue
                
                # ALL SAFETY CHECKS PASSED - Safe to resurrect!
                logger.info(f"Frame {frame_num}: 🔄 RESURRECTING track {best_existing_track_id} "
                        f"(dist={best_existing_dist:.1f}px, vel={implied_velocity:.1f}px/frame, "
                        f"aged {best_existing_track.frames_without_detection} frames)")
                
                # Update track state
                best_existing_track.bbox = det['bbox']
                best_existing_track.centroid = det['centroid']
                best_existing_track.species = det['species']
                best_existing_track.frames_without_detection = 0  # Reset age counter
                best_existing_track.last_yolo_confirmation = frame_num
                
                # Update trajectory history
                best_existing_track.trajectory_history.append((frame_num, det['centroid']))
                if len(best_existing_track.trajectory_history) > 30:
                    best_existing_track.trajectory_history.pop(0)
                
                # Update Kalman filter with new measurement
                measurement = np.array([[det['centroid'][0]], [det['centroid'][1]]], dtype=np.float32)
                best_existing_track.kalman.correct(measurement)
                
                # Mark this YOLO detection as used
                used_yolo_detections.add(idx)
                
            else:
                # No nearby track found - safe to create a NEW track
                # This YOLO detection represents a genuinely new bee
                
                if best_existing_track:
                    logger.debug(f"Frame {frame_num}: Creating new track (YOLO-only) at {det['centroid']} "
                            f"(nearest track {best_existing_track_id} is {best_existing_dist:.1f}px away)")
                else:
                    logger.debug(f"Frame {frame_num}: Creating new track (YOLO-only) at {det['centroid']} "
                            f"(no existing tracks)")
                
                self._create_track(det, frame_num)
                used_yolo_detections.add(idx)
        
        # Log summary statistics
        total_yolo = len(yolo_detections)
        used_yolo = len(used_yolo_detections)
        unused_yolo = total_yolo - used_yolo
        
        if unused_yolo > 0:
            logger.warning(f"Frame {frame_num}: {unused_yolo}/{total_yolo} YOLO detections unused (possible issue?)")








    
    def _update_matched_tracks(
        self,
        matched_blobs: List[Dict],
        matched_tracks: List[int],
        frame_num: int
    ):
        """Update tracks that were matched to blobs.
        
        Args:
            matched_blobs: Blobs that were matched
            matched_tracks: Track IDs that were matched
            frame_num: Current frame number
        """
        for blob, track_id in zip(matched_blobs, matched_tracks):
            track = self.tracks[track_id]
            
            # Update track
            track.bbox = blob['bbox']
            track.centroid = blob['centroid']
            track.frames_without_detection = 0
            track.age += 1
            
            # Add to trajectory history (keep last 30 positions)
            track.trajectory_history.append((frame_num, blob['centroid']))
            if len(track.trajectory_history) > 30:
                track.trajectory_history.pop(0)
            
            # Update Kalman filter
            measurement = np.array([[blob['centroid'][0]], [blob['centroid'][1]]], dtype=np.float32)
            track.kalman.correct(measurement)
    
    def _create_track(self, detection: Dict, frame_num: int):
        """Create new track from detection.
        
        Args:
            detection: Detection dictionary
            frame_num: Current frame number
        """
        # Initialize Kalman filter
        kalman = cv2.KalmanFilter(4, 2)  # 4 state vars (x, y, vx, vy), 2 measurements (x, y)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * self.config.tracking.kalman_process_noise
        
        # Initialize state
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
            species=detection.get('species', 'unknown'),
            age=1,
            last_yolo_confirmation=frame_num
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
    
    def _age_tracks(self, frame_num: int):
        """Age out tracks without recent detections.
        
        Args:
            frame_num: Current frame number
        """
        to_remove = []
        
        for track_id, track in self.tracks.items():
            track.frames_without_detection += 1
            
            if track.frames_without_detection > self.config.tracking.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def _get_current_detections(self, frame_num: int, debug_data: Optional[Dict] = None) -> List[Dict]:
        """Get current detections from active tracks with optional debug data.
        
        Args:
            frame_num: Current frame number
            debug_data: Optional dict with 'blobs' and 'yolo_detections' for visualization
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        for track_id, track in self.tracks.items():
            detection = {
                'frame': frame_num,
                'track_id': track_id,
                'x1': track.bbox[0],
                'y1': track.bbox[1],
                'x2': track.bbox[2],
                'y2': track.bbox[3],
                'species': track.species
            }
            
            # Add debug data if provided (for visualization)
            if debug_data is not None:
                detection['debug_blobs'] = debug_data.get('blobs', [])
                detection['debug_yolo'] = debug_data.get('yolo_detections', [])
            
            detections.append(detection)
        
        return detections
    
    def _visualize_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        is_low_res: bool,
        blobs: List[Dict] = None,
        yolo_detections: List[Dict] = None
    ) -> np.ndarray:
        """Visualize frame with tracks, blobs, YOLO detections, and trajectories.
        
        Args:
            frame: Frame to annotate
            frame_num: Frame number
            is_low_res: Whether in low-res mode
            blobs: Optional list of FG/BG blobs to visualize
            yolo_detections: Optional list of YOLO detections to visualize
            
        Returns:
            Annotated frame
        """
        viz_frame = frame.copy()
        
        # Draw FG/BG blobs (cyan boxes)
        if blobs:
            for blob in blobs:
                x1, y1, x2, y2 = [int(c) for c in blob['bbox']]
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Cyan
                # Draw blob centroid
                cx, cy = [int(c) for c in blob['centroid']]
                cv2.circle(viz_frame, (cx, cy), 2, (255, 255, 0), -1)
        
        # Draw YOLO detections (orange boxes)
        if yolo_detections:
            for det in yolo_detections:
                x1, y1, x2, y2 = [int(c) for c in det['bbox']]
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
                label = f"DL:{det.get('species', 'unknown')}"
                cv2.putText(
                    viz_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1
                )
        
        # Draw tracks with trajectory lines
        for track_id, track in self.tracks.items():
            x1, y1, x2, y2 = [int(c) for c in track.bbox]
            
            # Draw trajectory line (last 30 positions)  #  I think we should only draw active tracks not all tracks
            if len(track.trajectory_history) > 1:
                for i in range(len(track.trajectory_history) - 1):
                    pt1 = (int(track.trajectory_history[i][1][0]), int(track.trajectory_history[i][1][1]))
                    pt2 = (int(track.trajectory_history[i+1][1][0]), int(track.trajectory_history[i+1][1][1]))
                    cv2.line(viz_frame, pt1, pt2, (255, 255, 0), 1)  # Cyan trajectory
            
            # Color based on frames since YOLO confirmation
            frames_since_yolo = frame_num - track.last_yolo_confirmation
            if frames_since_yolo == 0:
                color = (0, 255, 0)  # Green = YOLO confirmed
            elif frames_since_yolo < 10:
                color = (0, 255, 255)  # Yellow = Recent YOLO
            else:
                color = (255, 0, 0)  # Blue = FG tracking only
            
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track_id} {track.species}"
            cv2.putText(
                viz_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Draw mode indicator
        # mode_text = "LOW-RES" if is_low_res else "HIGH-RES (HyDaT)"
        # mode_color = (255, 0, 0) if is_low_res else (0, 255, 0)
        
        # cv2.putText(
        #     viz_frame, f"Mode: {mode_text}", (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2
        # )
        
        # cv2.putText(
        #     viz_frame, f"Frame: {frame_num}", (10, 60),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        # )
        
        # cv2.putText(
        #     viz_frame, f"Tracks: {len(self.tracks)}", (10, 90),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        # )
        
        # Draw legend
        legend_y = 120
        cv2.putText(viz_frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz_frame, "Green=DL confirmed", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(viz_frame, "Yellow=Recent DL", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(viz_frame, "Blue=FG only", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(viz_frame, "Cyan=Blobs/Traj", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(viz_frame, "Orange=DL detections", (10, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        return viz_frame


# Backward compatibility aliases
AdaptiveMotionDetector = HyDaTTracker
HybridMotionDetector = HyDaTTracker
MotionDetector = HyDaTTracker