# """Motion detection and tracking module.

# This module handles detecting motion in video frames and tracking bees
# through sequences of frames.
# """

# import logging
# from typing import Dict, List, Tuple, Optional
# import cv2
# import numpy as np
# import pandas as pd
# import os
# import traceback

# from beemonitor.core.config import Config
# from beemonitor.tracking.bee_tracker import BeeTracker


# logger = logging.getLogger(__name__)

# # Type aliases
# BBox = Tuple[float, float, float, float]


# class MotionDetector:
#     """Detector for motion and bee tracking.
    
#     This class handles:
#     - Frame differencing for motion detection
#     - YOLO-based bee detection
#     - Bee tracking across frames
#     - Integration of motion and detection
    
#     Attributes:
#         model: YOLO model for bee detection
#         config: Configuration object
    
#     Example:
#         >>> detector = MotionDetector(model, config)
#         >>> results = detector.detect_and_track("video.mp4", hotel_roi, 720, 1280)
#     """
    
#     def __init__(self, model, config: Optional[Config] = None):
#         """Initialize MotionDetector.
        
#         Args:
#             model: YOLO model for bee detection
#             config: Configuration object (optional)
#         """
#         self.model = model
#         self.config = config if config is not None else Config.default()
    
#     def detect_and_track(
#         self,
#         video_path: str,
#         site_roi: BBox,
#         res_height: int,
#         res_width: int,
#         visualize: bool = False,
#         output_folder: str = "output"
#     ) -> pd.DataFrame:
#         """Detect motion and track bees in video.
        
#         Main method that orchestrates the entire motion detection and
#         tracking pipeline.
        
#         Args:
#             video_path: Path to video file
#             site_roi: Region of interest (x1, y1, x2, y2)
#             res_height: Target frame height
#             res_width: Target frame width
#             visualize: Whether to save visualization video
#             output_folder: Directory for output files
            
#         Returns:
#             DataFrame with columns: frame_number, tracks, detections
            
#         Example:
#             >>> detector = MotionDetector(model, config)
#             >>> roi = (100, 100, 500, 500)
#             >>> results = detector.detect_and_track("video.mp4", roi, 720, 1280)
#         """
#         # Ensure output folder exists
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
        
#         cap = cv2.VideoCapture(video_path)
        
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")
        
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_num = 0
        
#         frames = []
#         tracks = []
#         tracking_detections = []
#         track_id = 0
        
#         # Set up video output if visualizing
#         output_video = None
#         if visualize:
#             output_video = self._setup_video_output(
#                 video_path,
#                 output_folder,
#                 res_width,
#                 res_height
#             )
        
#         logger.info(f"Starting motion detection and tracking ({total_frames} frames)")
        
#         while frame_num < total_frames:
#             # Step 1: Detect motion
#             motion_frame = self._detect_motion(
#                 cap,
#                 frame_num,
#                 res_height,
#                 res_width,
#                 site_roi,
#                 visualize,
#                 output_video,
#                 output_folder
#             )
            
#             if motion_frame is None:
#                 break
            
#             frame_num = motion_frame
            
#             # Step 2: Track bees in the sequence
#             activity_start = max(0, frame_num - 5)
            
#             if activity_start + 5 > total_frames:
#                 frame_num = activity_start + 5
#                 track = []
#                 tracking_detection = []
#             else:
#                 track, frame_num, tracking_detection = self._detect_and_track_sequence(
#                     cap,
#                     activity_start,
#                     res_height,
#                     res_width,
#                     site_roi,
#                     visualize,
#                     output_video,
#                     output_folder,
#                     track_id
#                 )
            
#             activity_end = frame_num
            
#             # Avoid overlapping frames
#             if activity_end < motion_frame:
#                 frame_num = activity_end = motion_frame + 6
#                 track = []
            
#             frames.append((activity_start, activity_end))
#             tracks.append(track)
#             tracking_detections.append(tracking_detection)
            
#             track_id += len(track)
        
#         # Clean up
#         if visualize and output_video is not None:
#             output_video.release()
        
#         cap.release()
#         cv2.destroyAllWindows()
        
#         logger.info(f"Motion detection complete: {len(frames)} activity periods")
        
#         df = pd.DataFrame({
#             'frame_number': frames,
#             'tracks': tracks,
#             'detections': tracking_detections
#         })
        
#         return df
    
#     def _detect_motion(
#         self,
#         cap: cv2.VideoCapture,
#         frame_num: int,
#         res_height: int,
#         res_width: int,
#         site_roi: BBox,
#         visualize: bool = False,
#         video_output: Optional[cv2.VideoWriter] = None,
#         output_folder: Optional[str] = None
#     ) -> Optional[int]:
#         """Detect motion in video starting from frame_num.
        
#         Args:
#             cap: Video capture object
#             frame_num: Starting frame number
#             res_height: Frame height
#             res_width: Frame width
#             site_roi: Region of interest
#             visualize: Whether to save frames
#             video_output: Video writer for output
#             output_folder: Output directory
            
#         Returns:
#             Frame number where confirmed motion detected, or None if video ends
#         """
#         try:
#             # Set frame position
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
#             # Read initial frame
#             ret, frame = cap.read()
            
#             if not ret:
#                 logger.debug("No more frames to read")
#                 return None
            
#             # Resize and extract ROI
#             frame = cv2.resize(frame, (res_width, res_height))
#             x1, y1, x2, y2 = [int(c) for c in site_roi]
#             frame_roi = frame[y1:y2, x1:x2]
#             prev_frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            
#             while True:
#                 frame_num += 1
                
#                 # Read next frame
#                 ret, frame = cap.read()
                
#                 if not ret:
#                     break
                
#                 # Resize and extract ROI
#                 frame = cv2.resize(frame, (res_width, res_height))
#                 frame_roi = frame[y1:y2, x1:x2]
#                 frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
                
#                 # Detect motion
#                 frame_contours, motion_frame = self._detect_motion_on_frame(
#                     frame.copy(),
#                     prev_frame_gray,
#                     frame_roi_gray,
#                     site_roi,
#                     visualize,
#                     output_folder,
#                     frame_num
#                 )
                
#                 # Update previous frame
#                 prev_frame_gray = frame_roi_gray
                
#                 # If motion detected, confirm with object detection
#                 if len(frame_contours) > 0:
#                     _, labels, inference_frame = self._run_inference_on_frame(
#                         frame.copy(),
#                         frame_roi,
#                         site_roi,
#                         visualize,
#                         output_folder,
#                         frame_num
#                     )
                    
#                     if len(labels) > 0:
#                         # Confirmed detection
#                         if visualize and video_output is not None:
#                             cv2.rectangle(inference_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
#                             video_output.write(inference_frame)
#                         return frame_num
#                     else:
#                         # False positive, continue
#                         if visualize and video_output is not None:
#                             cv2.rectangle(inference_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
#                             video_output.write(inference_frame)
#                 else:
#                     if visualize and video_output is not None:
#                         cv2.rectangle(motion_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
#                         video_output.write(motion_frame)
            
#             return frame_num
        
#         except Exception as e:
#             logger.error(f"Error in motion detection at frame {frame_num}: {e}")
#             return frame_num + 1
    
#     def _detect_and_track_sequence(
#         self,
#         cap: cv2.VideoCapture,
#         frame_num: int,
#         res_height: int,
#         res_width: int,
#         site_roi: BBox,
#         visualize: bool = False,
#         video_output: Optional[cv2.VideoWriter] = None,
#         output_folder: Optional[str] = None,
#         track_id: int = 0
#     ) -> Tuple[List, int, Dict]:
#         """Track bees through a sequence of frames.
        
#         Args:
#             cap: Video capture object
#             frame_num: Starting frame number
#             res_height: Frame height
#             res_width: Frame width
#             site_roi: Region of interest
#             visualize: Whether to save frames
#             video_output: Video writer
#             output_folder: Output directory
#             track_id: Starting track ID
            
#         Returns:
#             Tuple of (tracks, final_frame_num, detections_dict)
#         """
#         try:
#             # Set frame position (start slightly before motion)
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            
#             ret, frame = cap.read()
            
#             if not ret:
#                 logger.debug("Cannot read tracking start frame")
#                 return [], frame_num + 1, {}
            
#             # Resize frame
#             frame = cv2.resize(frame, (res_width, res_height))
            
#             # Initialize tracker
#             tracker = BeeTracker(
#                 max_age=self.config.tracking.max_age,
#                 distance_threshold=self.config.tracking.distance_threshold,
#                 association_threshold=self.config.tracking.association_threshold,
#                 track_start_id=track_id
#             )
            
#             no_motion_counter = 0
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_detections_dict = {}
            
#             x1, y1, x2, y2 = [int(c) for c in site_roi]
            
#             while frame_num < total_frames - 1:
#                 # Read frame
#                 ret, frame = cap.read()
                
#                 if not ret:
#                     break
                
#                 # Resize
#                 frame = cv2.resize(frame, (res_width, res_height))
#                 frame_roi = frame[y1:y2, x1:x2]
                
#                 # Run detection
#                 inference_frame = frame.copy()
#                 boxes, labels, frame = self._run_inference_on_frame(
#                     inference_frame,
#                     frame_roi,
#                     site_roi,
#                     visualize,
#                     output_folder,
#                     frame_num
#                 )
                
#                 # Update tracker
#                 tracked_objects = tracker.update(boxes, frame_num)
#                 frame_num_detections = len(boxes)
                
#                 # Store detections
#                 frame_detections_dict[frame_num] = boxes
                
#                 # Update no motion counter
#                 if frame_num_detections == 0:
#                     no_motion_counter += 1
#                 else:
#                     no_motion_counter = 0
                
#                 # Stop if no motion for too long
#                 if no_motion_counter > 30:
#                     break
                
#                 # Visualize if requested
#                 if visualize and video_output is not None:
#                     frame = self._visualize_tracking(frame, tracked_objects, site_roi)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     video_output.write(frame)
                
#                 frame_num += 1
            
#             return tracker.get_tracks(), frame_num, frame_detections_dict
        
#         except Exception as e:
#             logger.error(f"Error in tracking at frame {frame_num}: {e}")
#             traceback.print_exc()
#             return [], frame_num + 1, {}
    
#     def _detect_motion_on_frame(
#         self,
#         current_frame: np.ndarray,
#         prev_frame_gray: np.ndarray,
#         current_frame_gray: np.ndarray,
#         site_roi: BBox,
#         visualize: bool = False,
#         output_folder: Optional[str] = None,
#         frame_num: int = 0
#     ) -> Tuple[List[BBox], np.ndarray]:
#         """Detect motion in a single frame using frame differencing.
        
#         Args:
#             current_frame: Current color frame
#             prev_frame_gray: Previous frame in grayscale
#             current_frame_gray: Current frame in grayscale
#             site_roi: Region of interest
#             visualize: Whether to draw on frame
#             output_folder: Output directory
#             frame_num: Frame number
            
#         Returns:
#             Tuple of (contour_boxes, annotated_frame)
#         """
#         # Calculate frame difference
#         frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
        
#         # Apply thresholding
#         threshold = self.config.tracking.motion_threshold
#         _, thresholded_frame = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
#         # Find contours
#         contours, _ = cv2.findContours(
#             thresholded_frame,
#             cv2.RETR_EXTERNAL,
#             cv2.CHAIN_APPROX_SIMPLE
#         )
        
#         # Extract bounding boxes
#         frame_contours = []
#         x1, y1, x2, y2 = [int(c) for c in site_roi]
        
#         min_area = self.config.tracking.min_contour_area
#         aspect_min = self.config.tracking.aspect_ratio_min
#         aspect_max = self.config.tracking.aspect_ratio_max

#         for contour in contours:
#             if cv2.contourArea(contour) > min_area:
#                 (x, y, w, h) = cv2.boundingRect(contour)
#                 aspect_ratio = w / h
                
#                 if aspect_min < aspect_ratio < aspect_max:
#                     # Convert to global coordinates
#                     bbox = (x + x1, y + y1, x + w + x1, y + h + y1)
#                     frame_contours.append(bbox)
                    
#                     if visualize:
#                         cv2.rectangle(
#                             current_frame,
#                             (bbox[0], bbox[1]),
#                             (bbox[2], bbox[3]),
#                             (255, 0, 0),
#                             2
#                         )
        
#         return frame_contours, current_frame
    
#     def _run_inference_on_frame(
#         self,
#         current_frame: np.ndarray,
#         current_frame_roi: np.ndarray,
#         site_roi: BBox,
#         visualize: bool = False,
#         output_folder: Optional[str] = None,
#         frame_num: int = 0
#     ) -> Tuple[List[BBox], List[int], np.ndarray]:
#         """Run YOLO inference on a frame.
        
#         Args:
#             current_frame: Current frame
#             current_frame_roi: ROI of current frame
#             site_roi: Region of interest coordinates
#             visualize: Whether to draw on frame
#             output_folder: Output directory
#             frame_num: Frame number
            
#         Returns:
#             Tuple of (boxes, labels, annotated_frame)
#         """
#         # Run YOLO

#         iou_threshold = self.config.tracking.iou_threshold
#         results = self.model(current_frame, verbose=False, iou=iou_threshold)

#         # results = self.model.predict(
#         #     current_frame,
#         #     verbose=False,
#         #     classes=self.config.detection.tracking_classes,
#         #     iou=self.config.detection.iou_threshold
#         # )
                
#         # Extract detections
#         boxes = results[0].boxes.xywh.tolist()
#         labels = results[0].boxes.cls.tolist()
        
#         normalized_boxes = []
#         aspect_min = self.config.tracking.aspect_ratio_min
#         aspect_max = self.config.tracking.aspect_ratio_max
        
#         for x, y, w, h in boxes:
#             x, y, w, h = int(x), int(y), int(w), int(h)
#             aspect_ratio = w / h
            
#             if aspect_min < aspect_ratio < aspect_max:
#                 # Convert from xywh to xyxy
#                 bbox = (
#                     x - int(w / 2),
#                     y - int(h / 2),
#                     x + w - int(w / 2),
#                     y + h - int(h / 2)
#                 )
#                 normalized_boxes.append(bbox)
                
#                 if visualize:
#                     cv2.rectangle(
#                         current_frame,
#                         (bbox[0], bbox[1]),
#                         (bbox[2], bbox[3]),
#                         (0, 0, 255),
#                         2
#                     )
        
#         return normalized_boxes, labels, current_frame
    
#     def _visualize_tracking(
#         self,
#         frame: np.ndarray,
#         tracks: List[Tuple],
#         site_roi: BBox
#     ) -> np.ndarray:
#         """Draw tracking visualization on frame.
        
#         Args:
#             frame: Frame to draw on
#             tracks: List of (bbox, track_id) tuples
#             site_roi: Region of interest
            
#         Returns:
#             Annotated frame
#         """
#         for track in tracks:
#             bbox, track_id = track[0], track[1]
#             x0, y0, x1, y1 = [int(c) for c in bbox]
            
#             cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
#             cv2.putText(
#                 frame,
#                 f"Track {track_id}",
#                 (x0, y0),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 255),
#                 2
#             )
        
#         return frame
    
#     def _setup_video_output(
#         self,
#         video_path: str,
#         output_folder: str,
#         width: int,
#         height: int
#     ) -> cv2.VideoWriter:
#         """Set up video writer for output.
        
#         Args:
#             video_path: Input video path
#             output_folder: Output directory
#             width: Frame width
#             height: Frame height
            
#         Returns:
#             VideoWriter object
#         """
#         filename = os.path.basename(video_path).split('.')[0]
#         output_file = os.path.join(output_folder, f"processed_video_{filename}.mp4")
        
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         fps = self.config.video.fps
        
#         return cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# """Motion detection and tracking module with multi-species support.

# This module handles detecting motion in video frames and tracking multiple
# species of insects through sequences of frames.
# """

# import logging
# from typing import Dict, List, Tuple, Optional
# import cv2
# import numpy as np
# import pandas as pd
# import os
# import traceback

# from beemonitor.core.config import Config
# from beemonitor.multiple_object_tracking.bee_tracker import BeeTracker


# logger = logging.getLogger(__name__)

# # Type aliases
# BBox = Tuple[float, float, float, float]


# class MotionDetector:
#     """Detector for motion and multi-species tracking.
    
#     This class handles:
#     - Frame differencing for motion detection
#     - YOLO-based insect detection (multiple species)
#     - Multi-species tracking across frames
#     - Integration of motion and detection with species labels
    
#     Attributes:
#         model: YOLO model for insect detection
#         config: Configuration object
#         species_map: Mapping of class IDs to species names
    
#     Example:
#         >>> detector = MotionDetector(model, config)
#         >>> results = detector.detect_and_track("video.mp4", hotel_roi, 720, 1280)
#     """
    
#     def __init__(self, model, config: Optional[Config] = None):
#         """Initialize MotionDetector.
        
#         Args:
#             model: YOLO model for insect detection
#             config: Configuration object (optional)
#         """
#         self.model = model
#         self.config = config if config is not None else Config.default()
        
#         # Initialize species mapping from config
#         self.species_map = self.config.tracking.species_map
        
#         logger.info(f"Initialized MotionDetector with {len(self.species_map)} species classes")
#         for class_id, name in self.species_map.items():
#             logger.info(f"  Class {class_id}: {name}")
    
#     def detect_and_track(
#         self,
#         video_path: str,
#         site_roi: BBox,
#         res_height: int,
#         res_width: int,
#         visualize: bool = False,
#         output_folder: str = "output"
#     ) -> pd.DataFrame:
#         """Detect motion and track insects in video.
        
#         Main method that orchestrates the entire motion detection and
#         tracking pipeline with multi-species support.
        
#         Args:
#             video_path: Path to video file
#             site_roi: Region of interest (x1, y1, x2, y2)
#             res_height: Target frame height
#             res_width: Target frame width
#             visualize: Whether to save visualization video
#             output_folder: Directory for output files
            
#         Returns:
#             DataFrame with columns: frame_number, tracks, detections
#             Each track includes species information
            
#         Example:
#             >>> detector = MotionDetector(model, config)
#             >>> roi = (100, 100, 500, 500)
#             >>> results = detector.detect_and_track("video.mp4", roi, 720, 1280)
#             >>> # Access species info: results['tracks'][0][0]['species']
#         """
#         # Ensure output folder exists
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
        
#         cap = cv2.VideoCapture(video_path)
        
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")
        
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_num = 0
        
#         frames = []
#         tracks = []
#         tracking_detections = []
#         track_id = 0
        
#         # Set up video output if visualizing
#         output_video = None
#         if visualize:
#             output_video = self._setup_video_output(
#                 video_path,
#                 output_folder,
#                 res_width,
#                 res_height
#             )
        
#         logger.info(f"Starting motion detection and tracking ({total_frames} frames)")
        
#         while frame_num < total_frames:
#             # Step 1: Detect motion
#             motion_frame = self._detect_motion(
#                 cap,
#                 frame_num,
#                 res_height,
#                 res_width,
#                 site_roi,
#                 visualize,
#                 output_video,
#                 output_folder
#             )
            
#             if motion_frame is None:
#                 break
            
#             frame_num = motion_frame
            
#             # Step 2: Track insects in the sequence
#             activity_start = max(0, frame_num - 5)
            
#             if activity_start + 5 > total_frames:
#                 frame_num = activity_start + 5
#                 track = []
#                 tracking_detection = []
#             else:
#                 track, frame_num, tracking_detection = self._detect_and_track_sequence(
#                     cap,
#                     activity_start,
#                     res_height,
#                     res_width,
#                     site_roi,
#                     visualize,
#                     output_video,
#                     output_folder,
#                     track_id
#                 )
            
#             activity_end = frame_num
            
#             # Avoid overlapping frames
#             if activity_end < motion_frame:
#                 frame_num = activity_end = motion_frame + 6
#                 track = []
            
#             frames.append((activity_start, activity_end))
#             tracks.append(track)
#             tracking_detections.append(tracking_detection)
            
#             track_id += len(track)
        
#         # Clean up
#         if visualize and output_video is not None:
#             output_video.release()
        
#         cap.release()
#         cv2.destroyAllWindows()
        
#         logger.info(f"Motion detection complete: {len(frames)} activity periods")
        
#         df = pd.DataFrame({
#             'frame_number': frames,
#             'tracks': tracks,
#             'detections': tracking_detections
#         })
        
#         return df
    
#     def _detect_motion(
#         self,
#         cap: cv2.VideoCapture,
#         frame_num: int,
#         res_height: int,
#         res_width: int,
#         site_roi: BBox,
#         visualize: bool = False,
#         video_output: Optional[cv2.VideoWriter] = None,
#         output_folder: Optional[str] = None
#     ) -> Optional[int]:
#         """Detect motion in video starting from frame_num.
        
#         Args:
#             cap: Video capture object
#             frame_num: Starting frame number
#             res_height: Frame height
#             res_width: Frame width
#             site_roi: Region of interest
#             visualize: Whether to save frames
#             video_output: Video writer for output
#             output_folder: Output directory
            
#         Returns:
#             Frame number where confirmed motion detected, or None if video ends
#         """
#         try:
#             # Set frame position
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
#             # Read initial frame
#             ret, frame = cap.read()
            
#             if not ret:
#                 logger.debug("No more frames to read")
#                 return None
            
#             # Resize and extract ROI
#             frame = cv2.resize(frame, (res_width, res_height))
#             x1, y1, x2, y2 = [int(c) for c in site_roi]
#             frame_roi = frame[y1:y2, x1:x2]
#             prev_frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            
#             while True:
#                 frame_num += 1
                
#                 # Read next frame
#                 ret, frame = cap.read()
                
#                 if not ret:
#                     break
                
#                 # Resize and extract ROI
#                 frame = cv2.resize(frame, (res_width, res_height))
#                 frame_roi = frame[y1:y2, x1:x2]
#                 frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
                
#                 # Detect motion
#                 frame_contours, motion_frame = self._detect_motion_on_frame(
#                     frame.copy(),
#                     prev_frame_gray,
#                     frame_roi_gray,
#                     site_roi,
#                     visualize,
#                     output_folder,
#                     frame_num
#                 )
                
#                 # Update previous frame
#                 prev_frame_gray = frame_roi_gray
                
#                 # If motion detected, confirm with object detection
#                 if len(frame_contours) > 0:
#                     _, labels, inference_frame = self._run_inference_on_frame(
#                         frame.copy(),
#                         frame_roi,
#                         site_roi,
#                         visualize,
#                         output_folder,
#                         frame_num
#                     )
                    
#                     if len(labels) > 0:
#                         # Confirmed detection
#                         if visualize and video_output is not None:
#                             cv2.rectangle(inference_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
#                             video_output.write(inference_frame)
#                         return frame_num
#                     else:
#                         # False positive, continue
#                         if visualize and video_output is not None:
#                             cv2.rectangle(inference_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
#                             video_output.write(inference_frame)
#                 else:
#                     if visualize and video_output is not None:
#                         cv2.rectangle(motion_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
#                         video_output.write(motion_frame)
            
#             return frame_num
        
#         except Exception as e:
#             logger.error(f"Error in motion detection at frame {frame_num}: {e}")
#             return frame_num + 1
    
#     def _detect_and_track_sequence(
#         self,
#         cap: cv2.VideoCapture,
#         frame_num: int,
#         res_height: int,
#         res_width: int,
#         site_roi: BBox,
#         visualize: bool = False,
#         video_output: Optional[cv2.VideoWriter] = None,
#         output_folder: Optional[str] = None,
#         track_id: int = 0
#     ) -> Tuple[List, int, Dict]:
#         """Track insects through a sequence of frames with species tracking.
        
#         Args:
#             cap: Video capture object
#             frame_num: Starting frame number
#             res_height: Frame height
#             res_width: Frame width
#             site_roi: Region of interest
#             visualize: Whether to save frames
#             video_output: Video writer
#             output_folder: Output directory
#             track_id: Starting track ID
            
#         Returns:
#             Tuple of (tracks_with_species, final_frame_num, detections_dict)
#             Each track includes species information
#         """
#         try:
#             # Set frame position (start slightly before motion)
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            
#             ret, frame = cap.read()
            
#             if not ret:
#                 logger.debug("Cannot read tracking start frame")
#                 return [], frame_num + 1, {}
            
#             # Resize frame
#             frame = cv2.resize(frame, (res_width, res_height))
            
#             # Initialize tracker with species tracking
#             tracker = BeeTracker(
#                 max_age=self.config.tracking.max_age,
#                 distance_threshold=self.config.tracking.distance_threshold,
#                 association_threshold=self.config.tracking.association_threshold,
#                 track_start_id=track_id,
#                 track_species=True  # Enable species tracking
#             )
            
#             no_motion_counter = 0
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_detections_dict = {}
            
#             x1, y1, x2, y2 = [int(c) for c in site_roi]
            
#             while frame_num < total_frames - 1:
#                 # Read frame
#                 ret, frame = cap.read()
                
#                 if not ret:
#                     break
                
#                 # Resize
#                 frame = cv2.resize(frame, (res_width, res_height))
#                 frame_roi = frame[y1:y2, x1:x2]
                
#                 # Run detection with species
#                 inference_frame = frame.copy()
#                 boxes, labels, frame = self._run_inference_on_frame(
#                     inference_frame,
#                     frame_roi,
#                     site_roi,
#                     visualize,
#                     output_folder,
#                     frame_num
#                 )
                
#                 # Update tracker with species information
#                 tracked_objects = tracker.update(boxes, frame_num, species_labels=labels)
#                 frame_num_detections = len(boxes)
                
#                 # Store detections with species
#                 frame_detections_dict[frame_num] = {
#                     'boxes': boxes,
#                     'species': [self.species_map.get(int(label), 'unknown') for label in labels]
#                 }
                
#                 # Update no motion counter
#                 if frame_num_detections == 0:
#                     no_motion_counter += 1
#                 else:
#                     no_motion_counter = 0
                
#                 # Stop if no motion for too long
#                 if no_motion_counter > self.config.tracking.no_motion_frames:
#                     break
                
#                 # Visualize if requested
#                 if visualize and video_output is not None:
#                     frame = self._visualize_tracking(frame, tracked_objects, site_roi)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     video_output.write(frame)
                
#                 frame_num += 1
            
#             return tracker.get_tracks(), frame_num, frame_detections_dict
        
#         except Exception as e:
#             logger.error(f"Error in tracking at frame {frame_num}: {e}")
#             traceback.print_exc()
#             return [], frame_num + 1, {}
    
#     def _detect_motion_on_frame(
#         self,
#         current_frame: np.ndarray,
#         prev_frame_gray: np.ndarray,
#         current_frame_gray: np.ndarray,
#         site_roi: BBox,
#         visualize: bool = False,
#         output_folder: Optional[str] = None,
#         frame_num: int = 0
#     ) -> Tuple[List[BBox], np.ndarray]:
#         """Detect motion in a single frame using frame differencing.
        
#         Args:
#             current_frame: Current color frame
#             prev_frame_gray: Previous frame in grayscale
#             current_frame_gray: Current frame in grayscale
#             site_roi: Region of interest
#             visualize: Whether to draw on frame
#             output_folder: Output directory
#             frame_num: Frame number
            
#         Returns:
#             Tuple of (contour_boxes, annotated_frame)
#         """
#         # Get scaled parameters for current resolution
#         width = current_frame.shape[1]
#         height = current_frame.shape[0]
        
#         # Calculate frame difference
#         frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
        
#         # Apply thresholding
#         threshold = self.config.tracking.motion_threshold
#         _, thresholded_frame = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
#         # Find contours
#         contours, _ = cv2.findContours(
#             thresholded_frame,
#             cv2.RETR_EXTERNAL,
#             cv2.CHAIN_APPROX_SIMPLE
#         )
        
#         # Extract bounding boxes with scaled parameters
#         frame_contours = []
#         x1, y1, x2, y2 = [int(c) for c in site_roi]
        
#         min_area = self.config.tracking.min_contour_area(width, height)
#         aspect_min = self.config.tracking.aspect_ratio_min
#         aspect_max = self.config.tracking.aspect_ratio_max

#         for contour in contours:
#             if cv2.contourArea(contour) > min_area:
#                 (x, y, w, h) = cv2.boundingRect(contour)
#                 aspect_ratio = w / h
                
#                 if aspect_min < aspect_ratio < aspect_max:
#                     # Convert to global coordinates
#                     bbox = (x + x1, y + y1, x + w + x1, y + h + y1)
#                     frame_contours.append(bbox)
                    
#                     if visualize:
#                         cv2.rectangle(
#                             current_frame,
#                             (bbox[0], bbox[1]),
#                             (bbox[2], bbox[3]),
#                             (255, 0, 0),
#                             2
#                         )
        
#         return frame_contours, current_frame
    
#     def _run_inference_on_frame(
#         self,
#         current_frame: np.ndarray,
#         current_frame_roi: np.ndarray,
#         site_roi: BBox,
#         visualize: bool = False,
#         output_folder: Optional[str] = None,
#         frame_num: int = 0
#     ) -> Tuple[List[BBox], List[int], np.ndarray]:
#         """Run YOLO inference on a frame with multi-species detection.
        
#         Args:
#             current_frame: Current frame
#             current_frame_roi: ROI of current frame
#             site_roi: Region of interest coordinates
#             visualize: Whether to draw on frame
#             output_folder: Output directory
#             frame_num: Frame number
            
#         Returns:
#             Tuple of (boxes, species_labels, annotated_frame)
#         """
#         # Get tracking classes from config
#         tracking_classes = self.config.tracking.tracking_classes
#         iou_threshold = self.config.tracking.iou_threshold
        
#         # Run YOLO with specified classes
#         if tracking_classes:
#             results = self.model(
#                 current_frame,
#                 verbose=False,
#                 classes=tracking_classes,
#                 iou=iou_threshold
#             )
#         else:
#             results = self.model(
#                 current_frame,
#                 verbose=False,
#                 iou=iou_threshold
#             )
        
#         # Extract detections
#         boxes = results[0].boxes.xywh.tolist()
#         labels = results[0].boxes.cls.tolist()
        
#         normalized_boxes = []
#         species_labels = []
#         aspect_min = self.config.tracking.aspect_ratio_min
#         aspect_max = self.config.tracking.aspect_ratio_max
        
#         for (x, y, w, h), label in zip(boxes, labels):
#             x, y, w, h = int(x), int(y), int(w), int(h)
#             aspect_ratio = w / h
            
#             if aspect_min < aspect_ratio < aspect_max:
#                 # Convert from xywh to xyxy
#                 bbox = (
#                     x - int(w / 2),
#                     y - int(h / 2),
#                     x + w - int(w / 2),
#                     y + h - int(h / 2)
#                 )
#                 normalized_boxes.append(bbox)
#                 species_labels.append(int(label))
                
#                 if visualize:
#                     # Get species name for visualization
#                     species_name = self.species_map.get(int(label), 'unknown')
#                     color = self._get_species_color(int(label))
                    
#                     cv2.rectangle(
#                         current_frame,
#                         (bbox[0], bbox[1]),
#                         (bbox[2], bbox[3]),
#                         color,
#                         2
#                     )
#                     cv2.putText(
#                         current_frame,
#                         species_name,
#                         (bbox[0], bbox[1] - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         color,
#                         2
#                     )
        
#         return normalized_boxes, species_labels, current_frame
    
#     def _visualize_tracking(
#         self,
#         frame: np.ndarray,
#         tracks: List[Tuple],
#         site_roi: BBox
#     ) -> np.ndarray:
#         """Draw tracking visualization on frame with species labels.
        
#         Args:
#             frame: Frame to draw on
#             tracks: List of track tuples with species info
#             site_roi: Region of interest
            
#         Returns:
#             Annotated frame
#         """
#         for track in tracks:
#             bbox, track_id = track[0], track[1]
#             x0, y0, x1, y1 = [int(c) for c in bbox]
            
#             # Get species if available (track may have species attribute)
#             species = getattr(track, 'species', None) if len(track) > 2 else None
#             species_label = self.species_map.get(species, 'unknown') if species is not None else 'tracking'
            
#             # Get color based on species
#             color = self._get_species_color(species) if species is not None else (0, 255, 255)
            
#             cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
#             cv2.putText(
#                 frame,
#                 f"Track {track_id} ({species_label})",
#                 (x0, y0 - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 color,
#                 2
#             )
        
#         return frame
    
#     def _get_species_color(self, class_id: Optional[int]) -> Tuple[int, int, int]:
#         """Get visualization color for a species class.
        
#         Args:
#             class_id: Species class ID
            
#         Returns:
#             BGR color tuple
#         """
#         if class_id is None:
#             return (0, 255, 255)  # Yellow for unknown
        
#         # Color mapping for different species
#         color_map = {
#             0: (0, 255, 0),      # Green
#             1: (255, 0, 0),      # Blue
#             2: (0, 0, 255),      # Red
#             3: (255, 255, 0),    # Cyan
#             4: (255, 0, 255),    # Magenta
#             5: (0, 255, 255),    # Yellow
#             6: (128, 0, 128),    # Purple
#             7: (255, 128, 0),    # Orange
#             8: (0, 128, 255),    # Light Blue
#             9: (128, 255, 0),    # Lime
#         }
        
#         return color_map.get(class_id, (255, 255, 255))  # White for undefined
    
#     def _setup_video_output(
#         self,
#         video_path: str,
#         output_folder: str,
#         width: int,
#         height: int
#     ) -> cv2.VideoWriter:
#         """Set up video writer for output.
        
#         Args:
#             video_path: Input video path
#             output_folder: Output directory
#             width: Frame width
#             height: Frame height
            
#         Returns:
#             VideoWriter object
#         """
#         filename = os.path.basename(video_path).split('.')[0]
#         output_file = os.path.join(output_folder, f"processed_video_{filename}.mp4")
        
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         fps = self.config.video.fps
        
#         return cv2.VideoWriter(output_file, fourcc, fps, (width, height))












"""Motion detection and tracking module with multi-species support.

This module handles detecting motion in video frames and tracking multiple
species of insects through sequences of frames.
"""

import logging
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import os
import traceback

from beemonitor.core.config import Config
from beemonitor.multiple_object_tracking.bee_tracker import BeeTracker


logger = logging.getLogger(__name__)

# Type aliases
BBox = Tuple[float, float, float, float]


class MotionDetector:
    """Detector for motion and multi-species tracking.
    
    This class handles:
    - Frame differencing for motion detection
    - YOLO-based insect detection (multiple species)
    - Multi-species tracking across frames
    - Integration of motion and detection with species labels
    
    Attributes:
        model: YOLO model for insect detection
        config: Configuration object
        species_map: Mapping of class IDs to species names
    
    Example:
        >>> detector = MotionDetector(model, config)
        >>> results = detector.detect_and_track("video.mp4", hotel_roi, 720, 1280)
    """
    
    def __init__(self, model, config: Optional[Config] = None):
        """Initialize MotionDetector.
        
        Args:
            model: YOLO model for insect detection
            config: Configuration object (optional)
        """
        self.model = model
        self.config = config if config is not None else Config.default()
        
        # Initialize species mapping from config
        self.species_map = self.config.tracking.species_map
        
        logger.info(f"Initialized MotionDetector with {len(self.species_map)} species classes")
        for class_id, name in self.species_map.items():
            logger.info(f"  Class {class_id}: {name}")
    
    def detect_and_track(
        self,
        video_path: str,
        site_roi: BBox,
        res_height: int,
        res_width: int,
        visualize: bool = False,
        output_folder: str = "output",
        config: Optional[Config] = None
    ) -> pd.DataFrame:
        """Detect motion and track insects in video.
        
        Main method that orchestrates the entire motion detection and
        tracking pipeline with multi-species support.
        
        Args:
            video_path: Path to video file
            site_roi: Region of interest (x1, y1, x2, y2)
            res_height: Target frame height
            res_width: Target frame width
            visualize: Whether to save visualization video
            output_folder: Directory for output files
            
        Returns:
            DataFrame with columns: frame_number, tracks, detections
            Each track includes species information
            
        Example:
            >>> detector = MotionDetector(model, config)
            >>> roi = (100, 100, 500, 500)
            >>> results = detector.detect_and_track("video.mp4", roi, 720, 1280)
            >>> # Access species info: results['tracks'][0][0]['species']
        """
        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        
        frames = []
        tracks = []
        tracking_detections = []
        track_id = 0
        
        # Set up video output if visualizing
        output_video = None
        if visualize:
            output_video = self._setup_video_output(
                video_path,
                output_folder,
                res_width,
                res_height
            )
        
        logger.info(f"Starting motion detection and tracking ({total_frames} frames)")
        
        while frame_num < total_frames:
            # Step 1: Detect motion
            motion_frame = self._detect_motion(
                cap,
                frame_num,
                res_height,
                res_width,
                site_roi,
                visualize,
                output_video,
                output_folder
            )
            
            if motion_frame is None:
                break
            
            frame_num = motion_frame
            
            # Step 2: Track insects in the sequence
            activity_start = max(0, frame_num - 5)
            
            if activity_start + 5 > total_frames:
                frame_num = activity_start + 5
                track = []
                tracking_detection = []
            else:
                track, frame_num, tracking_detection = self._detect_and_track_sequence(
                    cap,
                    activity_start,
                    res_height,
                    res_width,
                    site_roi,
                    visualize,
                    output_video,
                    output_folder,
                    track_id, 
                    config= config
                )
            
            activity_end = frame_num
            
            # Avoid overlapping frames
            if activity_end < motion_frame:
                frame_num = activity_end = motion_frame + 6
                track = []
            
            frames.append((activity_start, activity_end))
            tracks.append(track)
            tracking_detections.append(tracking_detection)
            
            track_id += len(track)
        
        # Clean up
        if visualize and output_video is not None:
            output_video.release()
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Motion detection complete: {len(frames)} activity periods")
        
        df = pd.DataFrame({
            'frame_number': frames,
            'tracks': tracks,
            'detections': tracking_detections
        })
        
        return df
    
    def _detect_motion(
        self,
        cap: cv2.VideoCapture,
        frame_num: int,
        res_height: int,
        res_width: int,
        site_roi: BBox,
        visualize: bool = False,
        video_output: Optional[cv2.VideoWriter] = None,
        output_folder: Optional[str] = None
    ) -> Optional[int]:
        """Detect motion in video starting from frame_num.
        
        Args:
            cap: Video capture object
            frame_num: Starting frame number
            res_height: Frame height
            res_width: Frame width
            site_roi: Region of interest
            visualize: Whether to save frames
            video_output: Video writer for output
            output_folder: Output directory
            
        Returns:
            Frame number where confirmed motion detected, or None if video ends
        """
        try:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            # Read initial frame
            ret, frame = cap.read()
            
            if not ret:
                logger.debug("No more frames to read")
                return None
            
            # Resize and extract ROI
            frame = cv2.resize(frame, (res_width, res_height))
            x1, y1, x2, y2 = [int(c) for c in site_roi]
            frame_roi = frame[y1:y2, x1:x2]
            prev_frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            
            while True:
                frame_num += 1
                
                # Read next frame
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Resize and extract ROI
                frame = cv2.resize(frame, (res_width, res_height))
                frame_roi = frame[y1:y2, x1:x2]
                frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
                
                # Detect motion
                frame_contours, motion_frame = self._detect_motion_on_frame(
                    frame.copy(),
                    prev_frame_gray,
                    frame_roi_gray,
                    site_roi,
                    visualize,
                    output_folder,
                    frame_num
                )
                
                # Update previous frame
                prev_frame_gray = frame_roi_gray
                
                # If motion detected, confirm with object detection
                if len(frame_contours) > 0:
                    _, labels, inference_frame = self._run_inference_on_frame(
                        frame.copy(),
                        frame_roi,
                        site_roi,
                        visualize,
                        output_folder,
                        frame_num
                    )
                    
                    if len(labels) > 0:
                        # Confirmed detection
                        if visualize and video_output is not None:
                            cv2.rectangle(inference_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                            video_output.write(inference_frame)
                        return frame_num
                    else:
                        # False positive, continue
                        if visualize and video_output is not None:
                            cv2.rectangle(inference_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                            video_output.write(inference_frame)
                else:
                    if visualize and video_output is not None:
                        cv2.rectangle(motion_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        video_output.write(motion_frame)
            
            return frame_num
        
        except Exception as e:
            logger.error(f"Error in motion detection at frame {frame_num}: {e}")
            return frame_num + 1
    
    def _detect_and_track_sequence(
        self,
        cap: cv2.VideoCapture,
        frame_num: int,
        res_height: int,
        res_width: int,
        site_roi: BBox,
        visualize: bool = False,
        video_output: Optional[cv2.VideoWriter] = None,
        output_folder: Optional[str] = None,
        track_id: int = 0, 
        config: Optional[Config] = None
    ) -> Tuple[List, int, Dict]:
        """Track insects through a sequence of frames with species tracking.
        
        Args:
            cap: Video capture object
            frame_num: Starting frame number
            res_height: Frame height
            res_width: Frame width
            site_roi: Region of interest
            visualize: Whether to save frames
            video_output: Video writer
            output_folder: Output directory
            track_id: Starting track ID
            config: Configuration object (optional)
            
        Returns:
            Tuple of (tracks_with_species, final_frame_num, detections_dict)
            Each track includes species information
        """
        try:
            # Set frame position (start slightly before motion)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            
            ret, frame = cap.read()
            
            if not ret:
                logger.debug("Cannot read tracking start frame")
                return [], frame_num + 1, {}
            
            # Resize frame
            frame = cv2.resize(frame, (res_width, res_height))
            
            # Get scaled tracking parameters for this resolution
            distance_threshold = self.config.tracking.distance_threshold(res_width, res_height)
            association_threshold = self.config.tracking.association_threshold(res_width, res_height)
            
            # Initialize tracker with species tracking
            tracker = BeeTracker(
                max_age=self.config.tracking.max_age,
                distance_threshold=distance_threshold,
                association_threshold=association_threshold,
                track_start_id=track_id,
                track_species=True  # Enable species tracking
            )
            
            no_motion_counter = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_detections_dict = {}
            
            x1, y1, x2, y2 = [int(c) for c in site_roi]
            
            while frame_num < total_frames - 1:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Resize
                frame = cv2.resize(frame, (res_width, res_height))
                frame_roi = frame[y1:y2, x1:x2]
                
                # Run detection with species
                inference_frame = frame.copy()
                boxes, labels, frame = self._run_inference_on_frame(
                    inference_frame,
                    frame_roi,
                    site_roi,
                    visualize,
                    output_folder,
                    frame_num,
                    config=config
                )
                
                # Update tracker with species information
                tracked_objects = tracker.update(boxes, frame_num, species_labels=labels)
                frame_num_detections = len(boxes)
                
                # Store detections with species
                frame_detections_dict[frame_num] = {
                    'boxes': boxes,
                    'species': [self.species_map.get(int(label), 'unknown') for label in labels]
                }
                
                # Update no motion counter
                if frame_num_detections == 0:
                    no_motion_counter += 1
                else:
                    no_motion_counter = 0
                
                # Stop if no motion for too long
                if no_motion_counter > self.config.tracking.no_motion_frames:
                    break
                
                # Visualize if requested
                if visualize and video_output is not None:
                    frame = self._visualize_tracking(frame, tracked_objects, site_roi)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    video_output.write(frame)
                
                frame_num += 1
            
            return tracker.get_tracks(), frame_num, frame_detections_dict
        
        except Exception as e:
            logger.error(f"Error in tracking at frame {frame_num}: {e}")
            traceback.print_exc()
            return [], frame_num + 1, {}
    
    def _detect_motion_on_frame(
        self,
        current_frame: np.ndarray,
        prev_frame_gray: np.ndarray,
        current_frame_gray: np.ndarray,
        site_roi: BBox,
        visualize: bool = False,
        output_folder: Optional[str] = None,
        frame_num: int = 0
    ) -> Tuple[List[BBox], np.ndarray]:
        """Detect motion in a single frame using frame differencing.
        
        Args:
            current_frame: Current color frame
            prev_frame_gray: Previous frame in grayscale
            current_frame_gray: Current frame in grayscale
            site_roi: Region of interest
            visualize: Whether to draw on frame
            output_folder: Output directory
            frame_num: Frame number
            
        Returns:
            Tuple of (contour_boxes, annotated_frame)
        """
        # Get scaled parameters for current resolution
        width = current_frame.shape[1]
        height = current_frame.shape[0]
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
        
        # Apply thresholding
        threshold = self.config.tracking.motion_threshold
        _, thresholded_frame = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresholded_frame,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract bounding boxes with scaled parameters
        frame_contours = []
        x1, y1, x2, y2 = [int(c) for c in site_roi]
        
        min_area = self.config.tracking.min_contour_area(width, height)
        aspect_min = self.config.tracking.aspect_ratio_min
        aspect_max = self.config.tracking.aspect_ratio_max

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if aspect_min < aspect_ratio < aspect_max:
                    # Convert to global coordinates
                    bbox = (x + x1, y + y1, x + w + x1, y + h + y1)
                    frame_contours.append(bbox)
                    
                    if visualize:
                        cv2.rectangle(
                            current_frame,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (255, 0, 0),
                            2
                        )
        
        return frame_contours, current_frame
    
    def _run_classification_on_objects(
        self,
        boxes: List[BBox],
        labels: List[int],
        current_frame: np.ndarray,
        frame_num: int = 0,
        config: Optional[Config] = None
    ) -> Tuple[List[BBox], List[int], np.ndarray]:
        """Filter detections based on aspect ratio and visualize.

        1. Use the boxes to crop objects from the current frame.
        2. Run the classification model on each cropped object.
        3. Update the labels based on classification results.
        4. Return the boxes, updated labels, and frame

        Args:
            boxes: List of bounding boxes
            labels: List of class labels
            current_frame: Frame to draw on
        Returns:
            Tuple of (filtered_boxes, filtered_labels, annotated_frame)
        """

        model = self.config.models.bee_classification

        # If no classification model configured, return inputs unchanged
        if model is None:
            return boxes, labels, current_frame

        updated_boxes: List[BBox] = []
        updated_labels: List[int] = []

        # Confidence threshold from config (fallback to 0.5)
        conf_thresh = getattr(self.config.tracking, "classification_confidence_threshold", 0.5)

        h_frame, w_frame = current_frame.shape[:2]

        for i, bbox in enumerate(boxes):
            try:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                # clamp to image
                x1 = max(0, min(x1, w_frame - 1))
                x2 = max(0, min(x2, w_frame - 1))
                y1 = max(0, min(y1, h_frame - 1))
                y2 = max(0, min(y2, h_frame - 1))

                if x2 <= x1 or y2 <= y1:
                    # invalid box; skip but keep original
                    updated_boxes.append(bbox)
                    updated_labels.append(labels[i] if i < len(labels) else -1)
                    continue
                
                # crop object with padding
                #crop = current_frame[y1:y2, x1:x2]
                padding = 100
                crop = current_frame[max(0, y1 - padding):min(h_frame, y2 + padding),
                                     max(0, x1 - padding):min(w_frame, x2 + padding)]

                # save cropped image for debugging
                #cv2.imwrite(f"{self.config.output.base_folder}/debug_crop_{i}_{frame_num}.jpg", crop)

                # Run classification - try common call patterns
                pred_label = None
                pred_conf = 1.0

                try:
                    # If model has predict and expects batch
                    if hasattr(model, "predict"):
                        res = model.predict(np.asarray([crop]))
                    else:
                        res = model(crop)
                except Exception:
                    # fallback to calling with batch
                    try:
                        res = model(np.asarray([crop]))
                    except Exception:
                        res = None

                # Parse result heuristically
                if isinstance(res, np.ndarray):
                    probs = res[0] if res.ndim > 1 else res
                    pred_idx = int(np.argmax(probs))
                    pred_conf = float(np.max(probs))
                    pred_label = pred_idx
                elif isinstance(res, (list, tuple)) and len(res) > 0:
                    first = res[0]
                    # YOLO-like output
                    if hasattr(first, "boxes"):
                        try:
                            cls_list = getattr(first.boxes, "cls").tolist()
                            if len(cls_list) > 0:
                                pred_label = int(cls_list[0])
                                pred_conf = 1.0
                        except Exception:
                            pass
                    elif isinstance(first, dict):
                        if "label" in first:
                            pred_label = int(first["label"])
                            pred_conf = float(first.get("confidence", 1.0))
                        elif "labels" in first:
                            lbls = first["labels"]
                            scores = first.get("scores", None)
                            pred_label = int(lbls[0]) if len(lbls) > 0 else None
                            pred_conf = float(scores[0]) if scores is not None and len(scores) > 0 else 1.0
                    elif isinstance(first, (np.ndarray, list)):
                        arr = np.asarray(first)
                        if arr.ndim == 1:
                            pred_label = int(np.argmax(arr))
                            pred_conf = float(np.max(arr))
                        elif arr.ndim > 1:
                            pred_label = int(np.argmax(arr[0]))
                            pred_conf = float(np.max(arr[0]))
                elif isinstance(res, dict):
                    if "label" in res:
                        pred_label = int(res["label"])
                        pred_conf = float(res.get("confidence", 1.0))
                    elif "scores" in res and "labels" in res:
                        pred_label = int(res["labels"][0])
                        pred_conf = float(res["scores"][0])

                # If parsing failed, fallback to original label
                if pred_label is None:
                    chosen_label = labels[i] if i < len(labels) else -1
                else:
                    # update label if confidence is acceptable, otherwise keep original
                    if pred_conf >= conf_thresh:
                        chosen_label = int(pred_label)
                    else:
                        chosen_label = labels[i] if i < len(labels) else int(pred_label)

                # Append results
                updated_boxes.append(bbox)
                updated_labels.append(chosen_label)

                # # Visualization: draw predicted species/name and confidence
                # try:
                #     species_name = self.species_map.get(int(chosen_label), str(chosen_label))
                #     color = self._get_species_color(int(chosen_label)) if chosen_label is not None else (0, 255, 255)
                #     cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
                #     cv2.putText(
                #         current_frame,
                #         f"{species_name} {pred_conf:.2f}",
                #         (x1, max(0, y1 - 6)),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.45,
                #         color,
                #         1,
                #         cv2.LINE_AA,
                #     )
                # except Exception:
                #     # silent on visualization errors
                #     pass

            except Exception as ex:
                logger.exception(f"Error classifying object {i}: {ex}")
                # keep original if something goes wrong
                updated_boxes.append(bbox)
                updated_labels.append(labels[i] if i < len(labels) else -1)

        return updated_boxes, updated_labels, current_frame




        pass
    
    def _run_inference_on_frame(
        self,
        current_frame: np.ndarray,
        current_frame_roi: np.ndarray,
        site_roi: BBox,
        visualize: bool = False,
        output_folder: Optional[str] = None,
        frame_num: int = 0, 
        config: Optional[Config] = None 
    ) -> Tuple[List[BBox], List[int], np.ndarray]:
        """Run YOLO inference on a frame with multi-species detection.
        
        Args:
            current_frame: Current frame
            current_frame_roi: ROI of current frame
            site_roi: Region of interest coordinates
            visualize: Whether to draw on frame
            output_folder: Output directory
            frame_num: Frame number
            
        Returns:
            Tuple of (boxes, species_labels, annotated_frame)
        """
        # Get tracking classes from config
        tracking_classes = self.config.tracking.tracking_classes
        iou_threshold = self.config.tracking.iou_threshold
        
        # Run YOLO with specified classes
        if tracking_classes:
            results = self.model(
                current_frame,
                verbose=False,
                classes=tracking_classes,
                iou=iou_threshold
            )
        else:
            results = self.model(
                current_frame,
                verbose=False,
                iou=iou_threshold
            )
        
        # Extract detections
        boxes = results[0].boxes.xywh.tolist()
        labels = results[0].boxes.cls.tolist()
        
        normalized_boxes = []
        species_labels = []
        aspect_min = self.config.tracking.aspect_ratio_min
        aspect_max = self.config.tracking.aspect_ratio_max
        
        for (x, y, w, h), label in zip(boxes, labels):
            x, y, w, h = int(x), int(y), int(w), int(h)
            aspect_ratio = w / h
            
            if aspect_min < aspect_ratio < aspect_max:
                # Convert from xywh to xyxy
                bbox = (
                    x - int(w / 2),
                    y - int(h / 2),
                    x + w - int(w / 2),
                    y + h - int(h / 2)
                )
                normalized_boxes.append(bbox)
                species_labels.append(int(label))
                
                if visualize:
                    # Get species name for visualization
                    species_name = self.species_map.get(int(label), 'unknown')
                    color = self._get_species_color(int(label))
                    
                    cv2.rectangle(
                        current_frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        color,
                        2
                    )
                    cv2.putText(
                        current_frame,
                        species_name,
                        (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

        # normalized_boxes, species_labels, current_frame = self._run_classification_on_objects(
        #     normalized_boxes,
        #     species_labels,
        #     current_frame, 
        #     frame_num= frame_num,
        #     config=config
        # )
        
        return normalized_boxes, species_labels, current_frame
    
    def _visualize_tracking(
        self,
        frame: np.ndarray,
        tracks: List[Tuple],
        site_roi: BBox
    ) -> np.ndarray:
        """Draw tracking visualization on frame with species labels.
        
        Args:
            frame: Frame to draw on
            tracks: List of track tuples with species info
            site_roi: Region of interest
            
        Returns:
            Annotated frame
        """
        for track in tracks:
            bbox, track_id = track[0], track[1]
            x0, y0, x1, y1 = [int(c) for c in bbox]
            
            # Get species if available (track may have species attribute)
            species = getattr(track, 'species', None) if len(track) > 2 else None
            species_label = self.species_map.get(species, 'unknown') if species is not None else 'tracking'
            
            # Get color based on species
            color = self._get_species_color(species) if species is not None else (0, 255, 255)
            
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                frame,
                f"Track {track_id} ({species_label})",
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return frame
    
    def _get_species_color(self, class_id: Optional[int]) -> Tuple[int, int, int]:
        """Get visualization color for a species class.
        
        Args:
            class_id: Species class ID
            
        Returns:
            BGR color tuple
        """
        if class_id is None:
            return (0, 255, 255)  # Yellow for unknown
        
        # Color mapping for different species
        color_map = {
            0: (0, 255, 0),      # Green
            1: (255, 0, 0),      # Blue
            2: (0, 0, 255),      # Red
            3: (255, 255, 0),    # Cyan
            4: (255, 0, 255),    # Magenta
            5: (0, 255, 255),    # Yellow
            6: (128, 0, 128),    # Purple
            7: (255, 128, 0),    # Orange
            8: (0, 128, 255),    # Light Blue
            9: (128, 255, 0),    # Lime
        }
        
        return color_map.get(class_id, (255, 255, 255))  # White for undefined
    
    def _setup_video_output(
        self,
        video_path: str,
        output_folder: str,
        width: int,
        height: int
    ) -> cv2.VideoWriter:
        """Set up video writer for output.
        
        Args:
            video_path: Input video path
            output_folder: Output directory
            width: Frame width
            height: Frame height
            
        Returns:
            VideoWriter object
        """
        filename = os.path.basename(video_path).split('.')[0]
        output_file = os.path.join(output_folder, f"processed_video_{filename}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.config.video.fps
        
        return cv2.VideoWriter(output_file, fourcc, fps, (width, height))