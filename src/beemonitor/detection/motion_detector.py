"""Motion detection and tracking module.

This module handles detecting motion in video frames and tracking bees
through sequences of frames.
"""

import logging
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import os
import traceback

from beemonitor.core.config import Config
from beemonitor.tracking.bee_tracker import BeeTracker


logger = logging.getLogger(__name__)

# Type aliases
BBox = Tuple[float, float, float, float]


class MotionDetector:
    """Detector for motion and bee tracking.
    
    This class handles:
    - Frame differencing for motion detection
    - YOLO-based bee detection
    - Bee tracking across frames
    - Integration of motion and detection
    
    Attributes:
        model: YOLO model for bee detection
        config: Configuration object
    
    Example:
        >>> detector = MotionDetector(model, config)
        >>> results = detector.detect_and_track("video.mp4", hotel_roi, 720, 1280)
    """
    
    def __init__(self, model, config: Optional[Config] = None):
        """Initialize MotionDetector.
        
        Args:
            model: YOLO model for bee detection
            config: Configuration object (optional)
        """
        self.model = model
        self.config = config if config is not None else Config.default()
    
    def detect_and_track(
        self,
        video_path: str,
        site_roi: BBox,
        res_height: int,
        res_width: int,
        visualize: bool = False,
        output_folder: str = "output"
    ) -> pd.DataFrame:
        """Detect motion and track bees in video.
        
        Main method that orchestrates the entire motion detection and
        tracking pipeline.
        
        Args:
            video_path: Path to video file
            site_roi: Region of interest (x1, y1, x2, y2)
            res_height: Target frame height
            res_width: Target frame width
            visualize: Whether to save visualization video
            output_folder: Directory for output files
            
        Returns:
            DataFrame with columns: frame_number, tracks, detections
            
        Example:
            >>> detector = MotionDetector(model, config)
            >>> roi = (100, 100, 500, 500)
            >>> results = detector.detect_and_track("video.mp4", roi, 720, 1280)
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
            
            # Step 2: Track bees in the sequence
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
                    track_id
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
        track_id: int = 0
    ) -> Tuple[List, int, Dict]:
        """Track bees through a sequence of frames.
        
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
            
        Returns:
            Tuple of (tracks, final_frame_num, detections_dict)
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
            
            # Initialize tracker
            tracker = BeeTracker(
                max_age=self.config.tracking.max_age,
                distance_threshold=self.config.tracking.distance_threshold,
                association_threshold=self.config.tracking.association_threshold,
                track_start_id=track_id
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
                
                # Run detection
                inference_frame = frame.copy()
                boxes, labels, frame = self._run_inference_on_frame(
                    inference_frame,
                    frame_roi,
                    site_roi,
                    visualize,
                    output_folder,
                    frame_num
                )
                
                # Update tracker
                tracked_objects = tracker.update(boxes, frame_num)
                frame_num_detections = len(boxes)
                
                # Store detections
                frame_detections_dict[frame_num] = boxes
                
                # Update no motion counter
                if frame_num_detections == 0:
                    no_motion_counter += 1
                else:
                    no_motion_counter = 0
                
                # Stop if no motion for too long
                if no_motion_counter > 30:
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
        
        # Extract bounding boxes
        frame_contours = []
        x1, y1, x2, y2 = [int(c) for c in site_roi]
        
        min_area = self.config.tracking.min_contour_area
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
    
    def _run_inference_on_frame(
        self,
        current_frame: np.ndarray,
        current_frame_roi: np.ndarray,
        site_roi: BBox,
        visualize: bool = False,
        output_folder: Optional[str] = None,
        frame_num: int = 0
    ) -> Tuple[List[BBox], List[int], np.ndarray]:
        """Run YOLO inference on a frame.
        
        Args:
            current_frame: Current frame
            current_frame_roi: ROI of current frame
            site_roi: Region of interest coordinates
            visualize: Whether to draw on frame
            output_folder: Output directory
            frame_num: Frame number
            
        Returns:
            Tuple of (boxes, labels, annotated_frame)
        """
        # Run YOLO

        iou_threshold = self.config.tracking.iou_threshold
        _class = self.config.tracking.tracking_class
        results = self.model(current_frame, verbose=False, cls= _class, iou=iou_threshold)

        # results = self.model.predict(
        #     current_frame,
        #     verbose=False,
        #     classes=self.config.detection.tracking_classes,
        #     iou=self.config.detection.iou_threshold
        # )
                
        # Extract detections
        boxes = results[0].boxes.xywh.tolist()
        labels = results[0].boxes.cls.tolist()
        
        normalized_boxes = []
        aspect_min = self.config.tracking.aspect_ratio_min
        aspect_max = self.config.tracking.aspect_ratio_max
        
        for x, y, w, h in boxes:
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
                
                if visualize:
                    cv2.rectangle(
                        current_frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 0, 255),
                        2
                    )
        
        return normalized_boxes, labels, current_frame
    
    def _visualize_tracking(
        self,
        frame: np.ndarray,
        tracks: List[Tuple],
        site_roi: BBox
    ) -> np.ndarray:
        """Draw tracking visualization on frame.
        
        Args:
            frame: Frame to draw on
            tracks: List of (bbox, track_id) tuples
            site_roi: Region of interest
            
        Returns:
            Annotated frame
        """
        for track in tracks:
            bbox, track_id = track[0], track[1]
            x0, y0, x1, y1 = [int(c) for c in bbox]
            
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"Track {track_id}",
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
        
        return frame
    
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