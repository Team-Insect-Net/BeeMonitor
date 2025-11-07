"""YOLO-only motion tracking - Simple and reliable.

This version removes optical flow complexity and just uses YOLO tracking.
It's simpler, more reliable, and still quite fast with GPU.
"""

import logging
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import os

from beemonitor.core.config import Config
from beemonitor.multiple_object_tracking.bee_tracker import BeeTracker

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]


class MotionTracking:
    """Motion detector and tracker using YOLO only."""
    
    def __init__(self, model, config: Optional[Config] = None):
        self.model = model
        self.config = config if config is not None else Config.default()
        
        # Auto-detect GPU
        self.use_gpu = self._detect_gpu()
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, # put these into config?
            varThreshold=16, # put these into config? 
            detectShadows=True # same here
        )
        
        self.label_map = self.config.tracking.label_map
        self.tracking_classes = self.config.tracking.tracking_classes
        
        logger.info(f"Initialized MotionTracking (GPU: {self.use_gpu})")
    
    def _detect_gpu(self) -> bool:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
                return True
        except ImportError:
            pass
        
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
        config: Optional[Config] = None
    ) -> pd.DataFrame:
        """Track bees using YOLO only - simple and reliable."""
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing {total_frames} frames with YOLO tracking")
        
        # Initialize output
        output_video = None
        if visualize:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) # extract fps from input video itself
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_tracking_visualization.mp4") # to ensure that files are unique to video
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (res_width, res_height))
        
        frame_num = 0
        frames = []
        tracks = []
        tracking_detections = []
        track_id = 0
        
        # Process video
        while frame_num < total_frames:
            # Detect motion
            motion_frame = self._detect_motion_mog2(
                cap, frame_num, res_height, res_width, site_roi,
                visualize, output_video, output_folder
            )
            
            if motion_frame is None:
                break
            
            activity_start = motion_frame
            
            # Track sequence with YOLO
            track, activity_end, tracking_detection = self._track_sequence_yolo(
                cap, activity_start, res_height, res_width, site_roi,
                visualize, output_video, output_folder, track_id, config
            )
            
            frame_num = activity_end
            
            if activity_end < motion_frame:
                frame_num = activity_end = motion_frame + 6
                track = []
            
            frames.append((activity_start, activity_end))
            tracks.append(track)
            tracking_detections.append(tracking_detection)
            
            track_id += len(track)
        
        # Cleanup
        if visualize and output_video is not None:
            output_video.release()
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Tracking complete: {len(tracks)} track sequences")
        
        return pd.DataFrame({
            'frame_number': frames,
            'tracks': tracks,
            'detections': tracking_detections
        })
    
    def _detect_motion_mog2(
        self, cap, frame_num, res_height, res_width, site_roi,
        visualize=False, video_output=None, output_folder=None
    ):
        """Detect motion using MOG2 background subtraction."""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            x1, y1, x2, y2 = [int(c) for c in site_roi]
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    return None
                
                frame = cv2.resize(frame, (res_width, res_height))
                frame_roi = frame[y1:y2, x1:x2]
                
                # Apply background subtraction
                fg_mask = self.bg_subtractor.apply(frame_roi)
                _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                
                # Morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                min_area = self.config.tracking.min_contour_area(res_width, res_height)
                valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                
                if len(valid_contours) > 0:
                    # Confirm with YOLO
                    boxes, labels, _ = self._run_yolo(frame.copy(), frame_roi, site_roi, frame_num)
                    
                    if len(labels) > 0:
                        return frame_num
                
                if visualize and video_output is not None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    video_output.write(frame)
                
                frame_num += 1
        
        except Exception as e:
            logger.error(f"Error in motion detection: {e}")
            return frame_num + 1
    
    def _track_sequence_yolo(
        self, cap, frame_num, res_height, res_width, site_roi,
        visualize=False, video_output=None, output_folder=None,
        track_id=0, config=None
    ):
        """Track sequence using YOLO on every frame."""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
            
            if not ret:
                return [], frame_num + 1, {}
            
            frame = cv2.resize(frame, (res_width, res_height))
            x1, y1, x2, y2 = [int(c) for c in site_roi]
            
            # Get tracking parameters
            distance_threshold = self.config.tracking.distance_threshold(res_width, res_height)
            association_threshold = self.config.tracking.association_threshold(res_width, res_height)
            
            # Initialize tracker
            tracker = BeeTracker(
                max_age=self.config.tracking.max_age,
                distance_threshold=distance_threshold,
                association_threshold=association_threshold,
                track_start_id=track_id,
                track_species=True
            )
            
            frame_detections_dict = {}
            no_motion_counter = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while frame_num < total_frames - 1:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (res_width, res_height))
                frame_roi = frame[y1:y2, x1:x2]
                
                # Run YOLO on every frame
                boxes, labels, inference_frame = self._run_yolo(
                    frame.copy(), frame_roi, site_roi, frame_num, config
                )
                
                # Update tracker
                tracked_objects = tracker.update(boxes, frame_num, species_labels=labels)
                
                frame_detections_dict[frame_num] = {
                    'boxes': boxes,
                    'label': [self.label_map.get(int(label), 'unknown') for label in labels]
                }
                
                no_motion_counter = 0 if len(boxes) > 0 else no_motion_counter + 1
                
                # Stop if no motion for too long
                if no_motion_counter > self.config.tracking.no_motion_frames:
                    break
                
                # Visualize
                if visualize and video_output is not None:
                    for obj in tracked_objects:
                        bbox = obj['bbox']
                        tid = obj['track_id']
                        species = obj.get('species', 'unknown')
                        
                        x1_b, y1_b, x2_b, y2_b = [int(c) for c in bbox]
                        cv2.rectangle(frame, (x1_b, y1_b), (x2_b, y2_b), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{tid} {species}", (x1_b, y1_b-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    video_output.write(frame)
                
                frame_num += 1
            
            return tracker.get_tracks(), frame_num, frame_detections_dict
        
        except Exception as e:
            logger.error(f"Error in tracking: {e}")
            return [], frame_num + 1, {}
    
    def _run_yolo(self, frame, frame_roi, site_roi, frame_num, config=None):
        """Run YOLO inference."""
        if config is None:
            config = self.config
        
        results = self.model.predict(
            frame, 
            conf=config.tracking.confidence_threshold,
            iou=config.tracking.iou_threshold,
            verbose=False,
            device='0' if self.use_gpu else 'cpu' # use GPU if available
        )
        
        boxes = []
        labels = []
        x1_roi, y1_roi, x2_roi, y2_roi = [int(c) for c in site_roi]
        
        if len(results) > 0 and results[0].boxes is not None:
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                class_id = int(cls.cpu().numpy())
                
                if class_id not in self.tracking_classes: # only track specified classes
                    continue
                
                x1, y1, x2, y2 = box.cpu().numpy()
                #bbox = (x1 + x1_roi, y1 + y1_roi, x2 + x1_roi, y2 + y1_roi) # adjust to full frame coords if frame_roi used
                bbox = (x1, y1, x2, y2)
                
                boxes.append(bbox)
                labels.append(class_id)
        
        return boxes, labels, frame

