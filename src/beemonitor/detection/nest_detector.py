"""Nest detection and processing module.

This module handles detecting nest holes in bee hotel videos and processing
them to identify individual nest locations with IDs.
"""

import logging
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import math

from beemonitor.core.config import Config
from beemonitor.utils.geometry import remove_overlapping_points


logger = logging.getLogger(__name__)

# Type aliases
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class NestDetector:
    """Detector for bee hotel nests.
    
    This class handles nest detection and processing, including:
    - Running YOLO detection on video frames
    - Clustering detections into rows and columns
    - Fixing missing nest holes
    - Assigning unique IDs to each nest
    
    Attributes:
        model: YOLO model for nest detection
        config: Configuration object
    
    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO("models/nest_model.pt")
        >>> detector = NestDetector(model, config)
        >>> detections = detector.detect_nests("video.mp4", 720, 1280)
        >>> nests = detector.process_detections("video.mp4", detections, 720, 1280)
    """
    
    def __init__(self, model, config: Optional[Config] = None):
        """Initialize NestDetector.
        
        Args:
            model: YOLO model for nest detection
            config: Configuration object (optional)
        """
        self.model = model
        self.config = config if config is not None else Config.default()
    
    def detect_nests(
        self,
        video_path: str,
        res_height: int,
        res_width: int
    ) -> pd.DataFrame:
        """Detect nests in video frames.
        
        Processes frames from the video to find nest holes using YOLO detection.
        Continues until sufficient detections are found (typically ~60).
        
        Args:
            video_path: Path to video file
            res_height: Target frame height
            res_width: Target frame width
            
        Returns:
            DataFrame with columns: frame, coordinates, state, confidence
            
        Example:
            >>> detections = detector.detect_nests("video.mp4", 720, 1280)
            >>> print(f"Found {len(detections.iloc[0]['coordinates'])} nests")
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_counter = 0
        nest_detections = [[]]
        nest_state = []
        frames = []
        confs = []
        
        min_detections = self.config.nest.min_detections
        frame_skip = self.config.nest.frame_skip
        confidence_threshold = self.config.nest.confidence_threshold
        
        logger.info(f"Starting nest detection (target: {min_detections} detections)")
        
        while len(nest_detections[0]) < min_detections:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
            
            # Read frame
            success, frame = cap.read()
            
            if not success:
                logger.warning(f"Could not read frame {frame_counter}, stopping detection")
                break
            
            # Resize frame
            frame = cv2.resize(frame, (res_width, res_height))
            
            logger.debug(f"Processing frame {frame_counter}")
            
            # Run YOLO inference
            results = self.model.predict(frame, conf=confidence_threshold, verbose=False)
            
            # Extract detections
            boxes = results[0].boxes.xyxy.tolist()
            boxes = [(x, y, x1, y1) for (x, y, x1, y1) in boxes]
            nest_detections = [boxes]
            
            # Extract labels
            labels = results[0].boxes.cls.tolist()
            nest_state = [labels]
            
            # Extract confidence scores
            conf = results[0].boxes.conf.tolist()
            confs = [conf]
            
            frames = [frame_counter]
            
            # Skip frames
            frame_counter += frame_skip
        
        cap.release()
        
        # Save first frame for visualization
        self._save_frame(video_path, frame, 0)
        
        logger.info(f"Detected {len(nest_detections[0])} nests in frame {frames[0]}")
        
        nest_df = pd.DataFrame({
            'frame': frames,
            'coordinates': nest_detections,
            'state': nest_state,
            'confidence': confs
        })
        
        return nest_df
    
    def process_detections(
        self,
        video_path: str,
        nest_detection: pd.DataFrame,
        res_height: int,
        res_width: int
    ) -> Dict:
        """Process nest detections to identify individual nest holes.
        
        Takes raw nest detections and processes them to:
        1. Extract nest hole coordinates
        2. Cluster into rows
        3. Fix missing holes
        4. Assign unique IDs
        5. Calculate hotel ROI
        
        Args:
            video_path: Path to video file
            nest_detection: DataFrame from detect_nests
            res_height: Frame height
            res_width: Frame width
            
        Returns:
            Dictionary with keys:
                - 'hotel': Tuple (x1, y1, x2, y2) for hotel ROI
                - 'nests': Dict mapping nest IDs to bounding boxes
                
        Example:
            >>> nests = detector.process_detections("video.mp4", detections, 720, 1280)
            >>> print(f"Hotel ROI: {nests['hotel']}")
            >>> print(f"Found {len(nests['nests'])} individual nests")
        """
        logger.info("Processing nest detections...")
        
        # Extract nest coordinates
        nest_coords = self._get_nest_coordinates(nest_detection)
        logger.debug(f"Extracted {len(nest_coords)} nest coordinates")
        
        # Cluster into rows
        dl_rows = self._cluster_points(
            nest_coords,
            row_threshold=self.config.nest.row_threshold,
            col_threshold=self.config.nest.col_threshold
        )
        logger.debug(f"Clustered into {len(dl_rows)} rows")
        
        # Get reference points for hole fixing
        rows_10 = [row for row in dl_rows if len(row) == 10]
        if rows_10:
            hole_first = self._get_average_x([x[0][0] for x in rows_10])
            hole_last = self._get_average_x([x[-1][0] for x in rows_10])
        else:
            # Use all rows if no perfect rows found
            hole_first = min(min(row, key=lambda p: p[0])[0] for row in dl_rows)
            hole_last = max(max(row, key=lambda p: p[0])[0] for row in dl_rows)
        
        # Calculate average x distance
        x_distances = []
        for row in dl_rows:
            if len(row) > 1:
                x_distances.append(self._get_row_average_nest_distance(row))
        x_distance = self._get_average_x(x_distances) if x_distances else 50
        
        # Fix missing holes in rows
        fixed_dl_rows = []
        for row in dl_rows:
            fixed_row = self._fix_row_coords(
                row,
                hole_first,
                hole_last,
                x_average_width=x_distance
            )
            fixed_dl_rows.append(fixed_row)
        
        logger.debug(f"Fixed missing holes, now have {sum(len(r) for r in fixed_dl_rows)} nests")
        
        # Calculate hotel boundaries
        hole_top = self._get_average_x([x[1] for x in fixed_dl_rows[0]])
        hole_bottom = self._get_average_x([x[1] for x in fixed_dl_rows[-1]])
        
        # Hotel ROI with padding
        hx = max(0, hole_first - self.config.nest.hotel_padding_x)
        hy = max(0, hole_top - self.config.nest.hotel_padding_y)
        hx1 = min(res_width, hole_last + self.config.nest.hotel_padding_x)
        hy2 = min(res_height, hole_bottom + self.config.nest.hotel_padding_y)
        
        # Generate nest hole coordinates with IDs
        nest_ids = self._assign_nest_ids(fixed_dl_rows)
        
        result = {
            "hotel": (hx, hy, hx1, hy2),
            "nests": nest_ids
        }
        
        logger.info(f"Processed {len(nest_ids)} nests in hotel ROI")
        
        # Save visualization
        self._save_visualization(video_path, result, res_height, res_width)
        
        return result
    
    def _get_nest_coordinates(
        self,
        nest: pd.DataFrame,
        index: int = 0
    ) -> List[Point]:
        """Extract nest hole midpoints from detection DataFrame.
        
        Args:
            nest: DataFrame with nest detections
            index: Row index to use (default: 0)
            
        Returns:
            List of (x, y) midpoints
        """
        def get_midpoint(nest_coords):
            x1, y1, x2, y2 = nest_coords
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            return (int(midpoint_x), int(midpoint_y))
        
        states = nest.iloc[index]['state']
        coordinates = nest.iloc[index]['coordinates']
        
        # Filter for nest_hole class (2.0)
        nest_coords = []
        for i in range(len(states)):
            if states[i] == 2.0:
                nest_coords.append(coordinates[i])
        
        return [get_midpoint(nest_hole) for nest_hole in nest_coords]
    
    def _cluster_points(
        self,
        points: List[Point],
        row_threshold: int = 10,
        col_threshold: int = 10
    ) -> List[List[Point]]:
        """Cluster points into rows and columns.
        
        Args:
            points: List of (x, y) points
            row_threshold: Y-distance threshold for row clustering
            col_threshold: X-distance threshold for column clustering
            
        Returns:
            List of rows, where each row is a list of points
        """
        if not points:
            return []
        
        # Sort points by y-coordinate to cluster into rows
        points_sorted_y = sorted(points, key=lambda x: x[1])
        rows = []
        current_row = [points_sorted_y[0]]
        
        for point in points_sorted_y[1:]:
            if abs(point[1] - current_row[-1][1]) < row_threshold:
                current_row.append(point)
            else:
                rows.append(current_row)
                current_row = [point]
        rows.append(current_row)
        
        # Sort each row by x-coordinate
        for i in range(len(rows)):
            rows[i] = sorted(rows[i], key=lambda x: x[0])
        
        # Remove rows with too few points
        min_row_size = self.config.nest.min_row_size
        rows = [row for row in rows if len(row) >= min_row_size]
        
        return rows
    
    def _get_row_average_nest_distance(self, row: List[Point]) -> int:
        """Calculate average horizontal distance between nests in a row.
        
        Args:
            row: List of points in the row
            
        Returns:
            Average distance between consecutive nests
        """
        if len(row) < 2:
            return 0
        
        distances = np.diff([x[0] for x in row]).tolist()
        distances = sorted(distances)
        
        # Filter out outliers
        filtered = []
        if distances:
            current = distances[0]
            for dist in distances:
                if abs(current - dist) < 10:
                    filtered.append(dist)
                current = dist
        
        return int(np.mean(filtered)) if filtered else int(np.mean(distances))
    
    def _get_average_x(self, nums: List[float]) -> int:
        """Calculate average of x-coordinates, filtering outliers.
        
        Args:
            nums: List of x-coordinates
            
        Returns:
            Average x-coordinate
        """
        if not nums:
            return 0
        
        nums = sorted(nums)
        
        filtered = []
        current = nums[0]
        for num in nums:
            if abs(current - num) < 10:
                filtered.append(num)
            current = num
        
        return int(np.mean(filtered)) if filtered else int(np.mean(nums))
    
    def _fix_row_coords(
        self,
        row: List[Point],
        first_hole_x: float,
        last_hole_x: float,
        pixel_threshold: int = 10,
        x_average_width: int = 0
    ) -> List[Point]:
        """Fix missing holes in a row by interpolating positions.
        
        Args:
            row: List of detected points in the row
            first_hole_x: X-coordinate of first hole
            last_hole_x: X-coordinate of last hole
            pixel_threshold: Threshold for position matching
            x_average_width: Average distance between holes
            
        Returns:
            List of points with missing holes filled in
        """
        if len(row) < 2:
            return row
        
        new_row_coords = []
        y_average = int(np.mean([x[1] for x in row]))
        
        # Check if first hole is missing
        if abs(first_hole_x - row[0][0]) > pixel_threshold:
            new_row_coords.append((first_hole_x, y_average))
        
        for i in range(len(row) - 1):
            current_hole = row[i]
            next_hole = row[i + 1]
            x_diff = next_hole[0] - current_hole[0]
            
            if abs(x_diff - x_average_width) < pixel_threshold:
                new_row_coords.append(current_hole)
                
                if i == len(row) - 2:  # Last pair
                    new_row_coords.append(next_hole)
            else:
                # Add current hole
                new_row_coords.append(current_hole)
                y_average = int((current_hole[1] + next_hole[1]) / 2)
                
                # Interpolate missing holes
                num_missing = math.ceil(x_diff / x_average_width)
                for j in range(num_missing - 1):
                    new_x = current_hole[0] + int(x_average_width * (j + 1))
                    new_row_coords.append((new_x, y_average))
        
        # Remove overlapping points
        new_row_coords = remove_overlapping_points(new_row_coords, threshold=30)
        
        # Check if last hole is missing
        if len(new_row_coords) < 10:
            new_row_coords.append((last_hole_x, y_average))
        
        return new_row_coords
    
    def _assign_nest_ids(self, rows: List[List[Point]]) -> Dict[str, BBox]:
        """Assign unique IDs to each nest and calculate bounding boxes.
        
        Args:
            rows: List of rows, each containing nest points
            
        Returns:
            Dictionary mapping nest IDs (str) to bounding boxes
        """
        width = self.config.nest.nest_width
        height = self.config.nest.nest_height
        pad_x = self.config.nest.padding_x
        pad_y = self.config.nest.padding_y
        
        nest_ids = {}
        
        for row_idx, row in enumerate(rows):
            sorted_row = sorted(row, key=lambda x: x[0])
            for col_idx, hole in enumerate(sorted_row):
                nest_id = str((col_idx + 1) + row_idx * 10)
                x, y = hole
                
                bbox = (
                    x - width // 2 - pad_x,
                    y - height // 2 - pad_y,
                    x + width // 2 + pad_x,
                    y + height // 2 + pad_y
                )
                
                nest_ids[nest_id] = bbox
        
        return nest_ids
    

    def _save_frame(self, video_path: str, frame, frame_number: int) -> None:
        """Save a frame as an image file with proper error handling.
        
        Args:
            video_path: Path to video file
            frame: Frame to save (numpy array)
            frame_number: Frame number for filename
        """
        # CRITICAL: Validate frame before attempting to save
        if frame is None:
            logger.warning(f"Cannot save frame {frame_number}: frame is None")
            return
        
        if not hasattr(frame, 'shape'):
            logger.warning(f"Cannot save frame {frame_number}: frame has no shape attribute")
            return
        
        if frame.size == 0:
            logger.warning(f"Cannot save frame {frame_number}: frame is empty")
            return
        
        try:
            output_folder = video_path.replace('.mp4', '_frames')
            filename = f"{output_folder}_frame_{frame_number:05d}.png"
            
            # Ensure parent directory exists
            from pathlib import Path
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Attempt to save
            success = cv2.imwrite(filename, frame)
            
            if success:
                logger.debug(f"Saved frame to {filename}")
            else:
                logger.warning(f"cv2.imwrite returned False for {filename}")
        
        except Exception as e:
            logger.error(f"Error saving frame {frame_number}: {e}")
            # Don't raise exception - just log and continue
    
    # def _save_frame(self, video_path: str, frame: np.ndarray, frame_number: int) -> None:
    #     """Save a frame as an image file.
        
    #     Args:
    #         video_path: Path to video file
    #         frame: Frame to save
    #         frame_number: Frame number for filename
    #     """
    #     output_folder = video_path.replace('.mp4', '_frames')
    #     filename = f"{output_folder}_frame_{frame_number:05d}.png"
    #     cv2.imwrite(filename, frame)
    #     logger.debug(f"Saved frame to {filename}")

    def _save_visualization(
        self,
        video_path: str,
        nest_ids: Dict,
        res_height: int,
        res_width: int
    ) -> None:
        return None
    
    # def _save_visualization(
    #     self,
    #     video_path: str,
    #     nest_ids: Dict,
    #     res_height: int,
    #     res_width: int
    # ) -> None:
    #     """Save visualization of detected nests.
        
    #     Args:
    #         video_path: Path to video file
    #         nest_ids: Dictionary with nest IDs and locations
    #         res_height: Frame height
    #         res_width: Frame width
    #     """
    #     # Load the saved frame
    #     frame_path = video_path.replace('.mp4', '_frames') + f"_frame_{0:05d}.png"
        
    #     try:
    #         frame = cv2.imread(frame_path)
            
    #         if frame is not None:
    #             # Draw nest IDs on frame
    #             for nest_id, bbox in nest_ids["nests"].items():
    #                 x1, y1, x2, y2 = bbox
    #                 cv2.rectangle(
    #                     frame,
    #                     (int(x1), int(y1)),
    #                     (int(x2), int(y2)),
    #                     (0, 255, 0),
    #                     2
    #                 )
    #                 cv2.putText(
    #                     frame,
    #                     nest_id,
    #                     (int(x1), int(y1) - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.5,
    #                     (0, 255, 0),
    #                     2
    #                 )
                
    #             # Save annotated frame
    #             output_folder = video_path.replace('.mp4', '_annotated_frames')
    #             filename = f"{output_folder}_frame_{10000:05d}.png"
    #             cv2.imwrite(filename, frame)
    #             logger.info(f"Saved nest visualization to {filename}")
        
    #     except Exception as e:
    #         logger.warning(f"Could not save visualization: {e}")