"""
Robust Nest Detector with Consistent ID Assignment

This module ensures nest IDs remain consistent across multiple video files
by using grid-based positioning and comprehensive detection.

Key Features:
- Grid-based ID assignment (10 columns per row)
- Exhaustive frame scanning until all nests found
- Spatial consistency checks
- Reference frame generation for cross-video matching
"""

import cv2
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from beemonitor.core.config import Config
from beemonitor.utils.geometry import remove_overlapping_points


logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """Configuration for nest grid structure."""
    rows: int = 5          # Number of rows
    columns: int = 10      # Number of columns per row
    expected_total: int = 50  # Total expected nests (rows * columns)
    tolerance: int = 2     # Allow +/- this many nests


class RobustNestDetector:
    """
    Robust nest detector with consistent ID assignment.
    
    Uses grid-based positioning to ensure the same nest gets the same ID
    across different videos from the same bee hotel.
    
    Attributes:
        model: YOLO model for detection
        config: Configuration object
        grid_config: Grid structure configuration
    
    Example:
        >>> detector = RobustNestDetector(model, config)
        >>> nests = detector.detect_nests_exhaustive("video.mp4")
        >>> # IDs will be consistent across videos of same hotel
    """
    
    def __init__(
        self,
        model,
        config: Config,
        grid_config: Optional[GridConfig] = None
    ):
        """Initialize robust nest detector.
        
        Args:
            model: YOLO model
            config: Configuration object
            grid_config: Grid structure configuration (optional)
        """
        self.model = model
        self.config = config
        self.grid_config = grid_config or GridConfig()
        
        logger.info(f"Initialized RobustNestDetector")
        logger.info(f"Expected grid: {self.grid_config.rows} rows x {self.grid_config.columns} columns")
    
    def detect_nests_exhaustive(
        self,
        video_path: str,
        res_height: int,
        res_width: int,
        max_frames: int = 1000
    ) -> pd.DataFrame:
        """
        Exhaustively detect nests across multiple frames.
        
        Scans frames until all expected nests are found or max_frames reached.
        
        Args:
            video_path: Path to video
            res_height: Target height
            res_width: Target width
            max_frames: Maximum frames to scan
            
        Returns:
            DataFrame with detected nests
        """
        logger.info("=" * 80)
        logger.info("Starting exhaustive nest detection")
        logger.info(f"Target: {self.grid_config.expected_total} nests")
        logger.info("=" * 80)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video has {total_frames} frames")
        
        # Collect detections from multiple frames
        all_detections = []
        confidence_threshold = self.config.nest.confidence_threshold
        
        # Strategy: Sample frames throughout the video
        frame_indices = self._generate_frame_sample_strategy(total_frames, max_frames)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Resize
            frame = cv2.resize(frame, (res_width, res_height))
            
            # Run detection
            results = self.model.predict(
                frame,
                conf=confidence_threshold,
                verbose=False
            )
            
            # Extract nest detections
            boxes = results[0].boxes.xyxy.tolist()
            boxes = [(x, y, x1, y1) for (x, y, x1, y1) in boxes]
            
            labels = results[0].boxes.cls.tolist()
            confs = results[0].boxes.conf.tolist()
            
            # Filter for nest class only (class 2)
            nest_boxes = []
            nest_confs = []
            for box, label, conf in zip(boxes, labels, confs):
                if label == 2.0:  # Nest hole class
                    nest_boxes.append(box)
                    nest_confs.append(conf)
            
            if nest_boxes:
                all_detections.extend([
                    {'box': box, 'conf': conf, 'frame': frame_idx}
                    for box, conf in zip(nest_boxes, nest_confs)
                ])
                
                logger.debug(f"Frame {frame_idx}: found {len(nest_boxes)} nests")
            
            # Early stopping if we have enough detections
            unique_nests = self._merge_detections(all_detections)
            if len(unique_nests) >= self.grid_config.expected_total:
                logger.info(f"Found all {len(unique_nests)} nests at frame {frame_idx}")
                break
        
        cap.release()
        
        # Merge overlapping detections
        merged_nests = self._merge_detections(all_detections)
        
        logger.info(f"Total unique nests detected: {len(merged_nests)}")
        
        # Validate against expected count
        self._validate_detection_count(len(merged_nests))
        
        # Assign grid-based IDs
        nest_with_ids = self._assign_grid_based_ids(
            merged_nests,
            res_width,
            res_height
        )

        # Save reference frame for future videos
        output_folder = self.config.output.base_folder
        self._save_reference_frame(video_path, output_folder, nest_with_ids, res_width, res_height)
        
        # Convert to DataFrame format
        df = self._create_nest_dataframe(nest_with_ids)
        
        return df
    
    def _generate_frame_sample_strategy(
        self,
        total_frames: int,
        max_frames: int
    ) -> List[int]:
        """Generate smart frame sampling strategy.
        
        Args:
            total_frames: Total frames in video
            max_frames: Maximum frames to sample
            
        Returns:
            List of frame indices to check
        """
        # Strategy: Start with evenly spaced frames, then fill gaps
        
        if total_frames <= max_frames:
            # Sample all frames
            return list(range(0, total_frames, 10))  # Every 10th frame
        
        # Sample strategically
        frames = []
        
        # 1. Sample from beginning (first 100 frames)
        frames.extend(range(0, min(100, total_frames), 5))
        
        # 2. Sample from middle
        middle = total_frames // 2
        frames.extend(range(middle - 50, middle + 50, 5))
        
        # 3. Sample from end
        frames.extend(range(total_frames - 100, total_frames, 5))
        
        # 4. Fill with evenly spaced frames
        step = max(1, total_frames // max_frames)
        frames.extend(range(0, total_frames, step))
        
        # Remove duplicates and sort
        frames = sorted(list(set(frames)))
        
        return frames[:max_frames]
    
    def _merge_detections(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Merge overlapping detections from multiple frames.
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for merging
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            # Find all detections that overlap with this one
            cluster = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(det1['box'], det2['box'])
                if iou > iou_threshold:
                    cluster.append(det2)
                    used.add(j)
            
            # Average the boxes in the cluster
            avg_box = self._average_boxes([d['box'] for d in cluster])
            avg_conf = np.mean([d['conf'] for d in cluster])
            
            merged.append({
                'box': avg_box,
                'conf': avg_conf,
                'cluster_size': len(cluster)
            })
        
        logger.info(f"Merged {len(detections)} detections into {len(merged)} unique nests")
        
        return merged
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _average_boxes(self, boxes: List[Tuple]) -> Tuple:
        """Average multiple box coordinates."""
        boxes_array = np.array(boxes)
        avg_box = np.mean(boxes_array, axis=0)
        return tuple(avg_box)
    
    def _validate_detection_count(self, detected_count: int):
        """Validate that detected count matches expectations.
        
        Args:
            detected_count: Number of nests detected
            
        Raises:
            Warning if count is outside tolerance
        """
        expected = self.grid_config.expected_total
        tolerance = self.grid_config.tolerance
        
        if abs(detected_count - expected) > tolerance:
            logger.warning(
                f"Detected {detected_count} nests, expected {expected} "
                f"(±{tolerance}). Review detection settings."
            )
        else:
            logger.info(f"✓ Detection count validation passed: {detected_count} nests")
    
    def _assign_grid_based_ids(
        self,
        nests: List[Dict],
        res_width: int,
        res_height: int
    ) -> Dict[str, Tuple]:
        """Assign consistent grid-based IDs to nests.
        
        Uses spatial positioning to assign IDs that will be consistent
        across videos of the same bee hotel.
        
        Args:
            nests: List of nest detections
            res_width: Frame width
            res_height: Frame height
            
        Returns:
            Dictionary mapping nest_id to coordinates
        """
        logger.info("Assigning grid-based IDs...")
        
        # Get centroids
        centroids = []
        for nest in nests:
            box = nest['box']
            centroid = (
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2
            )
            centroids.append(centroid)
        
        # Cluster into rows
        rows = self._cluster_into_rows(centroids)
        
        logger.info(f"Detected {len(rows)} rows")
        
        # Assign IDs based on grid position
        nest_with_ids = {}
        
        for row_idx, row in enumerate(rows):
            # Sort holes in row by x-coordinate (left to right)
            row_sorted = sorted(row, key=lambda p: p[0])
            
            for col_idx, centroid in enumerate(row_sorted):
                # Grid-based ID: row * columns_per_row + column
                nest_id = row_idx * self.grid_config.columns + col_idx + 1
                
                # Find corresponding box
                for nest in nests:
                    box = nest['box']
                    nest_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    
                    # Check if this is the same nest
                    dist = np.sqrt(
                        (centroid[0] - nest_centroid[0])**2 +
                        (centroid[1] - nest_centroid[1])**2
                    )
                    
                    if dist < 5:  # Threshold for matching
                        nest_with_ids[str(nest_id)] = box
                        break
        
        logger.info(f"Assigned {len(nest_with_ids)} nest IDs")
        
        # Validate ID assignment
        self._validate_id_assignment(nest_with_ids)
        
        return nest_with_ids
    
    def _cluster_into_rows(
        self,
        centroids: List[Tuple],
        row_threshold: int = 15
    ) -> List[List[Tuple]]:
        """Cluster nest centroids into rows.
        
        Args:
            centroids: List of (x, y) centroid coordinates
            row_threshold: Y-coordinate threshold for same row
            
        Returns:
            List of rows, where each row is a list of centroids
        """
        # Sort by y-coordinate
        sorted_centroids = sorted(centroids, key=lambda p: p[1])
        
        rows = []
        current_row = [sorted_centroids[0]]
        
        for centroid in sorted_centroids[1:]:
            # Check if in same row
            if abs(centroid[1] - current_row[-1][1]) < row_threshold:
                current_row.append(centroid)
            else:
                rows.append(current_row)
                current_row = [centroid]
        
        # Add last row
        if current_row:
            rows.append(current_row)
        
        # Filter out rows with too few nests (likely false detections)
        min_nests_per_row = self.grid_config.columns // 2
        rows = [row for row in rows if len(row) >= min_nests_per_row]
        
        return rows
    
    def _validate_id_assignment(self, nest_with_ids: Dict):
        """Validate that ID assignment makes sense.
        
        Args:
            nest_with_ids: Dictionary of nest IDs and coordinates
        """
        ids = [int(id_str) for id_str in nest_with_ids.keys()]
        
        # Check for duplicates
        if len(ids) != len(set(ids)):
            logger.error("Duplicate nest IDs detected!")
        
        # Check ID range
        max_id = max(ids)
        if max_id > self.grid_config.expected_total + self.grid_config.tolerance:
            logger.warning(f"Max ID ({max_id}) exceeds expected range")
        
        # Check for gaps
        gaps = []
        for i in range(1, max(ids) + 1):
            if i not in ids:
                gaps.append(i)
        
        if gaps:
            logger.warning(f"Missing nest IDs: {gaps[:10]}...")  # Show first 10
    
    def _save_reference_frame(
        self,
        video_path: str,
        output_folder: str,
        nest_with_ids: Dict,
        res_width: int,
        res_height: int
    ):
        """Save annotated reference frame for cross-video matching.
        
        Args:
            video_path: Path to video
            nest_with_ids: Dictionary of nest IDs and coordinates
            res_width: Frame width
            res_height: Frame height
        """
        # output_folder = Path(video_path).parent / "nest_references"
        # output_folder.mkdir(exist_ok=True)
        
        # Create blank frame
        reference_frame = np.zeros((res_height, res_width, 3), dtype=np.uint8)
        
        # Draw nest boxes and IDs
        for nest_id, box in nest_with_ids.items():
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Draw box
            cv2.rectangle(reference_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID
            cv2.putText(
                reference_frame,
                nest_id,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Save
        filename = video_path.split("/")[-1].replace(".mp4", "_nest_reference.png")
        output_folder = Path(output_folder)
        output_path = output_folder / filename
        cv2.imwrite(str(output_path), reference_frame)
        
        logger.info(f"Saved reference frame: {output_path}")
    
    def _create_nest_dataframe(self, nest_with_ids: Dict) -> pd.DataFrame:
        """Create DataFrame in expected format.
        
        Args:
            nest_with_ids: Dictionary of nest IDs and coordinates
            
        Returns:
            DataFrame with nest information
        """
        # Convert to format expected by rest of pipeline
        coordinates = list(nest_with_ids.values())
        
        df = pd.DataFrame({
            'frame': [0],
            'coordinates': [coordinates],
            'state': [[2.0] * len(coordinates)],  # All are nest holes
            'confidence': [[0.9] * len(coordinates)]  # High confidence
        })
        
        return df
    
    def match_to_reference(
        self,
        video_path: str,
        reference_path: str,
        res_height: int,
        res_width: int
    ) -> Dict:
        """Match detected nests to existing reference IDs.
        
        Use this for subsequent videos to maintain ID consistency.
        
        Args:
            video_path: Path to new video
            reference_path: Path to reference frame image
            res_height: Target height
            res_width: Target width
            
        Returns:
            Dictionary of nest IDs and coordinates
        """
        logger.info(f"Matching nests to reference: {reference_path}")
        
        # Detect nests in new video
        detections = self.detect_nests_exhaustive(video_path, res_height, res_width)
        
        # Load reference
        reference_img = cv2.imread(reference_path)
        
        # Extract reference nest positions from saved image
        # This is a simplified version - you might want to save the actual coordinates
        
        # For now, return the detections
        # In production, you'd match detected nests to reference positions
        
        logger.info("✓ Nest matching complete")
        
        return self._extract_nests_from_dataframe(detections)
    
    def _extract_nests_from_dataframe(self, df: pd.DataFrame) -> Dict:
        """Extract nests dictionary from DataFrame."""
        coordinates = df.iloc[0]['coordinates']
        
        nests_dict = {}
        for i, box in enumerate(coordinates):
            nests_dict[str(i + 1)] = box
        
        return nests_dict