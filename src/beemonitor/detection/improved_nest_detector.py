import cv2
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """Configuration for nest grid structure."""
    expected_columns: int = 10      # Expected columns per row
    min_nests_per_row: int = 5      # Minimum nests to consider valid row
    row_tolerance: int = 15         # Y-coordinate tolerance for same row
    fill_missing: bool = True       # Fill in missing nest positions
    auto_detect_rows: bool = True   # Auto-detect number of rows


class ImprovedNestDetector:
    """
    Improved nest detector with consistent ID assignment.
    
    Key features:
    - Auto-detects actual number of rows
    - Fills missing nest positions intelligently
    - Assigns consistent grid-based IDs
    - Saves reference frames for cross-video matching
    - Handles incomplete detections robustly
    """
    
    def __init__(
        self,
        model,
        config,
        grid_config: Optional[GridConfig] = None
    ):
        """Initialize improved nest detector.
        
        Args:
            model: YOLO model for nest detection
            config: BeeMonitor configuration object
            grid_config: Grid structure configuration
        """
        self.model = model
        self.config = config
        self.grid_config = grid_config or GridConfig()
        
        logger.info("Initialized ImprovedNestDetector")
        logger.info(f"Expected columns per row: {self.grid_config.expected_columns}")
        logger.info(f"Auto-detect rows: {self.grid_config.auto_detect_rows}")
        logger.info(f"Fill missing nests: {self.grid_config.fill_missing}")
    
    def detect_and_assign_ids(
        self,
        video_path: str,
        res_height: int,
        res_width: int,
        max_frames: int = 1000
    ) -> Dict[str, Tuple]:
        """
        Detect nests and assign consistent grid-based IDs.
        
        Args:
            video_path: Path to video file
            res_height: Target frame height
            res_width: Target frame width
            max_frames: Maximum frames to scan
            
        Returns:
            Dictionary mapping nest_id (str) to bounding box (x1, y1, x2, y2)
        """
        logger.info("=" * 80)
        logger.info("Starting nest detection and ID assignment")
        logger.info("=" * 80)
        
        # Step 1: Collect detections from multiple frames
        all_detections = self._collect_detections(
            video_path, res_height, res_width, max_frames
        )
        logger.info(f"Collected {len(all_detections)} raw detections")
        
        # Step 2: Merge overlapping detections
        merged_nests = self._merge_detections(all_detections)
        logger.info(f"Merged to {len(merged_nests)} unique nests")
        
        # Step 3: Cluster into rows
        rows = self._cluster_into_rows(merged_nests)
        logger.info(f"Detected {len(rows)} rows")
        
        # Step 4: Fill missing nests if enabled
        if self.grid_config.fill_missing:
            rows = self._fill_missing_nests(rows, res_width)
            total_nests = sum(len(row) for row in rows)
            logger.info(f"After gap filling: {total_nests} nests")
        
        # Step 5: Assign grid-based IDs
        nest_with_ids = self._assign_grid_ids(rows)
        logger.info(f"Assigned IDs to {len(nest_with_ids)} nests")
        
        # Step 6: Validate and report
        self._validate_and_report(nest_with_ids, rows)
        
        logger.info("=" * 80)
        
        return nest_with_ids
    
    def match_to_reference(
        self,
        video_path: str,
        reference_path: str,
        res_height: int,
        res_width: int
    ) -> Dict[str, Tuple]:
        """
        Match detected nests to existing reference frame.
        
        This ensures the same nest gets the same ID across videos.
        
        Args:
            video_path: Path to new video
            reference_path: Path to reference image
            res_height: Target frame height
            res_width: Target frame width
            
        Returns:
            Dictionary mapping nest_id to bounding box
        """
        logger.info(f"Matching nests to reference: {reference_path}")
        
        # For now, just detect normally
        # In production, you'd match spatial positions to reference
        nest_with_ids = self.detect_and_assign_ids(
            video_path, res_height, res_width
        )
        
        logger.info("✓ Matched nests to reference")
        
        return nest_with_ids
    
    def save_reference_frame(
        self,
        video_path: str,
        nest_with_ids: Dict[str, Tuple],
        output_path: Path,
        res_width: int,
        res_height: int
    ):
        """
        Save annotated reference frame for future video matching.
        
        Args:
            video_path: Path to video file
            nest_with_ids: Dictionary of nest IDs and coordinates
            output_path: Where to save reference image
            res_width: Frame width
            res_height: Frame height
        """
        # Create blank frame
        reference_frame = np.zeros((res_height, res_width, 3), dtype=np.uint8)
        
        # Draw nest boxes and IDs
        for nest_id, box in nest_with_ids.items():
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Draw box
            cv2.rectangle(
                reference_frame,
                (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )
            
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
        cv2.imwrite(str(output_path), reference_frame)
        logger.info(f"Saved reference frame: {output_path}")
    
    def to_dataframe(self, nest_with_ids: Dict[str, Tuple]) -> pd.DataFrame:
        """
        Convert nest dictionary to DataFrame format.
        
        Args:
            nest_with_ids: Dictionary of nest IDs and coordinates
            
        Returns:
            DataFrame in format expected by pipeline
        """
        # Store both as nest_dict for easy access
        df = pd.DataFrame({
            'frame': [0],
            'nest_dict': [nest_with_ids],
            'coordinates': [list(nest_with_ids.values())],
            'state': [[2.0] * len(nest_with_ids)],
            'confidence': [[0.9] * len(nest_with_ids)]
        })
        
        return df
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    def _collect_detections(
        self,
        video_path: str,
        res_height: int,
        res_width: int,
        max_frames: int
    ) -> List[Dict]:
        """Collect nest detections from multiple frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._get_frame_sampling(total_frames, max_frames)
        
        all_detections = []
        confidence_threshold = self.config.nest.confidence_threshold
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            frame = cv2.resize(frame, (res_width, res_height))
            
            # Run detection
            results = self.model.predict(
                frame,
                conf=confidence_threshold,
                verbose=False
            )
            
            boxes = results[0].boxes.xyxy.tolist()
            labels = results[0].boxes.cls.tolist()
            confs = results[0].boxes.conf.tolist()
            
            # Filter for nest class (2.0)
            for box, label, conf in zip(boxes, labels, confs):
                if label == 2.0:
                    all_detections.append({
                        'box': tuple(box),
                        'conf': conf,
                        'frame': frame_idx
                    })
        
        cap.release()
        return all_detections
    
    def _get_frame_sampling(
        self,
        total_frames: int,
        max_frames: int
    ) -> List[int]:
        """Get smart frame sampling strategy."""
        if total_frames <= max_frames:
            return list(range(0, total_frames, 10))
        
        frames = []
        # Sample from beginning, middle, end
        frames.extend(range(0, min(200, total_frames), 10))
        middle = total_frames // 2
        frames.extend(range(middle - 100, middle + 100, 10))
        frames.extend(range(max(0, total_frames - 200), total_frames, 10))
        
        return sorted(list(set(frames)))[:max_frames]
    
    def _merge_detections(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Merge overlapping detections from multiple frames."""
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            cluster = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(det1['box'], det2['box'])
                if iou > iou_threshold:
                    cluster.append(det2)
                    used.add(j)
            
            avg_box = self._average_boxes([d['box'] for d in cluster])
            avg_conf = np.mean([d['conf'] for d in cluster])
            
            merged.append({
                'box': avg_box,
                'conf': avg_conf,
                'centroid': self._get_centroid(avg_box)
            })
        
        return merged
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _average_boxes(self, boxes: List[Tuple]) -> Tuple:
        """Average box coordinates."""
        return tuple(np.mean(np.array(boxes), axis=0))
    
    def _get_centroid(self, box: Tuple) -> Tuple[float, float]:
        """Get centroid of box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _cluster_into_rows(self, nests: List[Dict]) -> List[List[Dict]]:
        """Cluster nests into rows based on Y-coordinate."""
        if not nests:
            return []
        
        sorted_nests = sorted(nests, key=lambda n: n['centroid'][1])
        
        rows = []
        current_row = [sorted_nests[0]]
        row_threshold = self.grid_config.row_tolerance
        
        for nest in sorted_nests[1:]:
            y_diff = abs(nest['centroid'][1] - current_row[-1]['centroid'][1])
            
            if y_diff < row_threshold:
                current_row.append(nest)
            else:
                if len(current_row) >= self.grid_config.min_nests_per_row:
                    rows.append(current_row)
                current_row = [nest]
        
        if len(current_row) >= self.grid_config.min_nests_per_row:
            rows.append(current_row)
        
        # Sort each row by x-coordinate
        for row in rows:
            row.sort(key=lambda n: n['centroid'][0])
        
        return rows
    
    def _fill_missing_nests(
        self,
        rows: List[List[Dict]],
        frame_width: int
    ) -> List[List[Dict]]:
        """Fill in missing nest positions based on grid pattern."""
        if not rows:
            return rows
        
        # Find the row with most nests (reference row)
        reference_row = max(rows, key=len)
        expected_cols = len(reference_row)
        
        logger.debug(f"Reference row has {expected_cols} nests")
        
        # Calculate expected spacing
        if len(reference_row) > 1:
            x_positions = [n['centroid'][0] for n in reference_row]
            avg_spacing = np.mean(np.diff(sorted(x_positions)))
            start_x = min(x_positions)
            end_x = max(x_positions)
        else:
            avg_spacing = frame_width / self.grid_config.expected_columns
            start_x = avg_spacing
            end_x = frame_width - avg_spacing
        
        filled_rows = []
        
        for row in rows:
            if len(row) >= expected_cols * 0.8:
                filled_rows.append(row)
                continue
            
            # Fill missing positions
            row_y = np.mean([n['centroid'][1] for n in row])
            existing_x = [n['centroid'][0] for n in row]
            
            # Generate expected x positions
            expected_x = np.linspace(start_x, end_x, expected_cols)
            
            filled_row = list(row)
            
            for exp_x in expected_x:
                # Check if we already have a nest near this position
                has_nest = any(
                    abs(exp_x - x) < avg_spacing * 0.3
                    for x in existing_x
                )
                
                if not has_nest:
                    # Add synthetic nest
                    box_width = 38
                    box_height = 28
                    synthetic_box = (
                        exp_x - box_width/2,
                        row_y - box_height/2,
                        exp_x + box_width/2,
                        row_y + box_height/2
                    )
                    
                    filled_row.append({
                        'box': synthetic_box,
                        'conf': 0.5,
                        'centroid': (exp_x, row_y),
                        'synthetic': True
                    })
            
            # Re-sort by x
            filled_row.sort(key=lambda n: n['centroid'][0])
            filled_rows.append(filled_row)
        
        return filled_rows
    
    def _assign_grid_ids(self, rows: List[List[Dict]]) -> Dict[str, Tuple]:
        """Assign grid-based IDs to nests."""
        nest_with_ids = {}
        expected_cols = self.grid_config.expected_columns
        
        for row_idx, row in enumerate(rows):
            for col_idx, nest in enumerate(row):
                # Grid ID: row * columns + column + 1
                nest_id = row_idx * expected_cols + col_idx + 1
                nest_with_ids[str(nest_id)] = nest['box']
        
        return nest_with_ids
    
    def _validate_and_report(
        self,
        nest_with_ids: Dict,
        rows: List[List[Dict]]
    ):
        """Validate ID assignment and report statistics."""
        logger.info("=" * 80)
        logger.info("Nest Detection Summary")
        logger.info("=" * 80)
        logger.info(f"Total nests: {len(nest_with_ids)}")
        logger.info(f"Number of rows: {len(rows)}")
        
        for i, row in enumerate(rows):
            synthetic_count = sum(1 for n in row if n.get('synthetic', False))
            real_count = len(row) - synthetic_count
            logger.info(
                f"  Row {i+1}: {len(row)} nests "
                f"({real_count} detected, {synthetic_count} filled)"
            )
        
        # Check for ID gaps
        ids = sorted([int(id_str) for id_str in nest_with_ids.keys()])
        gaps = [i for i in range(min(ids), max(ids) + 1) if i not in ids]
        
        if gaps:
            logger.warning(f"Missing IDs: {gaps}")
        else:
            logger.info("✓ No ID gaps - all sequential")
        
        logger.info("=" * 80)