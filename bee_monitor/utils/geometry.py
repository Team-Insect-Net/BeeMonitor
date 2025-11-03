"""Geometry utilities for bounding box operations and spatial calculations."""

import numpy as np
from typing import Tuple, List

# Type aliases for better readability
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
Point = Tuple[float, float]  # (x, y)


def compute_centroid(bbox: BBox) -> Point:
    """Compute the centroid of a bounding box.
    
    Args:
        bbox: Bounding box in format (x1, y1, x2, y2)
        
    Returns:
        Centroid coordinates (x, y)
        
    Example:
        >>> bbox = (10, 20, 30, 40)
        >>> compute_centroid(bbox)
        (20.0, 30.0)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def compute_iou(box1: BBox, box2: BBox) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
        
    Example:
        >>> box1 = (0, 0, 10, 10)
        >>> box2 = (5, 5, 15, 15)
        >>> iou = compute_iou(box1, box2)
        >>> 0.0 < iou < 1.0
        True
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou


def euclidean_distance(point1: Point, point2: Point) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Euclidean distance
        
    Example:
        >>> p1 = (0, 0)
        >>> p2 = (3, 4)
        >>> euclidean_distance(p1, p2)
        5.0
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def is_inside_bbox(point: Point, bbox: BBox, padding: int = 0) -> bool:
    """Check if a point is inside a bounding box with optional padding.
    
    Args:
        point: Point coordinates (x, y)
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        padding: Padding to add around bbox (default: 0)
        
    Returns:
        True if point is inside (padded) bbox, False otherwise
        
    Example:
        >>> point = (15, 15)
        >>> bbox = (10, 10, 20, 20)
        >>> is_inside_bbox(point, bbox)
        True
        >>> is_inside_bbox(point, bbox, padding=-10)
        False
    """
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    
    # Apply padding
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding
    
    return x_min <= x <= x_max and y_min <= y <= y_max


def expand_bbox(bbox: BBox, padding_x: int = 0, padding_y: int = 0) -> BBox:
    """Expand a bounding box by adding padding.
    
    Args:
        bbox: Original bounding box (x1, y1, x2, y2)
        padding_x: Horizontal padding to add
        padding_y: Vertical padding to add
        
    Returns:
        Expanded bounding box
        
    Example:
        >>> bbox = (10, 10, 20, 20)
        >>> expand_bbox(bbox, padding_x=5, padding_y=5)
        (5, 5, 25, 25)
    """
    x1, y1, x2, y2 = bbox
    return (x1 - padding_x, y1 - padding_y, x2 + padding_x, y2 + padding_y)


def clip_bbox(bbox: BBox, max_width: int, max_height: int) -> BBox:
    """Clip bounding box coordinates to image boundaries.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        max_width: Maximum width (image width)
        max_height: Maximum height (image height)
        
    Returns:
        Clipped bounding box
        
    Example:
        >>> bbox = (-5, -5, 1300, 800)
        >>> clip_bbox(bbox, max_width=1280, max_height=720)
        (0, 0, 1280, 720)
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, max_width))
    y1 = max(0, min(y1, max_height))
    x2 = max(0, min(x2, max_width))
    y2 = max(0, min(y2, max_height))
    return (x1, y1, x2, y2)


def bbox_area(bbox: BBox) -> float:
    """Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Area of the bounding box
        
    Example:
        >>> bbox = (0, 0, 10, 20)
        >>> bbox_area(bbox)
        200.0
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def aspect_ratio(bbox: BBox) -> float:
    """Calculate aspect ratio of a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Aspect ratio (width / height)
        
    Example:
        >>> bbox = (0, 0, 20, 10)
        >>> aspect_ratio(bbox)
        2.0
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width / height if height != 0 else 0


def xywh_to_xyxy(bbox_xywh: Tuple[float, float, float, float]) -> BBox:
    """Convert bounding box from (x_center, y_center, width, height) to (x1, y1, x2, y2).
    
    Args:
        bbox_xywh: Bounding box in xywh format
        
    Returns:
        Bounding box in xyxy format
        
    Example:
        >>> bbox_xywh = (10, 10, 20, 20)
        >>> xywh_to_xyxy(bbox_xywh)
        (0.0, 0.0, 20.0, 20.0)
    """
    x, y, w, h = bbox_xywh
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return (x1, y1, x2, y2)


def xyxy_to_xywh(bbox_xyxy: BBox) -> Tuple[float, float, float, float]:
    """Convert bounding box from (x1, y1, x2, y2) to (x_center, y_center, width, height).
    
    Args:
        bbox_xyxy: Bounding box in xyxy format
        
    Returns:
        Bounding box in xywh format
        
    Example:
        >>> bbox_xyxy = (0, 0, 20, 20)
        >>> xyxy_to_xywh(bbox_xyxy)
        (10.0, 10.0, 20.0, 20.0)
    """
    x1, y1, x2, y2 = bbox_xyxy
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return (x, y, w, h)


def remove_overlapping_points(points: List[Point], threshold: float = 20) -> List[Point]:
    """Remove points that are too close to each other.
    
    Keeps the first point from each group of overlapping points.
    
    Args:
        points: List of points (x, y)
        threshold: Minimum distance between points
        
    Returns:
        Filtered list of points
        
    Example:
        >>> points = [(0, 0), (5, 5), (100, 100)]
        >>> filtered = remove_overlapping_points(points, threshold=10)
        >>> len(filtered)
        2
    """
    if not points:
        return []
    
    points_array = np.array(points)
    keep_indices = []
    
    for i in range(len(points_array)):
        keep = True
        for j in keep_indices:
            distance = np.linalg.norm(points_array[i] - points_array[j])
            if distance < threshold:
                keep = False
                break
        if keep:
            keep_indices.append(i)
    
    return points_array[keep_indices].tolist()


def calculate_distance_matrix(points1: List[Point], points2: List[Point]) -> np.ndarray:
    """Calculate pairwise distance matrix between two sets of points.
    
    Args:
        points1: First set of points
        points2: Second set of points
        
    Returns:
        Distance matrix of shape (len(points1), len(points2))
        
    Example:
        >>> p1 = [(0, 0), (1, 1)]
        >>> p2 = [(2, 2), (3, 3)]
        >>> dist_matrix = calculate_distance_matrix(p1, p2)
        >>> dist_matrix.shape
        (2, 2)
    """
    if not points1 or not points2:
        return np.array([])
    
    arr1 = np.array(points1)
    arr2 = np.array(points2)
    
    # Calculate pairwise distances using broadcasting
    diff = arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return distances
