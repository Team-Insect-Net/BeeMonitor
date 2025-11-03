from __future__ import annotations
from typing import Any, Dict
from .base import NestIdentifier

def build_nest_identifier(cfg_root: Dict[str, Any]) -> NestIdentifier:
    """
    Factory for nest ID. Pass the full YAML dict (root), so we can access detectors + nestid.
    """
    nest = cfg_root.get("nestid") or {}
    name = (nest.get("name") or "yolo_grid").lower()
    if name in ("yolo_grid", "yolo+grid", "grid"):
        from .yolo_grid_identifier import YoloGridNestIdentifier
        return YoloGridNestIdentifier(
            cfg_root=cfg_root,
            rows=int(nest["rows"]),
            cols=int(nest["cols"]),
            template_path=str(nest.get("template_path", "beemonitor/config/nest_template.json")),
            cache_dir=str(nest.get("cache_dir", "outputs/nest_cache")),
        )
    raise ValueError(f"Unknown nestid type: {name}")
