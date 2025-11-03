import numpy as np
from beemonitor.nestid.assign import assign_ids
from beemonitor.nestid.template import NestTemplate
from beemonitor.types import Detection

def _grid(rows=6, cols=10, dx=30, dy=30, off=(100,200)):
    centers, ids, n = [], [], 1
    for r in range(rows):
        for c in range(cols):
            centers.append((off[0]+c*dx, off[1]+r*dy))
            ids.append(str(n)); n += 1
    return np.array(centers, dtype=np.float32), ids

def _dets_from_centers(centers, w=14, h=14):
    ds=[]
    for cx,cy in centers:
        ds.append(Detection(frame=0, bbox=(cx-w/2,cy-h/2,cx+w/2,cy+h/2), conf=0.9, cls="nest"))
    return ds

def test_partial_missing_and_alignment_fill():
    tpl_centers, ids = _grid()
    tpl = NestTemplate(rows=6, cols=10, ids=ids, centers=[(float(x),float(y)) for x,y in tpl_centers])

    # Warp with affine + light perspective
    A = np.array([[1.02, 0.03, -12.0], [-0.02, 1.01, 15.0]], dtype=np.float32)
    ones = np.ones((tpl_centers.shape[0],1), dtype=np.float32)
    aff = (A @ np.hstack([tpl_centers, ones]).T).T

    H = np.array([[1.0, 0.001, 15.0],
                  [-0.001, 1.0, -12.0],
                  [0.00002, 0.0002, 1.0]], dtype=np.float32)
    homo = (H @ np.hstack([tpl_centers, ones]).T).T
    homo = homo[:, :2] / np.clip(homo[:, 2:], 1e-6, None)

    # Compose small perspective on top of affine
    warped = homo

    dets = _dets_from_centers(warped)
    # remove a detection in the middle of row 1 (ID 5)
    del dets[4]

    tubes = assign_ids(
        dets, tpl,
        allow_partial=True, min_visible_fraction=0.85,
        method_order=("affine","homography"),
        aff_thr=4.0, hom_thr=3.0, residual_ok=8.0,
        fill_missing=True
    )

    # IDs remain 1..60, ID 5 exists and has a poly (filled)
    assert len(tubes) == 60
    assert tubes[4].tube_id == "5"
    assert tubes[4].poly != []   # filled from template
