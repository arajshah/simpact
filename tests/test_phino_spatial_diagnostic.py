"""The bounded per-fold rerun must reproduce the original probe's metrics."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import phino_spatial_diagnostic as diag


def test_fold_probe_and_decoder_match_original():
    rng = np.random.default_rng(2)
    cfgs, bank = [], {}
    a = diag.probe.ACTIONS[0]
    for g in ("A","B","C"):
        for i in range(6):
            images = [f"{g}/{i}/{k}" for k in range(6)]
            bank.update({p:rng.normal(size=3) for p in images})
            clip = dict(images=images,centroid_px=rng.normal(size=(6,2)),mask_area_px=rng.uniform(10,20,6))
            cfgs.append(dict(geometry=g,cluster=i,mass=0.2+i/50,friction=0.5-i/30,clips={a:clip}))
    X = diag.probe.build_matrix(cfgs,(a,),"visual",bank.__getitem__,{})
    ref = diag.probe.summarize(diag.probe.evaluate_split(cfgs,X,"logo"))
    ys,ps=[],[]
    for g in ("A","B","C"):
        scores = diag.fold_targets(cfgs,g,(a,),"visual",bank.__getitem__)
        for t,value in scores.items():
            np.testing.assert_allclose(value,ref[t]["per_fold"][g]["nmae"])
        y,p = diag.position_fold(cfgs,g,a,bank.__getitem__)
        ys.append(y)
        ps.append(p)
    got = diag.position_scores(np.concatenate(ys),np.concatenate(ps),True)
    original = diag.probe.position_decoding(cfgs,bank.__getitem__,a)
    for metric,score in original.items():
        np.testing.assert_allclose(score,got[metric])


def test_fold_constant_position_is_undefined():
    y = np.ones((5,15))
    assert diag.position_scores(y,y+0.1,False)["r2_mean"] is None
