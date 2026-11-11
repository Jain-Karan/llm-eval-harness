"""The committed baseline must be the eval of the released configuration. If it drifts (scoring
changed, golden set regenerated), every gate decision compares against the wrong numbers."""

import json
from pathlib import Path

import yaml

from evalharness.cli import evaluate

ROOT = Path(__file__).resolve().parents[1]
VOLATILE = {"p95_latency_ms"}


def test_baseline_matches_released_config():
    release = yaml.safe_load((ROOT / "release.yaml").read_text())
    fresh = evaluate(release["prompt"], guarded=release["guarded"]).to_json()
    stored = json.loads((ROOT / "reports" / "baseline.json").read_text())
    assert stored["prompt_version"] == release["prompt"]
    for scope in ["overall", *fresh["tenants"]]:
        a = fresh["overall"] if scope == "overall" else fresh["tenants"][scope]
        b = stored["overall"] if scope == "overall" else stored["tenants"][scope]
        for k in a.keys() - VOLATILE:
            assert a[k] == b[k], (
                f"{scope}.{k}: baseline {b[k]} != fresh {a[k]}; regenerate with "
                "evalh run --prompt <release> --guarded --out reports/baseline.json"
            )
