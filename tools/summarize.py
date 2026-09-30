"""Collect finished runs into a markdown table.

    python tools/summarize.py /home/user/runs/p1_* /home/user/runs/p2_* > results.md

nextYOLO runs provide summary.json (+ history.json); Ultralytics baseline runs provide eval_nextyolo_metric.json.
Both are scored by the same COCO-style evaluator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def row_for(run: Path) -> dict | None:
    s, u = run / "summary.json", run / "eval_nextyolo_metric.json"
    if s.exists():
        d = json.loads(s.read_text())
        hist = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else []
        mids = [h for h in hist if "mAP" in h]
        last = mids[-1] if mids else {}
        cfg = d["config"]
        changes = {**{f"model.{k}": v for k, v in cfg["model"].items() if k != "scale"},
                   **{f"loss.{k}": v for k, v in cfg["loss"].items()}}
        if cfg.get("optimizer", "musgd") != "musgd":
            changes["optimizer"] = cfg["optimizer"]
        if cfg.get("init"):
            changes["init"] = Path(cfg["init"]).name + ("" if cfg.get("init_head", "subset") == "subset"
                                                         else f" ({cfg['init_head']} cls head)")
        for k, v in cfg.get("distill", {}).items():
            changes[f"distill.{k}"] = Path(v).name if k == "teacher" else v
        if cfg.get("seed", 0):
            changes["seed"] = cfg["seed"]
        nms = d.get("final_o2m_nms", {})
        return dict(run=run.name, impl="nextYOLO", changes=", ".join(f"{k}={v}" for k, v in changes.items()) or "—",
                    e2e=last.get("mAP"), e2e50=last.get("mAP50"), aps=last.get("APs"),
                    nms=nms.get("mAP"), nms50=nms.get("mAP50"),
                    mid=mids[0].get("mAP") if len(mids) > 1 else None, hours=d.get("hours"),
                    epochs=len(hist))
    if u.exists():
        d = json.loads(u.read_text())
        e, n = d.get("e2e", {}), d.get("nms", {})
        return dict(run=run.name, impl="Ultralytics", changes="reference implementation",
                    e2e=e.get("mAP"), e2e50=e.get("mAP50"), aps=e.get("APs"), nms=n.get("mAP"),
                    nms50=n.get("mAP50"), mid=None, hours=None, epochs=None)
    return None


def fmt(x, pct=True):
    return "—" if x is None else (f"{100 * x:.2f}" if pct else f"{x:.1f}")


def main():
    rows = [r for p in sys.argv[1:] if (r := row_for(Path(p)))]
    print("| run | impl | change vs. YOLO26 recipe | AP e2e (o2o, NMS-free) | AP50 e2e | AP_S e2e | "
          "AP o2m+NMS | AP50 o2m+NMS | AP e2e @ first eval | hours |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run']} | {r['impl']} | {r['changes']} | {fmt(r['e2e'])} | {fmt(r['e2e50'])} | {fmt(r['aps'])} | "
              f"{fmt(r['nms'])} | {fmt(r['nms50'])} | {fmt(r['mid'])} | {fmt(r['hours'], False)} |")


if __name__ == "__main__":
    main()
