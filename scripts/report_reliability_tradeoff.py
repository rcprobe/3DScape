#!/usr/bin/env python
"""Report ReliabilityNet noise-vs-coverage tradeoffs from reconstruction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a ReliabilityNet tradeoff report.")
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help=(
            "Case encoded as name::baseline_manifest::reliability_manifest"
            "[::baseline_metrics_json::reliability_metrics_json]. Repeat for multiple scans."
        ),
    )
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    cases = [_parse_case(raw) for raw in args.case]
    report = _render_report(cases)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report, encoding="utf-8")
    print(f"report_md: {args.out_md}")


def _parse_case(raw: str) -> dict:
    parts = raw.split("::")
    if len(parts) not in (3, 5):
        raise ValueError(
            "--case must be name::baseline_manifest::reliability_manifest"
            "[::baseline_metrics_json::reliability_metrics_json]"
        )
    name, baseline_manifest, reliability_manifest = parts[:3]
    case = {
        "name": name,
        "baseline_manifest": _load_json(baseline_manifest),
        "reliability_manifest": _load_json(reliability_manifest),
        "baseline_metrics": None,
        "reliability_metrics": None,
    }
    if len(parts) == 5:
        case["baseline_metrics"] = _load_json(parts[3])
        case["reliability_metrics"] = _load_json(parts[4])
    return case


def _render_report(cases: list[dict]) -> str:
    lines = [
        "# ReliabilityNet Noise Vs Coverage Report",
        "",
        "This report treats learned reliability as a tradeoff, not a binary better/worse claim.",
        "Coverage is approximated by retained downsampled point count. Noise is measured with",
        "reference-mesh distance only when a reference mesh is available.",
        "",
        "## Point Retention",
        "",
        "| Scan | Baseline points | Reliability points | Retained | Reliability summary |",
        "|---|---:|---:|---:|---|",
    ]
    for case in cases:
        baseline = case["baseline_manifest"]
        reliable = case["reliability_manifest"]
        baseline_points = int(baseline["downsampled_points"])
        reliable_points = int(reliable["downsampled_points"])
        retained = reliable_points / max(baseline_points, 1)
        summary = _reliability_summary_text(reliable.get("reliability_summary"))
        lines.append(
            f"| {case['name']} | {baseline_points:,} | {reliable_points:,} | "
            f"{retained:.1%} | {summary} |"
        )

    metric_cases = [case for case in cases if case["baseline_metrics"] and case["reliability_metrics"]]
    if metric_cases:
        lines.extend(
            [
                "",
                "## Reference-Mesh Noise/Coverage Metrics",
                "",
                "| Scan | Run | source->ref median | source->ref p90 | ref coverage | ref->source p90 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for case in metric_cases:
            for label, metrics_key in (
                ("Baseline", "baseline_metrics"),
                ("Reliability", "reliability_metrics"),
            ):
                metrics = case[metrics_key]
                s2r = metrics["source_to_reference_m"]
                r2s = metrics["reference_to_source_m"]
                lines.append(
                    f"| {case['name']} | {label} | {s2r['median']:.4f} m | "
                    f"{s2r['p90']:.4f} m | {r2s['matched_fraction']:.1%} | {r2s['p90']:.4f} m |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A useful reliability model should reduce noise without destroying coverage.",
            "- Lower source-to-reference distance can be misleading if the model simply deletes difficult geometry.",
            "- Reference-to-source coverage and retained point count catch that failure mode.",
            "- Reliability-colored viewers are useful because they show what the model distrusts without hiding it.",
        ]
    )
    return "\n".join(lines) + "\n"


def _reliability_summary_text(summary: dict | None) -> str:
    if not summary:
        return "n/a"
    return f"mean {summary['mean']:.3f}, p10 {summary['p10']:.3f}, p90 {summary['p90']:.3f}"


def _load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
