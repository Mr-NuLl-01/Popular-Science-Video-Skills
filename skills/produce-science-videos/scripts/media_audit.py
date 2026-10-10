#!/usr/bin/env python3
"""Read-only FFmpeg/ffprobe audit. This does not perform subjective audiovisual review."""
import argparse
import json
from pathlib import Path
import subprocess


def run_json(command):
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def probe(path, ffprobe="ffprobe"):
    return run_json([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(Path(path).resolve())])


def packet_hashes(path, selector, ffprobe="ffprobe"):
    result = run_json([ffprobe, "-v", "error", "-select_streams", selector, "-show_packets",
                       "-show_entries", "packet=data_hash", "-show_data_hash", "sha256", "-of", "json", str(Path(path).resolve())])
    packets = result.get("packets", [])
    if not packets or any(not row.get("data_hash", "").startswith("SHA256:") for row in packets):
        raise ValueError(f"no hashable packets for {selector}: {path}")
    return [row["data_hash"] for row in packets]


def compare_stream(left, right, selector, ffprobe="ffprobe"):
    first, second = packet_hashes(left, selector, ffprobe), packet_hashes(right, selector, ffprobe)
    return {"selector": selector, "identical_packet_payloads": first == second,
            "left_packets": len(first), "right_packets": len(second),
            "timing_compared": False}


def audit(path, decode=False, compare_video=None, compare_audio=None, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    path = Path(path)
    if not path.is_file():
        raise ValueError("input file does not exist")
    structure = probe(path, ffprobe)
    if not structure.get("streams"):
        raise ValueError("no streams found")
    report = {"input": str(path), "bytes": path.stat().st_size, "structure": structure,
              "full_decode": "not_run", "subjective_listening": False, "subjective_visual_review": False,
              "comparisons": [], "passed": True}
    if decode:
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(path.resolve()),
                        "-map", "0:v?", "-map", "0:a?", "-f", "null", "-"],
                       check=True, capture_output=True, text=True)
        report["full_decode"] = "passed"
    for other, selector in ((compare_video, "v:0"), (compare_audio, "a:0")):
        if other is not None:
            comparison = compare_stream(path, other, selector, ffprobe)
            report["comparisons"].append(comparison)
            report["passed"] = report["passed"] and comparison["identical_packet_payloads"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--compare-video", type=Path, metavar="FILE")
    parser.add_argument("--compare-audio", type=Path, metavar="FILE")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    try:
        report = audit(args.input, args.decode, args.compare_video, args.compare_audio, args.ffmpeg, args.ffprobe)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        if not report["passed"]:
            raise SystemExit(1)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        parser.exit(2, f"audit failed: {detail}\n")


if __name__ == "__main__":
    main()
