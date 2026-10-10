#!/usr/bin/env python3
"""Plan frame coverage on a cumulative integer-sample audio clock (no media I/O)."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def ceil_fraction(value):
    return -(-value.numerator // value.denominator)


def build_timeline(data):
    rate = positive_int(data.get("sample_rate"), "sample_rate")
    raw_fps = data.get("fps")
    if isinstance(raw_fps, bool) or not isinstance(raw_fps, (str, int)):
        raise ValueError('fps must be an integer or rational string, e.g. "30000/1001"')
    fps = Fraction(raw_fps)
    if fps <= 0:
        raise ValueError("fps must be positive")
    clips = data.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("clips must be a nonempty list")
    seen, output, cursor = set(), [], 0
    for clip in clips:
        identity = clip.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identity) or identity in seen:
            raise ValueError("clip IDs must be unique safe ASCII names")
        seen.add(identity)
        count = positive_int(clip.get("samples"), f"{identity}.samples")
        if "sample_rate" in clip and clip["sample_rate"] != rate:
            raise ValueError("resample explicitly before planning mixed sample rates")
        end = cursor + count
        first = ceil_fraction(Fraction(cursor, rate) * fps)
        last = ceil_fraction(Fraction(end, rate) * fps)
        output.append({"id": identity, "start_sample": cursor, "end_sample_exclusive": end,
                       "samples": count, "start_seconds": float(Fraction(cursor, rate)),
                       "end_seconds": float(Fraction(end, rate)), "first_frame": first,
                       "end_frame_exclusive": last, "frames": last - first})
        cursor = end
    total_frames = ceil_fraction(Fraction(cursor, rate) * fps)
    return {"schema_version": 1, "sample_rate": rate, "fps": str(fps),
            "total_samples": cursor, "total_frames": total_frames,
            "audio_seconds": float(Fraction(cursor, rate)),
            "video_seconds": float(Fraction(total_frames, 1) / fps), "clips": output,
            "note": "Frames are assigned by start <= frame_time < end. Sub-frame clips may have zero frames; final video covers the full audio."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = build_timeline(json.loads(args.input.read_text(encoding="utf-8-sig")))
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            if args.output.is_symlink():
                raise ValueError("refusing symlink output")
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (ValueError, OSError, KeyError, TypeError, AttributeError, ZeroDivisionError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
