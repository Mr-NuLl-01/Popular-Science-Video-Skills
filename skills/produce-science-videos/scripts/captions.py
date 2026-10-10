#!/usr/bin/env python3
"""Export reviewed sample-clock cues to SRT; does not recognize or align speech."""
import argparse
import json
from pathlib import Path


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def milliseconds(samples, rate):
    # Exact round-half-up, without floating-point accumulation.
    return (samples * 2000 + rate) // (rate * 2)


def stamp(value):
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def to_srt(data):
    rate = integer(data.get("sample_rate"), "sample_rate", 1)
    cues = data.get("cues")
    if not isinstance(cues, list) or not cues:
        raise ValueError("cues must be a nonempty list")
    blocks, previous_end, previous_ms = [], 0, 0
    for number, cue in enumerate(cues, 1):
        start = integer(cue.get("start_sample"), "start_sample")
        end = integer(cue.get("end_sample"), "end_sample")
        text = cue.get("text")
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise ValueError(f"cue {number}: empty or invalid text")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if any(not line.strip() for line in text.split("\n")):
            raise ValueError(f"cue {number}: blank lines are not valid inside an SRT cue")
        if end <= start or start < previous_end:
            raise ValueError(f"cue {number}: reversed, overlapping, or unordered sample interval")
        start_ms, end_ms = milliseconds(start, rate), milliseconds(end, rate)
        if end_ms <= start_ms or start_ms < previous_ms:
            raise ValueError(f"cue {number}: interval collapses or overlaps at millisecond precision")
        blocks.append(f"{number}\n{stamp(start_ms)} --> {stamp(end_ms)}\n{text}\n")
        previous_end, previous_ms = end, end_ms
    return "\n".join(blocks) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        content = to_srt(json.loads(args.input.read_text(encoding="utf-8-sig")))
        if args.output.is_symlink():
            raise ValueError("refusing symlink output")
        args.output.write_text(content, encoding="utf-8")
        print(json.dumps({"output": str(args.output), "speech_alignment_performed": False}, ensure_ascii=False))
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
