#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Serial, local-only Qwen3-TTS Base narration. See references/qwen3-tts.md.

Dry-run and integrity checks use only the Python standard library. Inference
requires the official qwen-tts package and its dependencies, installed separately.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import tempfile
import time
import wave

SCHEMA = "science-video-qwen3-tts-v1"
LANGUAGES = {x.lower() for x in (
    "Auto Chinese English Japanese Korean German French Russian Portuguese Spanish Italian"
).split()}
RESERVED = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
            *[f"LPT{i}" for i in range(1, 10)]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    """Stable UTF-8 JSON digest; rejects non-finite JSON numbers."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def validate_jobs(value):
    """Validate without mutating; IDs are portable flat filenames, case-insensitive."""
    require(isinstance(value, list) and value, "Jobs must be a nonempty JSON array.")
    seen = set()
    result = []
    for job in value:
        require(isinstance(job, dict) and set(job) == {"id", "text", "language", "seed"},
                "Each job must contain exactly id, text, language, seed.")
        identity = job["id"]
        require(isinstance(identity, str) and
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identity), "Unsafe job ID.")
        require(identity.upper() not in RESERVED, f"Reserved filename: {identity}")
        require(identity.casefold() not in seen, f"Duplicate job ID: {identity}")
        seen.add(identity.casefold())
        require(isinstance(job["text"], str) and job["text"].strip() and
                "\x00" not in job["text"], f"Empty/invalid text: {identity}")
        require(isinstance(job["language"], str) and job["language"].lower() in LANGUAGES,
                f"Unsupported language: {identity}")
        require(type(job["seed"]) is int and 0 <= job["seed"] < 2**32,
                f"Seed must be an integer from 0 through 2**32-1: {identity}")
        result.append(dict(job))
    return result


def input_signature(job, context):
    """Pure function: text, voice, model identity, settings and runtime are bound."""
    return canonical_hash({"schema": SCHEMA, "job": job, "context": context})


def safe_output_root(value):
    """Reject symlink/junction output components; never resolve an output escape."""
    path = Path(os.path.abspath(Path(value).expanduser()))
    for part in (path, *path.parents):
        require(not part.is_symlink() and not getattr(part, "is_junction", lambda: False)(),
                f"Output path contains a symlink/junction: {part}")
        require(not part.exists() or part.is_dir(), f"Not a directory: {part}")
    return path


def output_paths(root, identity):
    # Validate even when called directly by tests or other scripts.
    validate_jobs([{"id": identity, "text": "test", "language": "Auto", "seed": 0}])
    root = safe_output_root(root)
    return root / f"{identity}.wav", root / f"{identity}.json"


def regular_output(path):
    require(not path.is_symlink() and path.is_file(), f"Not a regular output file: {path}")
    require(path.stat().st_nlink == 1, f"Unexpected hard-linked output: {path}")


def wav_info(path):
    """Decode/check mono PCM16 frames with stdlib, then hash the full file."""
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), f"Missing/unsafe WAV: {path}")
    with wave.open(str(path), "rb") as audio:
        channels, width, rate, frames, compression, _ = audio.getparams()
        require(channels == 1 and width == 2 and compression == "NONE" and
                rate > 0 and frames > 0, f"Expected nonempty mono PCM16 WAV: {path}")
        remaining, nonzero = frames, False
        while remaining:
            count = min(remaining, 65536)
            block = audio.readframes(count)
            require(len(block) == count * 2, f"Truncated WAV: {path}")
            nonzero = nonzero or any(block)
            remaining -= count
        require(nonzero, f"All-zero WAV: {path}")
    return {"frames": frames, "sample_rate": rate, "channels": channels,
            "sample_width": width, "duration_seconds": frames / rate,
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def check_existing(root, job, signature, resume):
    """Return verified metadata or None. Every partial/mismatch is an error."""
    root = Path(root)
    wav_path, record_path = output_paths(root, job["id"])
    require(not list(root.glob(f".{job['id']}-*.partial")),
            f"Unfinished temporary files for {job['id']}; preserve/review them first.")
    present = [os.path.lexists(p) for p in (wav_path, record_path)]
    if not any(present):
        return None
    require(resume, f"Existing output for {job['id']}; use --resume to verify it.")
    require(all(present), f"Incomplete WAV/JSON pair for {job['id']}; nothing overwritten.")
    regular_output(wav_path)
    regular_output(record_path)
    record = read_json(record_path)
    require(isinstance(record, dict) and record.get("schema") == SCHEMA and
            record.get("status") == "inference_complete" and record.get("id") == job["id"],
            f"Invalid completion record for {job['id']}.")
    require(record.get("input_signature") == signature,
            f"Input signature mismatch for {job['id']}; choose a new output directory.")
    require(record.get("wav") == wav_info(wav_path),
            f"WAV integrity mismatch for {job['id']}; nothing overwritten.")
    return record


def package_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("qwen-tts", "torch", "torchaudio", "transformers", "accelerate",
                 "numpy", "soundfile", "flash-attn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def make_context(args, model_dir, reference, ref_text):
    config = read_json(model_dir / "config.json")
    require(isinstance(config, dict) and config.get("model_type") == "qwen3_tts" and
            config.get("tts_model_type") == "base", "A local Qwen3-TTS Base model is required.")
    # Small configuration/tokenizer files only. Never hash/load model weight files.
    config_names = ("config.json", "generate_config.json", "generation_config.json",
                    "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                    "vocab.json", "merges.txt", "speech_tokenizer/config.json",
                    "speech_tokenizer/preprocessor_config.json")
    configs = {name: sha256_file(model_dir / name) for name in config_names
               if (model_dir / name).is_file()}
    return {"runner_sha256": sha256_file(Path(__file__)),
            "model_directory": str(model_dir), "model_revision": args.model_revision,
            "configuration_sha256": configs, "reference_audio_sha256": sha256_file(reference),
            "reference_text_sha256": hashlib.sha256(ref_text.encode("utf-8")).hexdigest(),
            "device": args.device, "dtype": args.dtype, "attention": args.attention,
            "generation": {"max_new_tokens": args.max_new_tokens, "non_streaming_mode": True},
            "sampling_defaults": "model generate_config.json, then installed qwen-tts defaults",
            "x_vector_only_mode": False, "output_format": "mono PCM_16 WAV; no trimming/normalization",
            "runtime": package_versions()}


@contextlib.contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    root = safe_output_root(root)
    lock = root / ".qwen3-tts.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError(f"Output is locked: {lock}. Check for an active run before recovery.") from exc
    try:
        yield
    finally:
        lock.rmdir()


def new_temp(root, identity):
    fd, name = tempfile.mkstemp(prefix=f".{identity}-", suffix=".partial", dir=root)
    os.close(fd)
    return Path(name)


def publish_no_clobber(temporary, target):
    """Atomic no-replace publication on a local filesystem with hard-link support."""
    require(not temporary.is_symlink() and temporary.is_file(), "Unsafe temporary file.")
    os.link(temporary, target)  # Atomic: fails if target exists, even as a dangling symlink.
    temporary.unlink()         # Unlinks only the temporary name created by this invocation.


def fsync_file(path):
    with Path(path).open("rb+") as stream:
        os.fsync(stream.fileno())


def save_result(root, job, signature, context, samples, rate, seconds, sf):
    wav_path, record_path = output_paths(root, job["id"])
    temporary_wav = new_temp(root, job["id"])
    sf.write(str(temporary_wav), samples, rate, format="WAV", subtype="PCM_16")
    fsync_file(temporary_wav)
    info = wav_info(temporary_wav)
    record = {"schema": SCHEMA, "id": job["id"], "status": "inference_complete",
              "input_signature": signature, "job": job, "context": context,
              "wav": info, "generation_seconds": seconds,
              "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "semantic_listening_performed": False, "listening_status": "pending"}
    temporary_record = new_temp(root, job["id"])
    with temporary_record.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # If interrupted between these commits, resume rejects the incomplete pair.
    publish_no_clobber(temporary_wav, wav_path)
    publish_no_clobber(temporary_record, record_path)
    return record


def run_inference(args, jobs, root, model_dir, reference, ref_text, context):
    # Set before importing Hugging Face libraries. No in-run model downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    if args.device.startswith("cuda"):
        require(torch.cuda.is_available(), "CUDA is unavailable; no automatic CPU fallback.")
        torch.cuda.set_device(torch.device(args.device))
        if args.dtype == "bfloat16":
            require(torch.cuda.is_bf16_supported(), "BF16 unsupported; select --dtype float16 explicitly.")
    ref_samples, ref_rate = sf.read(str(reference), dtype="float32", always_2d=False)
    require(ref_samples.ndim == 1 and ref_samples.size > 0 and
            np.isfinite(ref_samples).all() and np.any(ref_samples),
            "Reference must be nonempty, finite, non-silent mono audio.")
    require(sha256_file(reference) == context["reference_audio_sha256"],
            "Reference changed during preflight.")
    kwargs = {"device_map": args.device, "dtype": getattr(torch, args.dtype),
              "local_files_only": True}
    if args.attention != "default":
        kwargs["attn_implementation"] = args.attention
    model = Qwen3TTSModel.from_pretrained(str(model_dir), **kwargs)
    prompt = model.create_voice_clone_prompt(ref_audio=(ref_samples, ref_rate),
                                            ref_text=ref_text, x_vector_only_mode=False)
    for job in jobs:
        signature = input_signature(job, context)
        if check_existing(root, job, signature, args.resume):
            continue
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(job["seed"])
        started = time.monotonic()
        wavs, rate = model.generate_voice_clone(text=job["text"], language=job["language"],
                                               voice_clone_prompt=prompt, **context["generation"])
        require(len(wavs) == 1, "Expected exactly one waveform per job.")
        samples = np.asarray(wavs[0])
        require(samples.ndim == 1 and samples.size > 0 and np.isfinite(samples).all() and
                np.any(samples), f"Empty/non-finite/silent synthesis: {job['id']}")
        require(float(np.max(np.abs(samples))) <= 1.0,
                f"Out-of-range amplitude: {job['id']}; refusing silent PCM clipping.")
        require(isinstance(rate, (int, np.integer)) and rate > 0, "Invalid output sample rate.")
        record = save_result(root, job, signature, context, samples, int(rate),
                             time.monotonic() - started, sf)
        print(json.dumps({"id": job["id"], "status": record["status"],
                          "duration_seconds": record["wav"]["duration_seconds"],
                          "listening_status": "pending"}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-revision", required=True, help="Pinned model commit/release label; no download")
    parser.add_argument("--ref-audio", required=True, type=Path)
    parser.add_argument("--ref-text-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N (default: cpu)")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attention", choices=("default", "eager", "sdpa", "flash_attention_2"), default="default")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    require(re.fullmatch(r"cpu|cuda:[0-9]+", args.device), "Use --device cpu or cuda:N.")
    if args.dtype == "auto":
        args.dtype = "float32" if args.device == "cpu" else "bfloat16"
    require(args.device != "cpu" or args.dtype == "float32", "CPU path requires float32.")
    require(args.attention != "flash_attention_2" or
            (args.device.startswith("cuda:") and args.dtype in {"float16", "bfloat16"}),
            "FlashAttention 2 requires supported CUDA hardware and FP16/BF16.")
    require(args.max_new_tokens > 0 and args.model_revision.strip(), "Invalid generation limit/model revision.")
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    reference = args.ref_audio.expanduser().resolve(strict=True)
    require(model_dir.is_dir() and reference.is_file() and reference.stat().st_size > 0,
            "Model directory and nonempty local reference audio are required.")
    ref_text = args.ref_text_file.expanduser().read_text(encoding="utf-8-sig").strip()
    require(ref_text and "\x00" not in ref_text, "Reference transcript is empty/invalid.")
    jobs = validate_jobs(read_json(args.jobs.expanduser()))
    root = safe_output_root(args.output_dir)
    require(not root.is_relative_to(model_dir), "Output directory cannot be inside the model directory.")
    context = make_context(args, model_dir, reference, ref_text)

    def plan():
        return [{"id": job["id"], "input_signature": input_signature(job, context),
                 "action": "verified_skip" if check_existing(
                     root, job, input_signature(job, context), args.resume) else "generate"}
                for job in jobs]

    planned = plan()  # Fail before importing/allocating inference dependencies.
    if args.dry_run:
        print(json.dumps({"dry_run": True, "inference_performed": False,
                          "reference_audio_decoded": False, "jobs": planned}, indent=2))
        return 0
    with output_lock(root):
        planned = plan()  # Recheck under the lock.
        if any(item["action"] == "generate" for item in planned):
            run_inference(args, jobs, root, model_dir, reference, ref_text, context)
    print(json.dumps({"status": "all_inference_complete", "jobs": len(jobs),
                      "semantic_listening_performed": False}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, ImportError, wave.Error, EOFError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
