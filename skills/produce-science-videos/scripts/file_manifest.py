#!/usr/bin/env python3
"""Create/verify a portable SHA-256 manifest; never delete, upload, or unpack files."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_path(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("root must be a real directory, not a symlink")
    return path.resolve()


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\x00" in relative:
        raise ValueError("invalid portable relative path")
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or any(part in (".", "..") for part in parts) or "/".join(parts) != relative:
        raise ValueError("path is not a normalized relative path")
    target = root
    for part in parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"symlink is not allowed: {relative}")
    if not target.resolve().is_relative_to(root):
        raise ValueError("path escapes the root")
    return target


def list_files(root, exclude=None):
    root = root_path(root)
    excluded = Path(exclude).resolve() if exclude is not None else None
    paths = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"symlink is not allowed: {path.relative_to(root)}")
        for name in files:
            path = Path(directory) / name
            if path.resolve() != excluded:
                relative = path.relative_to(root).as_posix()
                if not safe_path(root, relative).is_file():
                    raise ValueError(f"not a regular file: {relative}")
                paths.append(path)
    return root, sorted(paths)


def create_manifest(root, exclude=None):
    root, paths = list_files(root, exclude)
    return {"schema_version": 1, "algorithm": "sha256", "files": [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths]}


def verify_manifest(root, manifest, manifest_path=None, allow_extra=False):
    root = root_path(root)
    if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256" or not isinstance(manifest.get("files"), list):
        raise ValueError("unsupported manifest schema")
    expected, failures = set(), []
    for entry in manifest["files"]:
        relative = entry.get("path")
        path = safe_path(root, relative)
        if relative in expected:
            raise ValueError(f"duplicate manifest path: {relative}")
        expected.add(relative)
        size, digest = entry.get("bytes"), entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid byte count: {relative}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA-256: {relative}")
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
        elif path.stat().st_size != size or sha256(path) != digest:
            failures.append({"path": relative, "reason": "size or SHA-256 mismatch"})
    _, actual_paths = list_files(root, manifest_path)
    actual = {p.relative_to(root).as_posix() for p in actual_paths}
    extras = sorted(actual - expected)
    if extras and not allow_extra:
        failures.extend({"path": p, "reason": "unexpected extra file"} for p in extras)
    return {"passed": not failures, "checked_files": len(expected), "extra_files": extras, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("root", type=Path)
    create.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("root", type=Path)
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--allow-extra", action="store_true", help="permit unlisted files; listed files are still verified")
    args = parser.parse_args()
    try:
        if args.command == "create":
            if args.output.is_symlink():
                raise ValueError("refusing symlink output")
            content = create_manifest(args.root, args.output)
            args.output.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"files": len(content["files"]), "manifest": str(args.output)}, ensure_ascii=False))
        else:
            result = verify_manifest(args.root, json.loads(args.manifest.read_text(encoding="utf-8-sig")), args.manifest, args.allow_extra)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result["passed"]:
                raise SystemExit(1)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
