"""Offline Qwen runner tests. Run from the release repository root.

No Qwen/torch/numpy/soundfile import or model download is needed. Synthetic PCM
fixtures are not speech and do not constitute a model-inference test.
"""
import ast
import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

RUNNER = Path(__file__).resolve().parents[1] / "skills" / "produce-science-videos" / "scripts" / "run_qwen3_tts.py"
spec = importlib.util.spec_from_file_location("qwen_runner", RUNNER)
qwen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen)
JOB = {"id": "demo_01", "text": "这是一个计时演示。", "language": "Chinese", "seed": 101}


def write_pcm(path, frames=100, value=b"\x01\x00"):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(value * frames)


class FakeSoundFile:
    @staticmethod
    def write(path, samples, rate, format, subtype):
        assert format == "WAV" and subtype == "PCM_16" and rate == 24000
        write_pcm(path)


class QwenRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"model_type": "qwen3_tts", "tts_model_type": "base"}), encoding="utf-8")
        self.reference = self.root / "reference.wav"
        write_pcm(self.reference)
        self.transcript = self.root / "reference.txt"
        self.transcript.write_text("这是一个计时演示。", encoding="utf-8")
        self.jobs = self.root / "jobs.json"
        self.jobs.write_text(json.dumps([JOB]), encoding="utf-8")
        self.output = self.root / "output"
        self.argv = ["--jobs", str(self.jobs), "--model-dir", str(self.model),
                     "--model-revision", "synthetic-fixture-v1", "--ref-audio", str(self.reference),
                     "--ref-text-file", str(self.transcript), "--output-dir", str(self.output)]

    def dry_run(self, *extra):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "soundfile", "qwen_tts"}:
                raise AssertionError(f"Unexpected inference import: {name}")
            return original_import(name, *args, **kwargs)
        captured = io.StringIO()
        with patch("builtins.__import__", guarded_import), contextlib.redirect_stdout(captured):
            self.assertEqual(qwen.main(self.argv + ["--dry-run", *extra]), 0)
        return json.loads(captured.getvalue())

    def existing_pair(self):
        signature = self.dry_run()["jobs"][0]["input_signature"]
        self.output.mkdir()
        wav, record = qwen.output_paths(self.output, JOB["id"])
        write_pcm(wav)
        data = {"schema": qwen.SCHEMA, "id": JOB["id"], "status": "inference_complete",
                "input_signature": signature, "wav": qwen.wav_info(wav)}
        record.write_text(json.dumps(data), encoding="utf-8")
        return signature, wav, record, data

    def test_syntax_and_valid_jobs(self):
        ast.parse(RUNNER.read_text(encoding="utf-8"))
        self.assertEqual(qwen.validate_jobs([JOB]), [JOB])

    def test_unsafe_and_reserved_ids(self):
        for identity in ("../escape", "a/b", "a\\b", "CON", "NUL", "com1", "a.b", "", "a" * 81):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                qwen.validate_jobs([{**JOB, "id": identity}])

    def test_duplicate_and_invalid_jobs(self):
        for jobs in ([], [JOB, {**JOB, "id": "DEMO_01"}], [{**JOB, "seed": True}],
                     [{**JOB, "seed": -1}], [{**JOB, "seed": 2**32}], [{**JOB, "text": " "}],
                     [{**JOB, "language": "unsupported"}], [{**JOB, "extra": 1}]):
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                qwen.validate_jobs(jobs)

    def test_duplicate_json_keys_rejected(self):
        self.jobs.write_text('{"id": "a", "id": "b"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            qwen.read_json(self.jobs)

    def test_signature_order_and_input_changes(self):
        self.assertEqual(qwen.canonical_hash({"a": 1, "b": 2}), qwen.canonical_hash({"b": 2, "a": 1}))
        baseline = qwen.input_signature(JOB, {"ref": "a"})
        self.assertNotEqual(baseline, qwen.input_signature(JOB, {"ref": "b"}))
        for key, value in (("text", "第二段保持同一条时钟。"), ("seed", 102), ("language", "Auto")):
            self.assertNotEqual(baseline, qwen.input_signature({**JOB, key: value}, {"ref": "a"}))

    def test_dry_run_no_output_or_inference_import(self):
        result = self.dry_run()
        self.assertFalse(self.output.exists())
        self.assertFalse(result["inference_performed"])
        self.assertFalse(result["reference_audio_decoded"])
        self.assertEqual(result["jobs"][0]["action"], "generate")

    def test_verified_resume_and_no_resume_rejection(self):
        signature, wav, record, data = self.existing_pair()
        self.assertEqual(qwen.check_existing(str(self.output), JOB, signature, True), data)
        self.assertEqual(self.dry_run("--resume")["jobs"][0]["action"], "verified_skip")
        with self.assertRaises(ValueError):
            qwen.check_existing(self.output, JOB, signature, False)

    def test_signature_mismatch_rejected(self):
        signature, wav, record, data = self.existing_pair()
        with self.assertRaises(ValueError):
            qwen.check_existing(self.output, JOB, "changed", True)

    def test_truncation_and_sha_mutation_rejected(self):
        signature, wav, record, data = self.existing_pair()
        original = wav.read_bytes()
        wav.write_bytes(original[:-2])
        with self.assertRaises((ValueError, wave.Error, EOFError)):
            qwen.check_existing(self.output, JOB, signature, True)
        wav.write_bytes(original[:-2] + b"\x02\x00")
        with self.assertRaises(ValueError):
            qwen.check_existing(self.output, JOB, signature, True)

    def test_orphan_and_partial_rejected(self):
        signature, wav, record, data = self.existing_pair()
        record.unlink()
        with self.assertRaises(ValueError):
            qwen.check_existing(self.output, JOB, signature, True)
        record.write_text(json.dumps(data), encoding="utf-8")
        (self.output / ".demo_01-test.partial").write_bytes(b"unfinished")
        with self.assertRaises(ValueError):
            qwen.check_existing(self.output, JOB, signature, True)

    def test_silent_wav_rejected(self):
        write_pcm(self.reference, value=b"\0\0")
        with self.assertRaises(ValueError):
            qwen.wav_info(self.reference)

    def test_no_clobber_publication(self):
        target, temporary = self.root / "target", self.root / "temporary"
        target.write_bytes(b"old")
        temporary.write_bytes(b"new")
        with self.assertRaises(FileExistsError):
            qwen.publish_no_clobber(temporary, target)
        self.assertEqual(target.read_bytes(), b"old")
        self.assertEqual(temporary.read_bytes(), b"new")
        target.unlink()
        qwen.publish_no_clobber(temporary, target)
        self.assertEqual(target.read_bytes(), b"new")
        self.assertFalse(temporary.exists())

    def test_mock_save_and_exclusive_lock(self):
        signature = qwen.input_signature(JOB, {})
        with qwen.output_lock(self.output):
            with self.assertRaises(ValueError):
                with qwen.output_lock(self.output):
                    self.fail("Acquired a second lock")
            row = qwen.save_result(self.output, JOB, signature, {}, [], 24000, 0.1, FakeSoundFile)
            self.assertEqual(qwen.check_existing(self.output, JOB, signature, True), row)
            self.assertFalse(row["semantic_listening_performed"])
            self.assertEqual(row["listening_status"], "pending")
        self.assertFalse((self.output / ".qwen3-tts.lock").exists())

    def test_symlink_root_and_output_rejected(self):
        self.output.mkdir()
        link = self.root / "link"
        try:
            link.symlink_to(self.output, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("OS does not allow test symlinks")
        with self.assertRaises(ValueError):
            qwen.safe_output_root(link / "child")
        target = self.output / "demo_01.wav"
        target.symlink_to(self.reference)
        with self.assertRaises(ValueError):
            qwen.regular_output(target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
