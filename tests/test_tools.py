"""Offline behavior tests. All generated files stay in temporary directories."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "produce-science-videos" / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timeline = load("timeline_plan")
captions = load("captions")
manifest = load("file_manifest")
media = load("media_audit")


class TimelineTests(unittest.TestCase):
    def test_cumulative_boundaries(self):
        data = {"sample_rate": 24000, "fps": "30", "clips": [{"id": "a", "samples": 24001}, {"id": "b", "samples": 23999}]}
        plan = timeline.build_timeline(data)
        self.assertEqual(plan["total_frames"], 60)
        self.assertEqual([c["frames"] for c in plan["clips"]], [31, 29])
        self.assertEqual(plan["clips"][0]["end_frame_exclusive"], plan["clips"][1]["first_frame"])

    def test_fractional_rate_full_tail(self):
        plan = timeline.build_timeline({"sample_rate": 48000, "fps": "30000/1001", "clips": [{"id": "a", "samples": 48001}]})
        self.assertGreaterEqual(plan["video_seconds"], plan["audio_seconds"])
        self.assertLess(plan["video_seconds"] - plan["audio_seconds"], 1001 / 30000)

    def test_many_short_clips_do_not_accumulate_frames(self):
        plan = timeline.build_timeline({"sample_rate": 24000, "fps": "30", "clips": [{"id": f"c{i}", "samples": 1} for i in range(1000)]})
        self.assertEqual(plan["total_frames"], 2)
        self.assertEqual(sum(c["frames"] for c in plan["clips"]), 2)

    def test_invalid_inputs(self):
        for rate, fps, clips in [(True, "30", [{"id": "a", "samples": 1}]), (24000, "0", []), (24000, "30", [{"id": "../a", "samples": 1}]), (24000, "30", [{"id": "a", "samples": 1, "sample_rate": 44100}])]:
            with self.subTest(rate=rate, fps=fps, clips=clips), self.assertRaises(ValueError):
                timeline.build_timeline({"sample_rate": rate, "fps": fps, "clips": clips})


class CaptionTests(unittest.TestCase):
    def test_unicode_line_break_and_exact_time(self):
        output = captions.to_srt({"sample_rate": 24000, "cues": [{"start_sample": 0, "end_sample": 24000, "text": "科学\n演示。"}]})
        self.assertEqual(output, "1\n00:00:00,000 --> 00:00:01,000\n科学\n演示。\n\n")

    def test_overlap_and_collapsed_cue_rejected(self):
        examples = [[{"start_sample": 0, "end_sample": 1, "text": "x"}], [{"start_sample": 0, "end_sample": 100, "text": "x"}, {"start_sample": 90, "end_sample": 200, "text": "y"}]]
        for cues in examples:
            with self.subTest(cues=cues), self.assertRaises(ValueError):
                captions.to_srt({"sample_rate": 24000, "cues": cues})

    def test_blank_line_rejected(self):
        with self.assertRaises(ValueError):
            captions.to_srt({"sample_rate": 1000, "cues": [{"start_sample": 0, "end_sample": 10, "text": "one\n\ntwo"}]})


class ManifestTests(unittest.TestCase):
    def test_restore_corruption_and_extra_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "text.txt").write_text("通用测试", encoding="utf-8")
            contents = manifest.create_manifest(root)
            self.assertTrue(manifest.verify_manifest(root, contents)["passed"])
            (root / "extra.txt").write_text("extra")
            self.assertFalse(manifest.verify_manifest(root, contents)["passed"])
            self.assertTrue(manifest.verify_manifest(root, contents, allow_extra=True)["passed"])
            (root / "nested" / "text.txt").write_text("changed")
            self.assertFalse(manifest.verify_manifest(root, contents, allow_extra=True)["passed"])

    def test_isolated_copy_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.mkdir()
            (original / "source.txt").write_text("可恢复测试", encoding="utf-8")
            contents = manifest.create_manifest(original)
            restored = root / "restored"
            shutil.copytree(original, restored)
            (original / "source.txt").write_text("old source changed")
            self.assertTrue(manifest.verify_manifest(restored, contents)["passed"])
            self.assertFalse(manifest.verify_manifest(original, contents)["passed"])

    def test_self_manifest_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("a")
            output = root / "manifest.json"
            contents = manifest.create_manifest(root, output)
            output.write_text(json.dumps(contents))
            self.assertTrue(manifest.verify_manifest(root, contents, output)["passed"])

    def test_traversal_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ["../a", "/a", "a/../b", "a//b", "a\\b", "C:/a"]:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    manifest.safe_path(root, path)
            (root / "a").write_text("a")
            contents = manifest.create_manifest(root)
            contents["files"].append(dict(contents["files"][0]))
            with self.assertRaises(ValueError):
                manifest.verify_manifest(root, contents)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_text("a")
            try:
                (root / "alias").symlink_to(root / "a")
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(ValueError):
                manifest.create_manifest(root)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
class MediaIntegrationTests(unittest.TestCase):
    def test_decode_and_stream_copy_and_changed_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, copied, changed = (root / name for name in ("first.mp4", "copied.mp4", "changed.mp4"))
            command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=0.5", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=24000:duration=0.5", "-c:v", "mpeg4", "-c:a", "aac", "-threads", "1", str(first)]
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(first), "-c", "copy", str(copied)], check=True, capture_output=True)
            result = media.audit(first, decode=True, compare_video=copied, compare_audio=copied)
            self.assertTrue(result["passed"])
            self.assertEqual(result["full_decode"], "passed")
            self.assertFalse(result["subjective_listening"])
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(first), "-c:v", "copy", "-af", "volume=0.5", "-c:a", "aac", str(changed)], check=True, capture_output=True)
            self.assertTrue(media.compare_stream(first, changed, "v:0")["identical_packet_payloads"])
            self.assertFalse(media.compare_stream(first, changed, "a:0")["identical_packet_payloads"])


if __name__ == "__main__":
    unittest.main()
