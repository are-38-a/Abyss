"""Tests for serve/ds41_vision.py (DeepSeek-V4.1's pictures: the resize plan, the markers and the spans).

    python -m unittest serve.test_ds41_vision -v
"""
from __future__ import annotations

import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import ds41_vision as dv  # noqa: E402


def encode(text: str) -> list[int]:
    return [ord(c) for c in text]


class PlanTest(unittest.TestCase):
    def test_span_fits_the_cap(self):
        g = dv.Geometry()
        for w, h in ((1920, 1080), (1080, 1920), (4000, 300), (300, 4000), (640, 640)):
            n_h, n_w, bh, bw = dv.plan(w, h, g)
            self.assertLessEqual(dv.span_tokens(n_h, n_w), g.max_tokens, (w, h))
            self.assertEqual(bh % g.patch, 0)
            self.assertEqual(bw % g.patch, 0)

    def test_small_picture_grows_to_min_pixels(self):
        g = dv.Geometry()
        self.assertEqual(dv.plan(100, 100, g), (13, 13, 546, 546))   # 544 -> 546 pixels, 39 patches, 13 cells

    def test_geometry_from_vision_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "vision.json"
            p.write_text(json.dumps({"config": {"vision.patch_size": 14, "vision.downsample_ratio": 3,
                                                "vision.max_image_tokens": 1024, "vision.min_pixels": 295936,
                                                "image_token_id": 129264}}), encoding="utf-8")
            g = dv.Geometry.from_json(p, max_tokens=512)
        self.assertEqual((g.max_tokens, g.min_pixels, g.image_token), (512, 295936, 129264))


class PrepareTest(unittest.TestCase):
    def test_resized_pixels_and_grid(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        buf = io.BytesIO()
        Image.new("RGB", (800, 600), (10, 20, 30)).save(buf, format="PNG")
        pic = dv.prepare(buf.getvalue(), dv.Geometry())
        self.assertEqual(len(pic.rgb), pic.width * pic.height * 3)
        self.assertEqual((pic.width % 14, pic.height % 14), (0, 0))
        self.assertEqual(pic.tokens, dv.span_tokens(pic.n_h, pic.n_w))
        with tempfile.TemporaryDirectory() as d:
            path = dv.write_dsr(Path(d) / "a.dsr", pic)
            data = path.read_bytes()
        self.assertEqual(struct.unpack("<4sii", data[:12]), (b"DSR1", pic.width, pic.height))
        self.assertEqual(len(data), 12 + len(pic.rgb))


class MarkersTest(unittest.TestCase):
    def messages(self):
        return [{"role": "system", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "look: "},
                                             {"type": "image", "source": "a.png"},
                                             {"type": "text", "text": " and <｜deepseek_image｜> stays text"},
                                             {"type": "image", "source": "b.png"}]}]

    def test_markers_become_spans_in_prompt_order(self):
        marked, pat = dv.mark_images(self.messages())
        self.assertIsInstance(marked[1]["content"], str)
        prompt = "S" + marked[1]["content"] + "E"
        ids, order = dv.expand(prompt, pat, encode, [3, 5], image_token=-7)
        self.assertEqual(order, [0, 1])
        self.assertEqual(ids.count(-7), 8)
        text = "".join(chr(i) for i in ids if i != -7)
        self.assertEqual(text, "Slook:  and <｜deepseek_image｜> stays textE")
        self.assertEqual(ids[len(encode("Slook: ")):len(encode("Slook: ")) + 3], [-7, -7, -7])

    def test_a_template_that_reorders_keeps_each_picture_with_its_marker(self):
        marked, pat = dv.mark_images(self.messages())
        m = list(pat.finditer(marked[1]["content"]))
        prompt = m[1].group(0) + "x" + m[0].group(0)          # the second picture rendered first
        ids, order = dv.expand(prompt, pat, encode, [2, 4], image_token=-7)
        self.assertEqual(order, [1, 0])
        self.assertEqual(ids, [-7] * 4 + encode("x") + [-7] * 2)

    def test_missing_or_repeated_marker_is_an_error(self):
        marked, pat = dv.mark_images(self.messages())
        m = list(pat.finditer(marked[1]["content"]))
        with self.assertRaises(ValueError):
            dv.expand(m[0].group(0), pat, encode, [2, 4])
        with self.assertRaises(ValueError):
            dv.expand(m[0].group(0) * 2 + m[1].group(0), pat, encode, [2, 4])

    def test_the_image_tokens_text_in_a_message_stays_text(self):
        def enc(s):   # a tokenizer that makes the placeholder's text the image token, as parse_special does
            out = []
            for i, part in enumerate(s.split(dv.PLACEHOLDER)):
                out += ([-7] if i else []) + encode(part)
            return out
        literal = dv.literal_ids(enc, image_token=-7)
        self.assertNotIn(-7, literal)
        self.assertEqual("".join(map(chr, literal)), dv.PLACEHOLDER)
        marked, pat = dv.mark_images(self.messages())
        ids, _ = dv.expand(marked[1]["content"], pat, enc, [3, 5], image_token=-7, literal=literal)
        self.assertEqual(ids.count(-7), 8)
        self.assertIn(dv.PLACEHOLDER, "".join(chr(i) for i in ids if i != -7))

    def test_text_only_messages_are_unchanged(self):
        msgs = [{"role": "user", "content": "plain"}]
        marked, _ = dv.mark_images(msgs)
        self.assertEqual(marked, msgs)


if __name__ == "__main__":
    unittest.main()
