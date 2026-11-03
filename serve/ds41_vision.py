"""serve/ds41_vision.py - pictures for DeepSeek-V4.1 (its encoder: tools/ds41_vision, `ds41-vision`).

The server's half of DeepSeek's inference/image_processor.py (as Project Maya's serve/ds4_vision.py): a picture is
resized (aspect kept, gray padding) to a pixel size whose 14 x 14 patch grid, merged 3 x 3 by the aligner, costs at
most `max_tokens` prompt tokens.  In the prompt it becomes the span

    [image_start] + ([cell] * n_w + [image_newline]) * n_h + [image_end]

every position of which carries the image token <｜deepseek_image｜>; the encoder writes the span's rows whole.

DeepSeek-V4.1's chat template does not render images, so the server writes a marker of its own where each picture
sits (`mark_images`), renders, and puts the picture's span where its marker is (`expand`): a message that contains the
image token's text keeps it as text.
"""
from __future__ import annotations

import io
import json
import math
import re
import secrets
import struct
from dataclasses import dataclass
from pathlib import Path

IMAGE_TOKEN = 129264   # <｜deepseek_image｜>
PLACEHOLDER = "<｜deepseek_image｜>"


def literal_ids(encode_plain, image_token: int = IMAGE_TOKEN) -> list[int]:
    """The image token's text as plain-text ids: encoded in two halves, which no special token matches."""
    ids = encode_plain(PLACEHOLDER)
    if image_token not in ids:
        return ids
    return encode_plain(PLACEHOLDER[:6]) + encode_plain(PLACEHOLDER[6:])


@dataclass(frozen=True)
class Geometry:
    patch: int = 14
    merge: int = 3
    min_pixels: int = 544 * 544
    max_tokens: int = 1024
    max_wh_ratio: int | None = None
    image_token: int = IMAGE_TOKEN

    @classmethod
    def from_json(cls, path: str | Path, max_tokens: int | None = None) -> "Geometry":
        """The vision weights' config (tools/ds4/vision_pack.py's vision.json); a missing key keeps its default."""
        cfg = json.loads(Path(path).read_text(encoding="utf-8")).get("config", {})

        def num(k, d):
            v = cfg.get(k)
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else d
        model_max = num("vision.max_image_tokens", 1024)
        ratio = num("vision.max_wh_ratio", 0)
        return cls(patch=num("vision.patch_size", 14), merge=num("vision.downsample_ratio", 3),
                   min_pixels=num("vision.min_pixels", 544 * 544),
                   max_tokens=min(max_tokens, model_max) if max_tokens else model_max,
                   max_wh_ratio=ratio or None, image_token=num("image_token_id", IMAGE_TOKEN))


def span_tokens(n_h: int, n_w: int) -> int:
    return n_h * (n_w + 1) + 2


def _llm_grid(best_h: int, best_w: int, g: Geometry) -> tuple[int, int]:
    return math.ceil((best_h // g.patch) / g.merge), math.ceil((best_w // g.patch) / g.merge)


def _solve_resize_ratio(height, width, g: Geometry) -> tuple[int, int]:
    """Largest aspect-preserving pixel size whose token grid still fits in max_tokens."""
    r = height / width
    max_w = math.sqrt((g.max_tokens - 2) / r + 0.25) - 0.5
    max_h = max_w * r
    cell = g.patch * g.merge
    if max_w < 1.0:   # very tall: a single column
        return (g.max_tokens - 2) // 2 * cell, cell
    if max_h < 1.0:   # very wide: a single row
        return cell, (g.max_tokens - 3) * cell
    beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
    return math.floor(height * beta / g.patch) * g.patch, math.floor(width * beta / g.patch) * g.patch


def plan(width: int, height: int, g: Geometry) -> tuple[int, int, int, int]:
    """-> (n_h, n_w, best_height, best_width): the reference's plan_image_grid for a picture of this size."""
    p = g.patch
    if g.max_wh_ratio is not None and width > height * g.max_wh_ratio:
        width = height * g.max_wh_ratio
    if 0 < width * height < g.min_pixels:
        ratio = (g.min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_w, best_h = math.ceil(width / p) * p, math.ceil(height / p) * p
    n_h, n_w = _llm_grid(best_h, best_w, g)
    if span_tokens(n_h, n_w) > g.max_tokens:
        best_h, best_w = _solve_resize_ratio(height, width, g)
        n_h, n_w = _llm_grid(best_h, best_w, g)
    return n_h, n_w, best_h, best_w


@dataclass
class Picture:
    rgb: bytes          # height x width x 3
    width: int
    height: int
    n_h: int
    n_w: int

    @property
    def tokens(self) -> int:
        return span_tokens(self.n_h, self.n_w)


def prepare(data: bytes, g: Geometry) -> Picture:
    """A picture's bytes (any format Pillow reads) -> its resized pixels and grid."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise ValueError("pictures need Pillow (python -m pip install pillow)") from None
    try:
        with Image.open(io.BytesIO(data)) as src:
            src.load()
            im = ImageOps.exif_transpose(src)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
    except Exception as e:  # noqa: BLE001 - whatever Pillow raises for a file it cannot read
        raise ValueError(f"the image could not be read ({e})") from None
    n_h, n_w, best_h, best_w = plan(im.width, im.height, g)
    if g.max_wh_ratio is not None and im.width >= g.max_wh_ratio * im.height:
        im = im.resize((best_w, best_h))
    else:
        im = ImageOps.pad(im, (best_w, best_h), color=(127, 127, 127))
    rgb = im.tobytes()
    if len(rgb) != best_w * best_h * 3:
        raise ValueError("the image could not be resized")
    return Picture(rgb, best_w, best_h, n_h, n_w)


def write_dsr(path: Path, pic: Picture) -> Path:
    """The encoder's input: int32 'DSR1', int32 width, int32 height, then the RGB bytes."""
    with open(path, "wb") as f:
        f.write(struct.pack("<4sii", b"DSR1", pic.width, pic.height))
        f.write(pic.rgb)
    return path


def mark_images(messages: list[dict]) -> tuple[list[dict], re.Pattern]:
    """Each message whose content is the template's list form (text and image items) -> its text, with a marker of
    this request's own where each image item was (the k-th image of the conversation: `\\ue000<nonce>:<k>\\ue001`).
    -> (the messages, the markers' pattern: group 1 is k)."""
    nonce = secrets.token_hex(6)
    out, k = [], 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    parts.append(f"{nonce}:{k}")
                    k += 1
                else:
                    parts.append(item.get("text", ""))
            m = dict(m, content="".join(parts))
        out.append(m)
    return out, re.compile("" + nonce + r":(\d+)")


def expand(prompt: str, pattern: re.Pattern, encode, spans: list[int], image_token: int = IMAGE_TOKEN,
           literal: list[int] | None = None) -> tuple[list[int], list[int]]:
    """The rendered prompt -> its ids, each image marker replaced by its picture's span (spans[k] image tokens).
    The image token's own text in the prompt (a message that quotes it) becomes `literal`, its ids as plain text, so
    only the markers are pictures.  -> (ids, the pictures' k in the order the prompt has them)."""
    def text(s: str) -> list[int]:
        out = encode(s)
        if literal is None or image_token not in out:
            return out
        return [x for t in out for x in (literal if t == image_token else (t,))]

    ids: list[int] = []
    order: list[int] = []
    at = 0
    for m in pattern.finditer(prompt):
        ids += text(prompt[at:m.start()])
        k = int(m.group(1))
        if k >= len(spans) or k in order:
            raise ValueError("the prompt and its images do not match")
        ids += [image_token] * spans[k]
        order.append(k)
        at = m.end()
    ids += text(prompt[at:])
    if len(order) != len(spans):
        raise ValueError("the prompt and its images do not match")
    return ids, order
