# -*- coding: utf-8 -*-
"""
SegRig Studio — Local web app for automated 2D character rigging prep.
================================================================================
Pipeline:  Image -> LangSAM (text-prompted instance segmentation)
                 -> per-instance transparent PNG layers
                 -> [optional] SD1.5 Inpainting background reconstruction
                 -> PSD (AE "Retain Layer Sizes" compatible) + ZIP export

Hardware target: RTX 2060 Super (8 GB VRAM) / i5-14400F / 32 GB RAM.

MEMORY DOCTRINE (strictly enforced):
    Only ONE heavy model may reside in VRAM at any moment. The segmenter is
    fully torn down (moved to CPU -> dereferenced -> gc -> empty_cache) BEFORE
    the diffusion pipeline is constructed, and vice-versa. Every stage
    transition passes through `hard_flush()`.

ARCHITECTURE NOTE (for your AE / C4D pipeline work):
    All business logic lives in pure, UI-agnostic functions and the
    `SegRigPipeline` class. The Gradio layer at the bottom is a thin adapter.
    You can `from segrig_studio import SegRigPipeline` in a headless script,
    or drive it from a CLI / watch-folder / AE ExtendScript bridge without
    touching Gradio.
================================================================================
"""

from __future__ import annotations
# -*- coding: utf-8 -*-
# ==============================================================================
# 0. PATH SETUP — ต้องอยู่ก่อน import torch / gradio เสมอ
# ==============================================================================
import os as _os

AI_ROOT = r"C:\AI"          # <<< แก้ไดรฟ์ที่นี่จุดเดียว ถ้าย้ายทีหลัง

for _k, _v in {
    "TORCH_HOME":      _os.path.join(AI_ROOT, "cache", "torch"),
    "GRADIO_TEMP_DIR": _os.path.join(AI_ROOT, "cache", "gradio"),
}.items():
    _os.environ.setdefault(_k, _v)
    _os.makedirs(_os.environ[_k], exist_ok=True)

_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---- import อื่น ๆ ทั้งหมดตามหลังบรรทัดนี้ ----
import gc
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================


class Config:
    """Central tuning knobs. Edit here, not in the logic below."""

    # ---- Devices -------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    # ---- Inpainting model ----------------------------------------------------
    # runwayml/* was taken down in Aug 2024. Tried in order, first hit wins.
    INPAINT_MODEL_CANDIDATES: Sequence[str] = (
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "botp/stable-diffusion-v1-5-inpainting",
        "runwayml/stable-diffusion-inpainting",  # legacy / local cache only
    )

    # ---- Segmentation --------------------------------------------------------
    BOX_THRESHOLD = 0.30          # GroundingDINO detection confidence
    TEXT_THRESHOLD = 0.25         # GroundingDINO phrase-grounding confidence
    MIN_MASK_AREA_RATIO = 0.0004  # discard specks < 0.04% of frame
    DEDUPE_IOU = 0.85             # merge near-duplicate instances

    # ---- Mask shaping --------------------------------------------------------
    DILATE_PX_DEFAULT = 12        # halo-killer dilation for inpaint mask
    ALPHA_FEATHER_PX = 1          # sub-pixel edge softening on exported layers

    # ---- Inpainting tiling (VRAM-safe) --------------------------------------
    TILE = 512                    # SD1.5 native resolution
    TILE_OVERLAP = 128            # blend seam width
    INPAINT_STEPS = 28
    GUIDANCE_SCALE = 7.5
    DEFAULT_POSITIVE = "seamless background, complete body, high quality, clean, coherent lighting"
    DEFAULT_NEGATIVE = "text, watermark, logo, extra limbs, deformed, blurry, jpeg artifacts, seam"

    # ---- IO ----
    # เดิม: WORK_ROOT = os.path.join(tempfile.gettempdir(), "segrig_studio")
    WORK_ROOT = os.path.join(AI_ROOT, "segrig_output")
    MAX_SIDE = 2048               # downscale monsters to keep 8 GB happy


os.makedirs(Config.WORK_ROOT, exist_ok=True)


# ==============================================================================
# 2. MEMORY MANAGEMENT UTILITIES
# ==============================================================================


def vram_report(tag: str = "") -> str:
    """Human-readable VRAM snapshot. Cheap, safe to call anywhere."""
    if not torch.cuda.is_available():
        return f"[{tag}] CPU mode"
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    resv = torch.cuda.memory_reserved() / 1024 ** 3
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return f"[{tag}] VRAM alloc {alloc:.2f} GB | reserved {resv:.2f} GB | total {total:.1f} GB"


def hard_flush(tag: str = "flush") -> None:
    """
    The single most important function in this file.

    Called at EVERY stage transition. Runs the garbage collector twice
    (once to break reference cycles created by torch modules, once to actually
    free them), then returns cached blocks to the driver and resets the
    allocator's peak stats.
    """
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    print(vram_report(tag))


def _deep_to_cpu(obj, _depth: int = 0) -> None:
    """
    Recursively walk an object graph and evict any nn.Module to CPU.

    LangSAM wraps GroundingDINO + SAM2 in nested attributes whose exact names
    change between releases, so we sweep rather than hard-code paths.
    """
    if _depth > 3 or obj is None:
        return
    try:
        if isinstance(obj, torch.nn.Module):
            obj.to("cpu")
            return
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if isinstance(child, torch.nn.Module):
                try:
                    child.to("cpu")
                except Exception:
                    pass
            elif hasattr(child, "__dict__") and not callable(child):
                _deep_to_cpu(child, _depth + 1)
    except Exception:
        pass


@contextmanager
def exclusive_gpu(tag: str):
    """
    Context manager guaranteeing a clean VRAM slate on entry AND exit.

    Usage:
        with exclusive_gpu("segmentation"):
            ...load model, run, delete...
    """
    hard_flush(f"{tag}:enter")
    t0 = time.time()
    try:
        yield
    finally:
        hard_flush(f"{tag}:exit ({time.time() - t0:.1f}s)")


# ==============================================================================
# 3. DATA MODEL
# ==============================================================================


@dataclass
class LayerData:
    """
    One extracted character part.

    `mask` is full-canvas (H, W) uint8 {0,255} — the source of truth.
    `bbox` is the tight (x0, y0, x1, y1) crop used for PSD layer sizing,
    which is what makes AE's "Retain Layer Sizes" import land pixel-perfect.
    """
    name: str                      # e.g. "hair_1"
    tag: str                       # e.g. "hair"
    index: int                     # 1-based instance number within the tag
    mask: np.ndarray = field(repr=False)
    score: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def area(self) -> int:
        return int((self.mask > 0).sum())

    def to_manifest(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        return {
            "name": self.name,
            "tag": self.tag,
            "index": self.index,
            "score": round(float(self.score), 4),
            "offset_x": int(x0),
            "offset_y": int(y0),
            "width": int(x1 - x0),
            "height": int(y1 - y0),
            "pixel_area": self.area,
        }


@dataclass
class PipelineResult:
    base_rgb: np.ndarray
    layers: List[LayerData]
    background_rgb: np.ndarray
    zip_path: Optional[str] = None
    psd_path: Optional[str] = None
    log: str = ""


# ==============================================================================
# 4. IMAGE / MASK HELPERS  (pure numpy + cv2, no GPU)
# ==============================================================================


def to_rgb_array(image) -> np.ndarray:
    """Normalize any Gradio/PIL/ndarray input to a contiguous uint8 RGB array."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return np.ascontiguousarray(arr.astype(np.uint8))


def clamp_resolution(rgb: np.ndarray, max_side: int = Config.MAX_SIDE) -> np.ndarray:
    """Downscale oversized input — the #1 cause of OOM on 8 GB cards."""
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    scale = max_side / float(max(h, w))
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """
    Elliptical dilation — REQUIREMENT #2.

    Growing the mask by ~10-15 px before inpainting pushes the diffusion
    boundary past the anti-aliased edge pixels of the extracted part, which is
    exactly what removes the coloured halo / ghost outline in the plate.
    """
    if px <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, k, iterations=1)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Close pinholes and drop disconnected noise blobs."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight bounding box as (x0, y0, x1, y1) exclusive-end, or None if empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    ab = np.logical_and(a > 0, b > 0).sum()
    ub = np.logical_or(a > 0, b > 0).sum()
    return float(ab) / float(ub) if ub else 0.0


def dedupe_masks(masks: List[np.ndarray], scores: List[float],
                 iou_thr: float = Config.DEDUPE_IOU) -> List[int]:
    """Greedy score-ordered NMS in mask space. Returns kept indices."""
    order = sorted(range(len(masks)), key=lambda i: -scores[i])
    kept: List[int] = []
    for i in order:
        if all(mask_iou(masks[i], masks[j]) < iou_thr for j in kept):
            kept.append(i)
    return kept


def sort_instances(masks: List[np.ndarray]) -> List[int]:
    """
    Deterministic instance ordering: left-to-right, then top-to-bottom.

    This is what makes `hair_1` / `hair_2` stable across re-runs — critical
    if you're re-linking layers to an existing AE rig.
    """
    def key(i: int):
        bb = mask_bbox(masks[i])
        return (bb[0], bb[1]) if bb else (10 ** 9, 10 ** 9)
    return sorted(range(len(masks)), key=key)


def extract_rgba(base_rgb: np.ndarray, mask: np.ndarray,
                 feather: int = Config.ALPHA_FEATHER_PX) -> np.ndarray:
    """
    Cut a full-canvas RGBA plate.

    Colour is *not* premultiplied and RGB is preserved under transparent
    pixels — this avoids dark fringing when AE composites the layer.
    """
    alpha = mask.copy()
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (feather * 2 + 1, feather * 2 + 1), 0)
    rgba = np.dstack([base_rgb, alpha]).astype(np.uint8)
    return rgba


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", text.strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "part"


def parse_tags(raw: str) -> List[str]:
    """'hair, face , arms,' -> ['hair', 'face', 'arms'] (order preserved)."""
    seen, out = set(), []
    for t in raw.split(","):
        t = t.strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ==============================================================================
# 5. STAGE A — SEGMENTATION (LangSAM)
# ==============================================================================


class SegmenterHandle:
    """
    Short-lived owner of the LangSAM model.

    Deliberately NOT a singleton: the model is built, used, and destroyed
    inside a single `with` block so its ~2.5 GB never coexists with SD.
    """

    def __init__(self):
        from lang_sam import LangSAM  # imported late: heavy + optional
        print(vram_report("segmenter:before-load"))
        self.model = LangSAM()
        print(vram_report("segmenter:loaded"))

    # -- API compatibility shim ------------------------------------------------
    def _predict_raw(self, pil: Image.Image, prompt: str):
        """
        Normalizes the two LangSAM generations into
        (masks: np[N,H,W], scores: List[float]).
        """
        # New batched API (returns list of dicts)
        try:
            res = self.model.predict(
                [pil], [prompt],
                box_threshold=Config.BOX_THRESHOLD,
                text_threshold=Config.TEXT_THRESHOLD,
            )
            if isinstance(res, (list, tuple)) and res and isinstance(res[0], dict):
                d = res[0]
                masks = np.asarray(d.get("masks", []))
                scores = d.get("scores", d.get("mask_scores", []))
                scores = np.asarray(scores).reshape(-1).tolist() if len(np.asarray(scores)) else [1.0] * len(masks)
                return masks, scores
        except TypeError:
            pass
        except Exception as e:
            print(f"  [warn] batched predict failed ({e}); trying legacy API")

        # Legacy API: masks, boxes, phrases, logits
        masks, boxes, phrases, logits = self.model.predict(
            pil, prompt,
            box_threshold=Config.BOX_THRESHOLD,
            text_threshold=Config.TEXT_THRESHOLD,
        )
        masks = masks.cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        logits = logits.cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
        scores = logits.reshape(-1).tolist() if logits.size else [1.0] * len(masks)
        return masks, scores

    @torch.inference_mode()
    def segment_tag(self, rgb: np.ndarray, tag: str) -> Tuple[List[np.ndarray], List[float]]:
        """
        Segment ALL instances of one tag — REQUIREMENT #1.

        We query one tag at a time rather than a single concatenated prompt.
        Reasons: (a) GroundingDINO phrase->box attribution across a long prompt
        is unreliable, so per-tag queries give unambiguous naming; (b) peak
        VRAM stays flat instead of scaling with prompt length.
        """
        h, w = rgb.shape[:2]
        pil = Image.fromarray(rgb)
        prompt = tag if tag.endswith(".") else f"{tag}."

        raw_masks, scores = self._predict_raw(pil, prompt)
        if raw_masks is None or len(raw_masks) == 0:
            return [], []

        masks: List[np.ndarray] = []
        for m in raw_masks:
            m = np.asarray(m)
            if m.ndim == 3:
                m = m[0]
            m = (m > 0.5).astype(np.uint8) * 255
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(clean_mask(m))

        # Filter specks, then NMS duplicates
        min_area = Config.MIN_MASK_AREA_RATIO * h * w
        keep = [i for i, m in enumerate(masks) if (m > 0).sum() >= min_area]
        masks = [masks[i] for i in keep]
        scores = [float(scores[i]) if i < len(scores) else 1.0 for i in keep]
        if not masks:
            return [], []

        keep = dedupe_masks(masks, scores)
        masks = [masks[i] for i in keep]
        scores = [scores[i] for i in keep]

        # Deterministic left-to-right numbering
        order = sort_instances(masks)
        return [masks[i] for i in order], [scores[i] for i in order]

    def release(self) -> None:
        """Evict to CPU, drop every reference, then flush."""
        try:
            _deep_to_cpu(self.model)
        except Exception:
            pass
        self.model = None
        del self.model
        hard_flush("segmenter:released")


def run_segmentation(rgb: np.ndarray, tags: List[str],
                     progress_cb=None) -> Tuple[List[LayerData], List[str]]:
    """
    STAGE A entry point. Loads LangSAM, produces named layers, fully unloads.

    Returns (layers, log_lines). Guaranteed to leave VRAM clean.
    """
    log: List[str] = []
    layers: List[LayerData] = []
    handle: Optional[SegmenterHandle] = None

    with exclusive_gpu("STAGE-A/segmentation"):
        try:
            handle = SegmenterHandle()
            for ti, tag in enumerate(tags):
                if progress_cb is not None:
                    progress_cb((ti + 1) / max(len(tags), 1) * 0.5,
                                desc=f"Segmenting '{tag}' ({ti + 1}/{len(tags)})")
                masks, scores = handle.segment_tag(rgb, tag)
                if not masks:
                    log.append(f"  ✗ '{tag}': no instances found")
                    continue
                base = slugify(tag)
                for i, (m, s) in enumerate(zip(masks, scores), start=1):
                    # Single instance -> "hair"; multiple -> "hair_1", "hair_2"
                    name = f"{base}_{i}" if len(masks) > 1 else base
                    bb = mask_bbox(m)
                    if bb is None:
                        continue
                    layers.append(LayerData(name=name, tag=base, index=i,
                                            mask=m, score=s, bbox=bb))
                log.append(f"  ✓ '{tag}': {len(masks)} instance(s)")
                hard_flush(f"after-tag:{tag}")
        finally:
            if handle is not None:
                handle.release()

    return layers, log


# ==============================================================================
# 6. STAGE B — GENERATIVE FILL (SD 1.5 Inpainting, tiled)
# ==============================================================================


class InpainterHandle:
    """
    Short-lived owner of the StableDiffusionInpaintPipeline.

    VRAM strategy on 8 GB:
      - fp16 weights
      - enable_model_cpu_offload(): only the currently-executing submodule
        (text encoder / UNet / VAE) sits on the GPU; peak stays ~3.5 GB
      - attention + VAE slicing for the tail-end spikes
      - safety checker disabled (it's another 1.2 GB of CLIP for no benefit here)
    """

    def __init__(self):
        from diffusers import StableDiffusionInpaintPipeline

        last_err = None
        self.pipe = None
        for model_id in Config.INPAINT_MODEL_CANDIDATES:
            try:
                print(f"  Loading inpaint model: {model_id}")
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    model_id,
                    torch_dtype=Config.DTYPE,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                self.model_id = model_id
                break
            except Exception as e:
                last_err = e
                print(f"  [warn] {model_id} unavailable: {type(e).__name__}")
        if self.pipe is None:
            raise RuntimeError(
                "No SD inpainting checkpoint could be loaded. Tried: "
                f"{list(Config.INPAINT_MODEL_CANDIDATES)}. Last error: {last_err}"
            )

        self.pipe.set_progress_bar_config(disable=True)
        if torch.cuda.is_available():
            # NOTE: cpu_offload REPLACES .to('cuda') — never call both.
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing("max")
            self.pipe.enable_vae_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers optional; slicing already covers us
        print(vram_report("inpainter:ready"))

    @torch.inference_mode()
    def inpaint_tile(self, tile_rgb: np.ndarray, tile_mask: np.ndarray,
                     prompt: str, negative: str, seed: int) -> np.ndarray:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=Image.fromarray(tile_rgb),
            mask_image=Image.fromarray(tile_mask),
            num_inference_steps=Config.INPAINT_STEPS,
            guidance_scale=Config.GUIDANCE_SCALE,
            height=Config.TILE,
            width=Config.TILE,
            generator=gen,
        ).images[0]
        return np.array(out.convert("RGB"))

    def release(self) -> None:
        try:
            for attr in ("unet", "vae", "text_encoder", "safety_checker"):
                mod = getattr(self.pipe, attr, None)
                if isinstance(mod, torch.nn.Module):
                    mod.to("cpu")
        except Exception:
            pass
        self.pipe = None
        del self.pipe
        hard_flush("inpainter:released")


def _feather_weights(size: int, overlap: int) -> np.ndarray:
    """1-D cosine ramp -> 2-D separable blend kernel for seamless tile joins."""
    w = np.ones(size, dtype=np.float32)
    if overlap > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap, dtype=np.float32))
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return np.outer(w, w)


def run_generative_fill(base_rgb: np.ndarray, hole_mask: np.ndarray,
                        prompt: str, negative: str, seed: int,
                        progress_cb=None) -> Tuple[np.ndarray, List[str]]:
    """
    STAGE B entry point — REQUIREMENT #3.

    Reconstructs whatever the extracted parts were covering, using a sequential
    overlapping-tile sweep at SD1.5's native 512px. Sequential (not batched) is
    intentional twice over: it caps VRAM at one tile, and each tile sees the
    already-filled output of its predecessors, which keeps the fill coherent
    across a large canvas.

    Only pixels inside `hole_mask` are ever written back — original pixels are
    bit-for-bit preserved.
    """
    log: List[str] = []
    H, W = base_rgb.shape[:2]

    if (hole_mask > 0).sum() == 0:
        return base_rgb.copy(), ["  ↷ Generative fill skipped: nothing to fill"]

    canvas = base_rgb.astype(np.float32).copy()
    handle: Optional[InpainterHandle] = None

    with exclusive_gpu("STAGE-B/inpainting"):
        try:
            handle = InpainterHandle()
            log.append(f"  Model: {handle.model_id}")

            T, OV = Config.TILE, Config.TILE_OVERLAP
            stride = T - OV
            ys = list(range(0, max(H - T, 0) + 1, stride)) or [0]
            xs = list(range(0, max(W - T, 0) + 1, stride)) or [0]
            if ys[-1] + T < H:
                ys.append(H - T)
            if xs[-1] + T < W:
                xs.append(W - T)
            ys = [max(0, y) for y in ys]
            xs = [max(0, x) for x in xs]

            weights = _feather_weights(T, OV)
            jobs = [(y, x) for y in ys for x in xs]
            done = 0

            for (y, x) in jobs:
                done += 1
                y1, x1 = min(y + T, H), min(x + T, W)
                sub_mask = hole_mask[y:y1, x:x1]
                if (sub_mask > 0).sum() == 0:
                    continue  # nothing to reconstruct here — skip the GPU entirely

                if progress_cb is not None:
                    progress_cb(0.5 + 0.4 * done / len(jobs),
                                desc=f"Generative fill tile {done}/{len(jobs)}")

                # Pad edge tiles up to exactly TILE x TILE (SD needs /8 sizes)
                th, tw = y1 - y, x1 - x
                tile_img = np.zeros((T, T, 3), np.uint8)
                tile_msk = np.zeros((T, T), np.uint8)
                tile_img[:th, :tw] = canvas[y:y1, x:x1].astype(np.uint8)
                tile_msk[:th, :tw] = sub_mask
                if th < T or tw < T:  # edge replicate so SD doesn't invent a border
                    tile_img = cv2.copyMakeBorder(
                        tile_img[:th, :tw], 0, T - th, 0, T - tw, cv2.BORDER_REPLICATE)

                filled = handle.inpaint_tile(tile_img, tile_msk, prompt, negative,
                                             seed + done)

                # Weighted accumulation, restricted to hole pixels
                wgt = weights[:th, :tw][..., None]
                hole = (sub_mask > 0).astype(np.float32)[..., None]
                blend = wgt * hole
                canvas[y:y1, x:x1] = (canvas[y:y1, x:x1] * (1.0 - blend)
                                      + filled[:th, :tw].astype(np.float32) * blend)

                # Per-tile flush: this is what keeps 8 GB stable over 30+ tiles
                del filled
                hard_flush(f"tile-{done}")

            log.append(f"  Tiles processed: {done}")
        finally:
            if handle is not None:
                handle.release()

    result = np.clip(canvas, 0, 255).astype(np.uint8)

    # Hard guarantee: untouched pixels come from the original, no drift.
    keep = (hole_mask == 0)
    result[keep] = base_rgb[keep]

    # Soft 3px seam blend at the hole boundary
    soft = cv2.GaussianBlur(hole_mask, (7, 7), 0).astype(np.float32)[..., None] / 255.0
    result = np.clip(base_rgb.astype(np.float32) * (1 - soft)
                     + result.astype(np.float32) * soft, 0, 255).astype(np.uint8)
    return result, log


# ==============================================================================
# 7. EXPORTERS — ZIP + PSD
# ==============================================================================


def build_background_plate(base_rgb: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    """
    Fallback background when generative fill is OFF: the original image with
    the extracted regions knocked out. Cheap edge-aware fill (Telea) is applied
    so the plate isn't pure black — much friendlier for AE previews.
    """
    plate = base_rgb.copy()
    if (hole_mask > 0).any():
        plate = cv2.inpaint(plate, (hole_mask > 0).astype(np.uint8),
                            3, cv2.INPAINT_TELEA)
    return plate


def export_zip(out_dir: str, base_rgb: np.ndarray, layers: List[LayerData],
               background_rgb: np.ndarray, manifest: dict) -> str:
    """
    ZIP of individual transparent PNGs.

    Each layer is written TWICE on purpose:
      layers/<name>.png       -> full-canvas RGBA (drop into AE, zero offset)
      cropped/<name>.png      -> tight crop (for C4D texture atlases / sprite work)
    `manifest.json` carries offsets so the crops can be re-registered.
    """
    png_dir = os.path.join(out_dir, "layers")
    crop_dir = os.path.join(out_dir, "cropped")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    Image.fromarray(background_rgb).save(os.path.join(out_dir, "background.png"))
    Image.fromarray(base_rgb).save(os.path.join(out_dir, "original.png"))

    for lyr in layers:
        rgba = extract_rgba(base_rgb, lyr.mask)
        Image.fromarray(rgba).save(os.path.join(png_dir, f"{lyr.name}.png"))
        x0, y0, x1, y1 = lyr.bbox
        Image.fromarray(rgba[y0:y1, x0:x1]).save(os.path.join(crop_dir, f"{lyr.name}.png"))

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    zip_path = os.path.join(out_dir, "segrig_layers.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if fn.endswith(".zip") or fn.endswith(".psd"):
                    continue
                full = os.path.join(root, fn)
                zf.write(full, os.path.relpath(full, out_dir))
    return zip_path


def export_psd(out_dir: str, background_rgb: np.ndarray,
               layers: List[LayerData], base_rgb: np.ndarray) -> Optional[str]:
    """
    PSD assembly — REQUIREMENT #4.

    AE "Retain Layer Sizes" contract:
      * Every layer is written at its OWN tight bounding box (top/left/bottom/
        right), not stretched to the canvas. AE then creates a comp layer whose
        anchor point sits at that box — so your rig pivots land where the art
        actually is.
      * A full-canvas `background` layer at the bottom fixes comp dimensions.
      * Layers are ordered back-to-front by pixel area (largest = furthest
        back), a sane default for character rigs. Override via LAYER_ORDER_KEY.

    Returns path, or None if pytoshop isn't installed (ZIP still works).
    """
    try:
        import pytoshop
        from pytoshop import enums
        from pytoshop.user import nested_layers
    except ImportError:
        print("[warn] pytoshop not installed — skipping PSD export")
        return None

    H, W = background_rgb.shape[:2]
    psd_path = os.path.join(out_dir, "segrig_rig.psd")

    # Back-to-front: big shapes behind, small details in front.
    ordered = sorted(layers, key=lambda l: -l.area)

    def make_layer(name: str, rgba: np.ndarray, top: int, left: int):
        h, w = rgba.shape[:2]
        return nested_layers.Image(
            name=name,
            visible=True,
            opacity=255,
            top=top, left=left, bottom=top + h, right=left + w,
            channels={
                0: np.ascontiguousarray(rgba[:, :, 0]),   # R
                1: np.ascontiguousarray(rgba[:, :, 1]),   # G
                2: np.ascontiguousarray(rgba[:, :, 2]),   # B
                -1: np.ascontiguousarray(rgba[:, :, 3]),  # A
            },
            color_mode=enums.ColorMode.rgb,
            blend_mode=enums.BlendMode.normal,
        )

    psd_layers = []

    # --- foreground parts, each cropped tight ---------------------------------
    for lyr in ordered:
        x0, y0, x1, y1 = lyr.bbox
        rgba = extract_rgba(base_rgb, lyr.mask)[y0:y1, x0:x1]
        psd_layers.append(make_layer(lyr.name, rgba, top=y0, left=x0))

    # --- background, full canvas, opaque --------------------------------------
    bg_rgba = np.dstack([background_rgb,
                         np.full((H, W), 255, np.uint8)]).astype(np.uint8)
    psd_layers.append(make_layer("background", bg_rgba, top=0, left=0))

    # pytoshop's list is TOP-first. Our list is currently back-to-front, so the
    # background is already last == bottom. Flip this flag if your build differs.
    PSD_LAYERS_TOP_FIRST = True
    if not PSD_LAYERS_TOP_FIRST:
        psd_layers = psd_layers[::-1]
    else:
        # foreground (small parts) should be topmost -> reverse the part block
        parts, bg = psd_layers[:-1], psd_layers[-1]
        psd_layers = parts[::-1] + [bg]

    psd = nested_layers.nested_layers_to_psd(
        psd_layers,
        color_mode=enums.ColorMode.rgb,
        size=(W, H),
        compression=enums.Compression.rle,
    )
    with open(psd_path, "wb") as fd:
        psd.write(fd)
    return psd_path


# ==============================================================================
# 8. ORCHESTRATOR  (UI-agnostic — import this in your AE/C4D tooling)
# ==============================================================================


class SegRigPipeline:
    """
    Headless-callable orchestrator.

    Example (no Gradio required):
        from segrig_studio import SegRigPipeline
        res = SegRigPipeline().run("char.png", "hair, face, arms, shirt",
                                   generative_fill=True)
        print(res.psd_path, res.zip_path)
    """

    def __init__(self, work_root: str = Config.WORK_ROOT):
        self.work_root = work_root

    def _session_dir(self) -> str:
        d = os.path.join(self.work_root, f"session_{int(time.time() * 1000)}")
        os.makedirs(d, exist_ok=True)
        return d

    def run(self, image, tags_raw: str,
            generative_fill: bool = True,
            dilate_px: int = Config.DILATE_PX_DEFAULT,
            prompt: str = Config.DEFAULT_POSITIVE,
            negative: str = Config.DEFAULT_NEGATIVE,
            seed: int = 1234,
            progress_cb=None) -> PipelineResult:

        log: List[str] = []
        t_start = time.time()

        # ---------- 0. Input validation & normalization ----------------------
        if isinstance(image, str):
            image = Image.open(image)
        base_rgb = clamp_resolution(to_rgb_array(image))
        H, W = base_rgb.shape[:2]
        log.append(f"Canvas: {W}x{H}")

        tags = parse_tags(tags_raw)
        if not tags:
            raise ValueError("No tags provided. Example: 'hair, face, arms, shirt'")
        log.append(f"Tags: {', '.join(tags)}")

        out_dir = self._session_dir()

        # ---------- STAGE A: segmentation ------------------------------------
        log.append("── STAGE A: Segmentation (LangSAM) ──")
        layers, seg_log = run_segmentation(base_rgb, tags, progress_cb)
        log.extend(seg_log)
        if not layers:
            raise ValueError(
                "No objects matched your tags. Try simpler, more literal nouns "
                "('hair', 'shirt') or lower Config.BOX_THRESHOLD."
            )
        log.append(f"Total layers: {len(layers)}")

        # ---------- Dilation: build the inpaint hole -------------------------
        union = np.zeros((H, W), np.uint8)
        for lyr in layers:
            union = np.maximum(union, lyr.mask)
        hole_mask = dilate_mask(union, dilate_px)
        log.append(f"Hole mask dilated by {dilate_px}px "
                   f"({(hole_mask > 0).mean() * 100:.1f}% of canvas)")

        # ---------- STAGE B: generative fill (optional) ----------------------
        if generative_fill:
            log.append("── STAGE B: Generative Fill (SD1.5 Inpaint) ──")
            background_rgb, fill_log = run_generative_fill(
                base_rgb, hole_mask, prompt, negative, seed, progress_cb)
            log.extend(fill_log)
        else:
            log.append("── STAGE B: SKIPPED (checkbox off) ──")
            if progress_cb is not None:
                progress_cb(0.85, desc="Building background plate")
            background_rgb = build_background_plate(base_rgb, hole_mask)

        # ---------- Export ----------------------------------------------------
        if progress_cb is not None:
            progress_cb(0.92, desc="Exporting ZIP / PSD")
        manifest = {
            "canvas": {"width": W, "height": H},
            "generative_fill": bool(generative_fill),
            "dilation_px": int(dilate_px),
            "tags": tags,
            "layers": [l.to_manifest() for l in layers],
            "note": "Import PSD into After Effects with 'Retain Layer Sizes'.",
        }
        zip_path = export_zip(out_dir, base_rgb, layers, background_rgb, manifest)
        psd_path = export_psd(out_dir, background_rgb, layers, base_rgb)
        log.append(f"ZIP: {zip_path}")
        log.append(f"PSD: {psd_path or 'unavailable (pytoshop missing)'}")
        log.append(f"Done in {time.time() - t_start:.1f}s")

        hard_flush("pipeline:complete")
        return PipelineResult(base_rgb=base_rgb, layers=layers,
                              background_rgb=background_rgb,
                              zip_path=zip_path, psd_path=psd_path,
                              log="\n".join(log))


# ==============================================================================
# 9. GRADIO UI  (thin adapter over SegRigPipeline)
# ==============================================================================

import gradio as gr  # noqa: E402  (kept late so headless import stays light)
# ==============================================================================
# PATCH: กันบั๊ก gradio_client "argument of type 'bool' is not iterable"
# ==============================================================================

import gradio_client.utils as _gcu

_orig_json_to_type = _gcu._json_schema_to_python_type
_orig_get_type = _gcu.get_type
# ==============================================================================
# PATCH 2: กัน gr.Progress ระเบิดตอนถูกเช็ค truthiness
# ==============================================================================
def _safe_progress_len(self):
    try:
        if not self.iterables:
            return 0
        return self.iterables[-1].length or 0
    except Exception:
        return 0


try:
    gr.Progress.__len__ = _safe_progress_len
    gr.Progress.__bool__ = lambda self: True
except Exception:
    pass
# ==============================================================================
# ==============================================================================
# PATCH 3: ซ่อม pytoshop เขียน PSD ไม่ได้ (NameError: packbits)
# ==============================================================================
import numpy as _np

_PSD_FORCE_RAW = {"on": False}

try:
    from pytoshop import codecs as _pt_codecs
    from pytoshop import layers as _pt_layers
    from pytoshop import enums as _pt_enums

    class _PackBitsShim:
        """PackBits encoder แทนตัว Cython ที่ build ไม่สำเร็จบน Windows"""

        @staticmethod
        def encode(data):
            if isinstance(data, _np.ndarray):
                buf = data.astype(_np.uint8, copy=False).tobytes()
            elif isinstance(data, (bytes, bytearray, memoryview)):
                buf = bytes(data)
            else:
                buf = bytes(bytearray(data))

            n = len(buf)
            if n == 0:
                return b""

            out = bytearray()

            # fast path: ทั้งแถวเป็นค่าเดียว (เจอบ่อยมากในภาพ mask)
            first = buf[0]
            if buf.count(first) == n:
                i = 0
                while i < n:
                    run = min(127, n - i)
                    out.append(0 if run == 1 else 257 - run)
                    out.append(first)
                    i += run
                return bytes(out)

            i = 0
            while i < n:
                j = i + 1
                while j < n and buf[j] == buf[i] and (j - i) < 127:
                    j += 1
                run = j - i
                if run >= 2:
                    out.append(257 - run)
                    out.append(buf[i])
                    i = j
                    continue

                start = i
                j = i
                while j < n and (j - start) < 128:
                    if j + 2 < n and buf[j] == buf[j + 1] == buf[j + 2]:
                        break
                    j += 1
                if j == start:
                    j = start + 1
                out.append(j - start - 1)
                out += buf[start:j]
                i = j
            return bytes(out)

        @staticmethod
        def decode(data):
            buf = bytes(data)
            out = bytearray()
            i, n = 0, len(buf)
            while i < n:
                h = buf[i]
                i += 1
                if h == 128:
                    continue
                if h < 128:
                    out += buf[i:i + h + 1]
                    i += h + 1
                else:
                    out += bytes([buf[i]]) * (257 - h)
                    i += 1
            return bytes(out)

    if getattr(_pt_codecs, "packbits", None) is None:
        _pt_codecs.packbits = _PackBitsShim

    # แผนสำรอง: บังคับเขียนแบบ raw (ไม่บีบอัด) ถ้า RLE ยังพัง
    _orig_cid_init = _pt_layers.ChannelImageData.__init__

    def _cid_init(self, *args, **kwargs):
        _orig_cid_init(self, *args, **kwargs)
        if _PSD_FORCE_RAW["on"]:
            try:
                self.compression = _pt_enums.Compression.raw
            except Exception:
                pass

    _pt_layers.ChannelImageData.__init__ = _cid_init

    try:
        from pytoshop import image_data as _pt_image_data

        _orig_id_init = _pt_image_data.ImageData.__init__

        def _id_init(self, *args, **kwargs):
            _orig_id_init(self, *args, **kwargs)
            if _PSD_FORCE_RAW["on"]:
                try:
                    self.compression = _pt_enums.Compression.raw
                except Exception:
                    pass

        _pt_image_data.ImageData.__init__ = _id_init
    except Exception:
        pass

except Exception as _e:
    print("[patch] pytoshop patch skipped:", _e)
# ==============================================================================

def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


def _safe_json_to_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_to_type(schema, defs)


_gcu.get_type = _safe_get_type
_gcu._json_schema_to_python_type = _safe_json_to_type
# ==============================================================================
PIPELINE = SegRigPipeline()


def ui_process(image, tags_raw, do_fill, dilate_px, prompt, negative, seed,
               progress=gr.Progress()):
    """Gradio callback. Returns (AnnotatedImage, background, zip, psd, log)."""
    if image is None:
        raise gr.Error("Please upload an image first.")
    try:
        progress(0.02, desc="Preparing…")
        res = PIPELINE.run(
            image=image,
            tags_raw=tags_raw,
            generative_fill=bool(do_fill),
            dilate_px=int(dilate_px),
            prompt=prompt.strip() or Config.DEFAULT_POSITIVE,
            negative=negative.strip() or Config.DEFAULT_NEGATIVE,
            seed=int(seed),
            progress_cb=progress,
        )
    except torch.cuda.OutOfMemoryError:
        hard_flush("OOM-recovery")
        raise gr.Error(
            "CUDA OOM. Try: lower Config.MAX_SIDE to 1280, reduce tags per run, "
            "or uncheck Generative Fill."
        )
    except Exception as e:
        hard_flush("error-recovery")
        raise gr.Error(f"{type(e).__name__}: {e}")

    # gr.AnnotatedImage wants (base_image, [(bool_mask, label), ...])
    annotations = [((l.mask > 0), l.name) for l in res.layers]
    progress(1.0, desc="Complete")
    return (res.base_rgb, annotations), res.background_rgb, \
        res.zip_path, res.psd_path, res.log


CSS = """
.gradio-container {max-width: 1500px !important;}
#log textarea {font-family: ui-monospace, Consolas, monospace; font-size: 12px;}
"""

with gr.Blocks(title="SegRig Studio", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        "## 🎬 SegRig Studio — 2D Character Rigging Prep\n"
        "LangSAM segmentation → optional SD1.5 generative fill → "
        "**PSD** (AE *Retain Layer Sizes*) + **ZIP** of transparent PNGs.  \n"
        "*Models are loaded and unloaded one at a time — safe for 8 GB VRAM.*"
    )

    with gr.Row():
        # ------------------------ LEFT: controls ------------------------------
        with gr.Column(scale=1):
            in_image = gr.Image(label="Base Image", type="pil", height=340)
            in_tags = gr.Textbox(
                label="Tags (comma-separated)",
                value="hair, face, arms, shirt",
                placeholder="hair, face, arms, shirt, legs, shoes",
                info="One noun per part. Multiple matches auto-number: hair_1, hair_2…",
            )
            in_fill = gr.Checkbox(
                label="Reconstruct Background (Generative Fill)",
                value=False,
                info="Off = instant. On = SD1.5 inpainting, ~1–3 min on a 2060 Super.",
            )
            with gr.Accordion("Advanced", open=False):
                in_dilate = gr.Slider(0, 30, value=Config.DILATE_PX_DEFAULT, step=1,
                                      label="Mask Dilation (px)",
                                      info="10–15 px removes halos before inpainting.")
                in_prompt = gr.Textbox(label="Fill Prompt", value=Config.DEFAULT_POSITIVE)
                in_neg = gr.Textbox(label="Negative Prompt", value=Config.DEFAULT_NEGATIVE)
                in_seed = gr.Number(label="Seed", value=1234, precision=0)
            btn = gr.Button("🚀 Process", variant="primary", size="lg")

        # ------------------------ RIGHT: results ------------------------------
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Mask Preview"):
                    out_annot = gr.AnnotatedImage(label="Detected Layers", height=520)
                with gr.Tab("Background Plate"):
                    out_bg = gr.Image(label="Reconstructed / Knocked-out Background",
                                      height=520)
                with gr.Tab("Log"):
                    out_log = gr.Textbox(label="Pipeline Log", lines=22,
                                         elem_id="log", show_copy_button=True)
            with gr.Row():
                out_zip = gr.File(label="⬇ Layers (.zip)")
                out_psd = gr.File(label="⬇ Rig (.psd)")

    btn.click(
        fn=ui_process,
        inputs=[in_image, in_tags, in_fill, in_dilate, in_prompt, in_neg, in_seed],
        outputs=[out_annot, out_bg, out_zip, out_psd, out_log],
        concurrency_limit=1,   # HARD single-flight: never two models at once
    )

    gr.Markdown(
        "**AE import:** File → Import → File → select `.psd` → *Composition – "
        "Retain Layer Sizes*.  \n"
        "**C4D:** use `cropped/*.png` + `manifest.json` offsets for planar setup."
    )

# ==============================================================================
# PATCH 3b: export_psd แบบมีแผนสำรอง (RLE → raw)
# ==============================================================================
_ORIG_EXPORT_PSD = export_psd


def export_psd(*args, **kwargs):
    try:
        return _ORIG_EXPORT_PSD(*args, **kwargs)
    except Exception as e:
        print(f"[psd] RLE failed ({type(e).__name__}: {e}) -> retry raw mode")
        _PSD_FORCE_RAW["on"] = True
        try:
            return _ORIG_EXPORT_PSD(*args, **kwargs)
        finally:
            _PSD_FORCE_RAW["on"] = False
# ==============================================================================

if __name__ == "__main__":
    print(vram_report("boot"))
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        show_api=False,          # <<< เพิ่มบรรทัดนี้ — ตัดต้นเหตุโดยตรง
    )
