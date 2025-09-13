from __future__ import annotations
import io
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

from .model import get_u2netp, get_u2net
from .downloader import ensure_default_weights, ensure_weights


def _to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    im = np.array(img).astype(np.float32) / 255.0
    im = (im - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im = im.transpose((2, 0, 1))
    return torch.from_numpy(im).unsqueeze(0)


def refine_mask(mask: Image.Image, feather: int = 8, erode: int = 0, dilate: int = 0,
                fg_thresh: float = 0.7, bg_thresh: float = 0.3, conservative: bool = False) -> Image.Image:
    """Refine a grayscale mask using morphology and feathering.
    feather: Gaussian blur radius (pixels) to soften edges
    erode/dilate: pixels for morphological operations (0 to disable)
    """
    m = np.array(mask.convert("L"))
    if conservative:
        # Map strong BG/FG zones to 0/255 and only feather the uncertain band
        m_f = m.astype(np.float32) / 255.0
        # Stretch uncertain band
        denom = max(1e-6, (fg_thresh - bg_thresh))
        m_f = (m_f - bg_thresh) / denom
        m_f = np.clip(m_f, 0.0, 1.0)
        m = (m_f * 255.0).astype(np.uint8)
    # Morphology kernel
    k = max(1, int(round(feather / 4)))
    kernel = np.ones((k, k), np.uint8)
    if erode > 0:
        m = cv2.erode(m, kernel, iterations=max(1, erode // k))
    if dilate > 0:
        m = cv2.dilate(m, kernel, iterations=max(1, dilate // k))
    # Feather edges
    f = max(0, int(feather))
    if f > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=f, sigmaY=f)
    m = np.clip(m, 0, 255).astype(np.uint8)
    return Image.fromarray(m, mode="L")


class U2NetBackgroundRemover:
    def __init__(self, device: Optional[str] = None, progress: Optional[Callable[[str, int, int], None]] = None,
                 model_preference: str = "auto"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # choose weights based on preference
        try:
            weights = ensure_weights(model_preference, progress=progress)
        except Exception:
            weights = ensure_default_weights(progress)
        # Pick architecture based on filename
        wname = Path(weights).name.lower()
        if "u2netp" in wname:
            self.model = get_u2netp()
            print(f"[ClearCut] Using model: U2NETP, weights: {weights}")
        else:
            self.model = get_u2net()
            print(f"[ClearCut] Using model: U2NET (full), weights: {weights}")
        state = torch.load(weights, map_location="cpu")
        # Some checkpoints save under 'model' key
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def remove_bg(self, img: Image.Image, target_size: int = 320, feather: int = 10, erode: int = 0, dilate: int = 0,
                  conservative: bool = True, fg_thresh: float = 0.7, bg_thresh: float = 0.3,
                  decontaminate_strength: float = 0.0, decontaminate_band: int = 6) -> Tuple[Image.Image, Image.Image]:
        """Return (original, cutout with transparency)."""
        orig = img.convert("RGB")
        w, h = orig.size
        # Letterbox to preserve aspect ratio: resize so max side=target_size, then pad to square
        ts = int(target_size)
        scale = ts / max(w, h)
        rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img_small = orig.resize((rw, rh), Image.LANCZOS)
        pad_w, pad_h = ts - rw, ts - rh
        pad_left, pad_top = pad_w // 2, pad_h // 2
        canvas = Image.new('RGB', (ts, ts), (0, 0, 0))
        canvas.paste(img_small, (pad_left, pad_top))

        x = _to_tensor(canvas).to(self.device)
        pred = self.model(x)  # [1,1,ts,ts]
        if pred.shape[2] != ts or pred.shape[3] != ts:
            pred = F.interpolate(pred, size=(ts, ts), mode='bilinear', align_corners=False)
        pred = pred[0, 0].cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        # Remove padding area and upscale mask back to original size
        pred_img = Image.fromarray((pred * 255).astype(np.uint8))
        crop = pred_img.crop((pad_left, pad_top, pad_left + rw, pad_top + rh))
        mask = crop.resize((w, h), Image.BILINEAR)
        # refine
        mask = refine_mask(mask, feather=feather, erode=erode, dilate=dilate,
                           fg_thresh=fg_thresh, bg_thresh=bg_thresh, conservative=conservative)
        # optional color decontamination near edges
        base = orig
        if decontaminate_strength > 0.0:
            base = decontaminate_colors(orig, mask, strength=decontaminate_strength, band_px=decontaminate_band)
        cutout = apply_mask(base, mask)
        return orig, cutout


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    if mask.mode != "L":
        mask = mask.convert("L")
    r, g, b, _ = image.split()
    return Image.merge("RGBA", (r, g, b, mask))


def pil_to_qpixmap(img: Image.Image):
    from PyQt6.QtGui import QPixmap
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return QPixmap.fromImageReader(buf)


def ensure_default_weights_public(progress=None):
    return ensure_default_weights(progress)


def _estimate_bg_hue_rgb(img: np.ndarray, band: int = 10) -> float:
    """Estimate dominant background hue from image border in HSV space (0..180 OpenCV hue)."""
    h, w, _ = img.shape
    b = max(1, min(band, min(h, w) // 10))
    border = np.zeros((h, w), dtype=bool)
    border[:b, :] = True; border[-b:, :] = True; border[:, :b] = True; border[:, -b:] = True
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0][border]
    sat = hsv[..., 1][border]
    # Only consider reasonably saturated pixels
    mask = sat > 30
    if not np.any(mask):
        return float(np.median(hue.astype(np.float32)))
    return float(np.median(hue[mask].astype(np.float32)))


def decontaminate_colors(image: Image.Image, mask: Image.Image, strength: float = 0.6, band_px: int = 6) -> Image.Image:
    """Reduce background color spill near mask edges.
    strength: 0..1 amount of saturation reduction when hue matches background hue
    band_px: width of edge band (pixels) to process
    """
    rgb = np.array(image.convert('RGB'))
    a = np.array(mask.convert('L'))
    # Build edge band from alpha gradient
    edge = cv2.Canny(a, 10, 40)
    if band_px > 0:
        k = max(1, band_px // 2)
        kernel = np.ones((k, k), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=2)
    band = edge > 0

    bg_h = _estimate_bg_hue_rgb(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Compute hue distance in circular space (0..180)
    dh = np.minimum(np.abs(H - bg_h), 180 - np.abs(H - bg_h))
    # Target: pixels in band with hue close to bg hue
    mask_band = band & (dh < 20)
    # Reduce saturation proportionally
    S[mask_band] = S[mask_band] * (1.0 - np.clip(strength, 0.0, 1.0))
    hsv2 = cv2.merge([H, S, V]).astype(np.uint8)
    out_rgb = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)
    return Image.fromarray(out_rgb)


def refine_with_grabcut(orig: Image.Image, init_mask: Image.Image,
                        fg_strokes: np.ndarray, bg_strokes: np.ndarray,
                        iterations: int = 5, feather: int = 6,
                        gentle: bool = False) -> Image.Image:
    """Run GrabCut refinement using initial mask and user strokes.
    fg_strokes, bg_strokes are uint8 arrays (H,W) in {0,1} indicating user marks.
    Returns refined binary mask as PIL.L
    """
    img = np.array(orig.convert('RGB'))
    h, w = img.shape[:2]
    init = np.array(init_mask.convert('L').resize((w, h), Image.BILINEAR))

    # Initialize grabcut mask categories
    gc_mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)  # probable FG
    gc_mask[init < 20] = cv2.GC_BGD
    gc_mask[init > 220] = cv2.GC_FGD

    # sanitize strokes to 2D uint8 aligned to (h,w)
    def _prep(m):
        if m is None:
            return None
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    fg = _prep(fg_strokes)
    bg = _prep(bg_strokes)
    if fg is not None:
        gc_mask[fg.astype(bool)] = (cv2.GC_PR_FGD if gentle else cv2.GC_FGD)
    if bg is not None:
        gc_mask[bg.astype(bool)] = (cv2.GC_PR_BGD if gentle else cv2.GC_BGD)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    iters = max(1, int(iterations))
    if gentle:
        iters = min(iters, 3)
    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)

    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if feather > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=feather, sigmaY=feather)
    return Image.fromarray(out, mode='L')
