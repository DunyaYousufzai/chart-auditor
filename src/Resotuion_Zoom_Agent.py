from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageFilter, ImageOps, ImageStat


@dataclass
class ImageInspection:
    width: int
    height: int
    max_side: int
    min_side: int
    aspect_ratio: float
    mean_luma: float
    contrast_std: float
    is_small: bool
    is_low_contrast: bool


@dataclass
class ZoomResult:
    image_path: str
    applied: bool
    mode: str  # "none", "upscale", "crop_upscale"
    scale_factor: float
    reason: str
    inspection: Optional[ImageInspection] = None


class ResolutionZoomAgent:
    """Inspecciona resolución y nitidez de imágenes de gráficos y genera versiones ampliadas o recortadas."""

    def __init__(
        self,
        min_side_px: int = 1000,
        min_contrast_std: float = 22.0,
        max_zoom_scale: float = 3.0,
        cache_dir: str = "/tmp/chart_auditor_zoom",
    ):
        self.min_side_px = min_side_px
        self.min_contrast_std = min_contrast_std
        self.max_zoom_scale = max_zoom_scale
        self.cache_dir = Path(cache_dir)

    def inspect(self, image_path: str) -> ImageInspection:
        with Image.open(image_path) as im:
            gray = im.convert("L")
            w, h = gray.size
            stat = ImageStat.Stat(gray)
            mean_luma = float(stat.mean[0])
            contrast_std = float(stat.stddev[0])

        max_side = max(w, h)
        min_side = min(w, h)
        aspect_ratio = max_side / max(min_side, 1)

        return ImageInspection(
            width=w,
            height=h,
            max_side=max_side,
            min_side=min_side,
            aspect_ratio=round(aspect_ratio, 3),
            mean_luma=round(mean_luma, 2),
            contrast_std=round(contrast_std, 2),
            is_small=max_side < self.min_side_px,
            is_low_contrast=contrast_std < self.min_contrast_std,
        )

    def should_zoom(self, image_path: str, high_spread: bool = False) -> Tuple[bool, str, ImageInspection]:
        info = self.inspect(image_path)
        reasons = []
        if info.is_small:
            reasons.append("low_res")
        if info.is_low_contrast:
            reasons.append("low_contrast")
        if high_spread:
            reasons.append("high_spread")

        return (len(reasons) > 0), "+".join(reasons) if reasons else "ok", info

    def zoom_image(
        self,
        image_path: str,
        target_scale: Optional[float] = None,
        reason: str = "upscale",
        tag: str = "zoom",
    ) -> ZoomResult:
        src = Path(image_path)
        info = self.inspect(image_path)

        if target_scale is None:
            if info.max_side > 0 and info.max_side < self.min_side_px:
                scale = min(self.max_zoom_scale, self.min_side_px / info.max_side)
            else:
                scale = 1.5
        else:
            scale = max(1.0, min(self.max_zoom_scale, float(target_scale)))

        if scale <= 1.01 and not info.is_low_contrast:
            return ZoomResult(
                image_path=str(src),
                applied=False,
                mode="none",
                scale_factor=1.0,
                reason="already_sharp_and_large",
                inspection=info,
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.cache_dir / f"{src.stem}_{tag}{src.suffix if src.suffix else '.png'}"

        with Image.open(src) as im:
            rgb = im.convert("RGB")
            new_w = max(1, int(round(rgb.width * scale)))
            new_h = max(1, int(round(rgb.height * scale)))

            resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if info.is_low_contrast or reason.find("low_contrast") != -1:
                resized = ImageOps.autocontrast(resized, cutoff=1)

            sharpened = resized.filter(
                ImageFilter.UnsharpMask(radius=1.2, percent=125, threshold=3)
            )
            sharpened.save(out_path)

        return ZoomResult(
            image_path=str(out_path),
            applied=True,
            mode="upscale",
            scale_factor=round(scale, 3),
            reason=reason,
            inspection=info,
        )

    def crop_region(
        self,
        image_path: str,
        box_norm: Tuple[float, float, float, float],
        min_output_side: int = 900,
        tag: str = "crop",
    ) -> ZoomResult:
        """Recorta una región en coordenadas normalizadas (x0, y0, x1, y1) en [0, 1] y la amplía."""
        src = Path(image_path)
        info = self.inspect(image_path)
        x0, y0, x1, y1 = box_norm

        x0_px = max(0, min(info.width - 1, int(round(x0 * info.width))))
        y0_px = max(0, min(info.height - 1, int(round(y0 * info.height))))
        x1_px = max(x0_px + 1, min(info.width, int(round(x1 * info.width))))
        y1_px = max(y0_px + 1, min(info.height, int(round(y1 * info.height))))

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.cache_dir / f"{src.stem}_{tag}{src.suffix if src.suffix else '.png'}"

        with Image.open(src) as im:
            cropped = im.convert("RGB").crop((x0_px, y0_px, x1_px, y1_px))
            cw, ch = cropped.size
            c_max = max(cw, ch)
            scale = max(1.0, min(self.max_zoom_scale, min_output_side / max(c_max, 1)))
            if scale > 1.01:
                cropped = cropped.resize(
                    (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
                    Image.Resampling.LANCZOS,
                )
            cropped = ImageOps.autocontrast(cropped, cutoff=1)
            cropped = cropped.filter(
                ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3)
            )
            cropped.save(out_path)

        return ZoomResult(
            image_path=str(out_path),
            applied=True,
            mode="crop_upscale",
            scale_factor=round(scale, 3),
            reason=f"crop({x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f})",
            inspection=info,
        )