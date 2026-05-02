"""
Multi-scale tiling (paper 1 pre-processing).

For scale S with tile_size T (default 518):
  1. Resize the image so its SHORT side equals S*T, preserving aspect ratio
     (Lanczos filter, matching ImageMagick's Lanczos used in the Rust code).
  2. Center-crop to exactly (S*T) × (S*T) pixels.
  3. Extract S² non-overlapping T×T tiles in row-major order (top-left first).

Scale 1 → one 518×518 tile (whole-image baseline).
Scales [6,5,4,3,2,1] → 36+25+16+9+4+1 = 91 tiles (paper 1 full multi-scale).
"""

from typing import List

from PIL import Image


def resize_and_center_crop(img: Image.Image, target_size: int) -> Image.Image:
    """Resize *img* so its short side = *target_size*, then center-crop to square."""
    w, h = img.size
    if h < w:  # landscape
        new_h = target_size
        new_w = round(target_size * w / h)
    else:       # portrait or square
        new_w = target_size
        new_h = round(target_size * h / w)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_size) // 2
    top  = (new_h - target_size) // 2
    return img.crop((left, top, left + target_size, top + target_size))


def extract_tiles(img: Image.Image, scale: int, tile_size: int = 518) -> List[Image.Image]:
    """
    Return the S² tiles (as PIL Images) for the given *scale*.

    Tiles are ordered row-major: tile (i, j) has offset
    (row=i*tile_size, col=j*tile_size) in the cropped square image.
    """
    target_size = scale * tile_size
    square = resize_and_center_crop(img, target_size)

    tiles = []
    for i in range(scale):
        for j in range(scale):
            left  = j * tile_size
            upper = i * tile_size
            tile  = square.crop((left, upper, left + tile_size, upper + tile_size))
            tiles.append(tile)
    return tiles


def total_tiles(scales: List[int]) -> int:
    return sum(s * s for s in scales)
