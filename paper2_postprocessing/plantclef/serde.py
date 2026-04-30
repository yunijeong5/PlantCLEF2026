"""Module for encoding and decoding data structures to and from raw bytes"""

from PIL import Image
import numpy as np
import zlib
import io


def deserialize_image(buffer: bytes | bytearray) -> Image.Image:
    """Decode the image from raw bytes using PIL."""
    buffer = io.BytesIO(bytes(buffer))
    return Image.open(buffer)


def serialize_image(image: Image.Image) -> bytes:
    """Encode the image as raw bytes using PIL."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def preprocess_tile(tile: Image.Image) -> Image.Image:
    """
    Preprocess a tile image by JPEG recompression to match model training distribution.
    
    Applies:
    - JPEG compression at quality=85
    - 4:2:2 chroma subsampling (subsampling=1)
    - Converts to RGB if necessary
    
    Args:
        tile: PIL Image object
    
    Returns:
        Preprocessed PIL Image
    """
    # Ensure image is in RGB mode for JPEG
    if tile.mode != "RGB":
        tile = tile.convert("RGB")
    
    # Compress to JPEG and reload
    buffer = io.BytesIO()
    tile.save(buffer, format="JPEG", quality=85, subsampling=1)
    buffer.seek(0)
    preprocessed_tile = Image.open(buffer)
    # Force load to avoid buffer closure issues
    preprocessed_tile.load()
    return preprocessed_tile


def deserialize_mask(buffer: bytes | bytearray, use_compression=True) -> np.ndarray:
    """Decode the numpy mask array from raw bytes using np.load()."""
    if use_compression:
        buffer = zlib.decompress(bytes(buffer))
    return np.load(io.BytesIO(buffer))


def serialize_mask(mask: np.ndarray, use_compression=True) -> bytes:
    """Encode the numpy mask array as raw bytes using np.save()."""
    fp = io.BytesIO()
    np.save(fp, mask)
    buffer = fp.getvalue()
    if use_compression:
        buffer = zlib.compress(buffer)
    return buffer
