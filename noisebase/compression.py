"""
noisebase.compression
---------------------

Compressed data formats used by Noisebase
"""
import numpy as np

def decompress_RGBE(color, exposures):
    """Decompresses per-sample radiance from RGBE compressed data

    Args:
        color (ndarray, uint8, 4HWS): radiance data in RGBE representation
        [min_exposure, max_exposure]: exposure range for decompression

    Returns:
        color (ndarray, 3HWS): per-sample RGB radiance
    """
    exponents = (color.astype(np.float32)[3] + 1)/256
    #exposures = np.reshape(exposures, (1, 1, 1, 2))

    exponents = np.exp(exponents * (exposures[1] - exposures[0]) + exposures[0])
    color = color.astype(np.float32)[:3] / 255 * exponents[np.newaxis]
    return color

def compress_RGBE(color):
    """Computes RGBE compressed representation of radiance data

    Args:
        color (ndarray, 3HWS): per-sample RGB radiance

    Returns:
        color (ndarray, uint8, 4HWS): radiance data in RGBE representation
        [min_exposure, max_exposure]: exposure range for decompression
    """
    log_radiance = np.log(color[np.where(color > 0)])

    if log_radiance.size == 0: # Handle black frames
        return np.zeros((4, color.shape[1], color.shape[2], color.shape[3]), dtype=np.uint8), [0, 0]
    
    # Calculate exposure
    min_exp = np.min(log_radiance)
    max_exp = np.max(log_radiance)

    # Get exponent from brightest channel
    brightest_channel = np.max(color, axis = 0)
    exponent = np.ones_like(brightest_channel) * -np.inf
    np.log(brightest_channel, out=exponent, where=brightest_channel > 0)

    # Quantise exponent with ceiling function
    e_channel = np.minimum((exponent - min_exp) / (max_exp - min_exp) * 256, 255).astype(np.uint8)[np.newaxis]
    # Actually encoded exponent
    exponent = np.exp(((e_channel.astype(np.float32) + 1)/256) * (max_exp - min_exp) + min_exp)

    # Quantise colour channels
    rgb_float = (color / exponent) * 255
    rgb_channels = (rgb_float).astype(np.uint8)
    # Add dither (exponents were quantised with ceiling so this doesn't go over 255)
    rgb_channels += ((rgb_float - rgb_channels) > np.random.random(rgb_channels.shape))

    return np.concatenate([rgb_channels, e_channel]), [min_exp, max_exp]