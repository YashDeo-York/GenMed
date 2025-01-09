import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom, sobel

def validate_shapes(image1, image2):
    """
    Validates and aligns the shapes of two images.
    If the shapes don't match, raises an error.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Shape mismatch: image1 has shape {image1.shape}, image2 has shape {image2.shape}")

def calculate_psnr(image1, image2, max_pixel_value=255.0):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) for 2D, 3D, or 4D images.
    Args:
        image1 (ndarray): First image or stack of images.
        image2 (ndarray): Second image or stack of images.
        max_pixel_value (float): Maximum possible pixel value of the images.

    Returns:
        float: PSNR value (average if multiple images).
    """
    validate_shapes(image1, image2)
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def calculate_ssim(image1, image2):
    """
    Calculate the Structural Similarity Index (SSIM) for 2D, 3D, or 4D images.
    Args:
        image1 (ndarray): First image or stack of images.
        image2 (ndarray): Second image or stack of images.

    Returns:
        float: SSIM value (average if multiple images).
    """
    validate_shapes(image1, image2)
    if image1.ndim == 2:
        return ssim(image1, image2)
    elif image1.ndim >= 3:
        scores = []
        for i in range(image1.shape[0]):
            scores.append(ssim(image1[i], image2[i]))
        return np.mean(scores)


def calculate_ms_ssim(image1, image2, weights=None):
    """
    Calculate the Multi-Scale Structural Similarity Index (MS-SSIM) for 2D, 3D, or 4D images.
    Args:
        image1 (ndarray): First image or stack of images.
        image2 (ndarray): Second image or stack of images.
        weights (list): List of weights for different scales.

    Returns:
        float: MS-SSIM value (average if multiple images).
    """
    validate_shapes(image1, image2)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def calculate_single_ms_ssim(img1, img2):
        msssim = []
        for weight in weights:
            score, _ = ssim(img1, img2, full=True)
            msssim.append(weight * score)
            # Downsample images by factor of 2
            img1 = zoom(img1, 0.5, order=1)
            img2 = zoom(img2, 0.5, order=1)
        return np.sum(msssim)

    if image1.ndim == 2:
        return calculate_single_ms_ssim(image1, image2)
    elif image1.ndim >= 3:
        scores = []
        for i in range(image1.shape[0]):
            scores.append(calculate_single_ms_ssim(image1[i], image2[i]))
        return np.mean(scores)


def calculate_4gr_ssim(image1, image2):
    """
    Calculate the 4th-Order Gradient Structural Similarity Index (4GR-SSIM) for 2D, 3D, or 4D images.
    Args:
        image1 (ndarray): First image or stack of images.
        image2 (ndarray): Second image or stack of images.

    Returns:
        float: 4GR-SSIM value (average if multiple images).
    """
    validate_shapes(image1, image2)

    def calculate_single_4gr_ssim(img1, img2):
        # Compute gradients (4th-order Sobel approximation)
        grad1_x = sobel(img1, axis=0)
        grad1_y = sobel(img1, axis=1)
        grad2_x = sobel(img2, axis=0)
        grad2_y = sobel(img2, axis=1)

        grad1 = np.sqrt(grad1_x**2 + grad1_y**2)
        grad2 = np.sqrt(grad2_x**2 + grad2_y**2)

        return ssim(grad1, grad2)

    if image1.ndim == 2:
        return calculate_single_4gr_ssim(image1, image2)
    elif image1.ndim >= 3:
        scores = []
        for i in range(image1.shape[0]):
            scores.append(calculate_single_4gr_ssim(image1[i], image2[i]))
        return np.mean(scores)


def calculate_metrics_for_sets(set1, set2, metric_fn):
    """
    Calculate a metric for two sets of images (or stacks of images) with potentially different sizes.
    Args:
        set1 (ndarray): First set of images.
        set2 (ndarray): Second set of images.
        metric_fn (function): Metric function to apply (e.g., PSNR, SSIM).

    Returns:
        list: Metric values for paired images.
    """
    num_images = min(len(set1), len(set2))
    scores = []
    for i in range(num_images):
        scores.append(metric_fn(set1[i], set2[i]))
    return scores
