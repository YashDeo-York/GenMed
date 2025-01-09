import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_kernels
from lpips import LPIPS  # Requires pip install lpips
from sklearn.metrics import pairwise_distances


# Helper Functions
def validate_shapes(image1, image2):
    """
    Validates and aligns the shapes of two images.
    If the shapes don't match, raises an error.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Shape mismatch: image1 has shape {image1.shape}, image2 has shape {image2.shape}")


def calculate_metrics_for_sets(set1, set2, metric_fn):
    """
    Calculate a metric for two sets of images (or stacks of images) with potentially different sizes.
    Args:
        set1 (ndarray): First set of images.
        set2 (ndarray): Second set of images.
        metric_fn (function): Metric function to apply (e.g., DreamSim, LPIPS).

    Returns:
        list: Metric values for paired images.
    """
    num_images = min(len(set1), len(set2))
    scores = []
    for i in range(num_images):
        scores.append(metric_fn(set1[i], set2[i]))
    return scores


# Metrics Implementation
def calculate_dreamsim(image1, image2):
    """
    Calculate DreamSim metric (measures perceptual similarity using cosine similarity).
    Args:
        image1 (ndarray): First image or stack of images.
        image2 (ndarray): Second image or stack of images.

    Returns:
        float: DreamSim score (average if multiple images).
    """
    validate_shapes(image1, image2)
    flattened1 = image1.flatten()
    flattened2 = image2.flatten()
    cosine_similarity = np.dot(flattened1, flattened2) / (
        np.linalg.norm(flattened1) * np.linalg.norm(flattened2)
    )
    return cosine_similarity


def calculate_fls(image1, image2):
    """
    Calculate Feature Loss Score (FLS), a perceptual feature loss.
    Args:
        image1 (ndarray): First image.
        image2 (ndarray): Second image.

    Returns:
        float: FLS value.
    """
    validate_shapes(image1, image2)
    return np.mean(np.abs(image1 - image2))


def calculate_realism_score(real_images, generated_images, kernel='rbf'):
    """
    Calculate Realism Score using kernel similarity between real and generated images.
    Args:
        real_images (ndarray): Stack of real images.
        generated_images (ndarray): Stack of generated images.
        kernel (str): Kernel type (default: 'rbf').

    Returns:
        float: Realism Score.
    """
    distances = pairwise_kernels(real_images.reshape(len(real_images), -1),
                                 generated_images.reshape(len(generated_images), -1),
                                 metric=kernel)
    return np.mean(distances)


def calculate_lpips(image1, image2, model="vgg"):
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS).
    Args:
        image1 (ndarray): First image.
        image2 (ndarray): Second image.
        model (str): Feature model to use ('vgg', 'alex', etc.).

    Returns:
        float: LPIPS value.
    """
    validate_shapes(image1, image2)
    lpips_model = LPIPS(net=model).to("cpu")  # Load LPIPS model
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)
    score = lpips_model(image1, image2).item()
    return score


def calculate_kid(real_features, generated_features):
    """
    Calculate Kernel Inception Distance (KID).
    Args:
        real_features (ndarray): Feature vectors from real images.
        generated_features (ndarray): Feature vectors from generated images.

    Returns:
        float: KID score.
    """
    real_features = real_features.reshape(len(real_features), -1)
    generated_features = generated_features.reshape(len(generated_features), -1)
    mmd2 = np.mean(pairwise_kernels(real_features, real_features)) \
        + np.mean(pairwise_kernels(generated_features, generated_features)) \
        - 2 * np.mean(pairwise_kernels(real_features, generated_features))
    return mmd2


def calculate_asw(real_images, generated_images):
    """
    Calculate Average Surface Width (ASW) for vessel-like structures.
    Args:
        real_images (ndarray): Stack of real images.
        generated_images (ndarray): Stack of generated images.

    Returns:
        float: ASW score (average of pairwise distances).
    """
    distances = []
    for i in range(len(real_images)):
        dist = cdist(real_images[i].reshape(-1, 1), generated_images[i].reshape(-1, 1), 'euclidean')
        distances.append(np.mean(dist))
    return np.mean(distances)
