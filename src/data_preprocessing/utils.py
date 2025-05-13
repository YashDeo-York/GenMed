# src/data_processing/utils.py
import numpy as np
import cv2 # OpenCV for image resizing, used in preprocess_images

def norm(image_2D):
    """
    Normalize a 2D image to the range [0, 1].
    Clips values outside the 1st and 99th percentiles.
    """
    # Remove outliers by clipping values outside the 1st and 99th percentiles
    p1 = np.percentile(image_2D, 1)
    p99 = np.percentile(image_2D, 99)
    image_2D = np.clip(image_2D, p1, p99)
    
    # Normalize to [0, 1]
    image_norm = (image_2D - np.min(image_2D)) / (np.max(image_2D) - np.min(image_2D) + 1e-6) # Add epsilon to avoid division by zero
    return image_norm

def norm3(image_3D):
    """
    Normalize a 3D image (stack of 2D images) slice by slice to the range [0, 1].
    """
    image_norm_3D = np.zeros_like(image_3D, dtype=np.float32)
    for i in range(image_3D.shape[2]):
        image_norm_3D[:, :, i] = norm(image_3D[:, :, i])
    return image_norm_3D

def centre_crop_2D(image_2D, crop_size=(192, 192)):
    """
    Perform a center crop on a 2D image.
    Args:
        image_2D (np.array): Input 2D image.
        crop_size (tuple): Desired (height, width) for the crop.
    Returns:
        np.array: Center-cropped image.
    """
    h, w = image_2D.shape[:2] # Works for grayscale or single channel depth
    ch, cw = crop_size

    if h < ch or w < cw:
        raise ValueError(f"Crop size ({ch},{cw}) is larger than image size ({h},{w}).")

    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    
    if image_2D.ndim == 2: # Grayscale
        return image_2D[start_h:start_h + ch, start_w:start_w + cw]
    elif image_2D.ndim == 3: # Assuming last dimension is channel
         return image_2D[start_h:start_h + ch, start_w:start_w + cw, :]
    else:
        raise ValueError(f"Unsupported image dimension: {image_2D.ndim}")


def preprocess_images_for_inception(images_list, target_size=(299, 299)):
    """
    Preprocesses a list of images for InceptionV3 model.
    - Resizes to target_size.
    - Converts grayscale to 3-channel RGB by repeating the channel.
    - Normalizes pixel values to the range [-1, 1] as expected by InceptionV3.

    Args:
        images_list (list of np.array): List of 2D grayscale images (H, W).
                                         Assumes pixel values are already in a suitable range 
                                         (e.g., 0-1 or 0-255) before [-1,1] normalization.
                                         If images are not 0-1, they should be normalized first.
        target_size (tuple): The target (height, width) for resizing.

    Returns:
        np.array: Batch of preprocessed images (N, H, W, 3).
    """
    processed_images = []
    for img_2d in images_list:
        # Ensure image is float32 for processing
        img = img_2d.astype(np.float32)

        # If image is not already 0-1, normalize it.
        # This step is crucial. Assuming input images might be, e.g., 0-255 or already 0-1.
        # If they are from norm() or norm3(), they are already 0-1.
        if img.max() > 1.0: # A simple check, might need refinement
            img = img / 255.0 
        
        # Resize
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) # INTER_AREA is good for downscaling

        # Add channel dimension if it's grayscale (H, W) -> (H, W, 1)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        # Convert to 3-channel RGB by repeating the grayscale channel
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] != 3:
            raise ValueError(f"Image has unsupported number of channels: {img.shape[-1]}")

        # Normalize to [-1, 1] for InceptionV3
        # InceptionV3 preprocessing: (pixel_value / 255.0) * 2.0 - 1.0
        # Or if already 0-1: pixel_value * 2.0 - 1.0
        img = (img * 2.0) - 1.0 
        
        processed_images.append(img)
        
    return np.array(processed_images)

