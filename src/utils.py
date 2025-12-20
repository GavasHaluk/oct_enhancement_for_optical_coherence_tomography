"""
Utility functions for OCT image enhancement project.
Provides image loading, saving, display, and dataset management functions.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional


def load_image(path: str) -> np.ndarray:
    """
    Load grayscale image from disk.
    
    Parameters:
    -----------
    path : str
        Path to the image file
        
    Returns:
    --------
    np.ndarray
        Grayscale image as numpy array
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def save_image(img: np.ndarray, path: str) -> None:
    """
    Save image to disk.
    
    Parameters:
    -----------
    img : np.ndarray
        Image to save
    path : str
        Output path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def display_images(images: List[np.ndarray], titles: List[str], 
                   figsize: Tuple[int, int] = (15, 5), 
                   cmap: str = 'gray') -> plt.Figure:
    """
    Display multiple images side by side.
    
    Parameters:
    -----------
    images : List[np.ndarray]
        List of images to display
    titles : List[str]
        Titles for each image
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap to use
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def get_all_images(directory: str, categories: List[str] = None) -> List[Dict]:
    """
    Get list of all image paths in directory.
    
    Parameters:
    -----------
    directory : str
        Base directory containing category subfolders
    categories : List[str], optional
        Categories to include. Defaults to ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        
    Returns:
    --------
    List[Dict]
        List of dictionaries with path, category, and filename keys
    """
    if categories is None:
        categories = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    
    image_paths = []
    
    for category in categories:
        cat_dir = os.path.join(directory, category)
        if os.path.exists(cat_dir):
            for filename in os.listdir(cat_dir):
                if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                    image_paths.append({
                        'path': os.path.join(cat_dir, filename),
                        'category': category,
                        'filename': filename
                    })
    
    return image_paths


def estimate_image_quality(img: np.ndarray) -> Dict[str, float]:
    """
    Estimate image quality based on various metrics.
    Used to classify images as high or low quality.
    
    Parameters:
    -----------
    img : np.ndarray
        Grayscale image
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing quality metrics
    """
    # Sharpness estimation using Laplacian variance
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    
    # Contrast estimation using standard deviation
    contrast = np.std(img)
    
    # Mean intensity
    mean_intensity = np.mean(img)
    
    # Dynamic range utilization
    dynamic_range = (np.max(img) - np.min(img)) / 255.0
    
    # Entropy (information content)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Combined quality score (higher is better)
    # Normalized and weighted combination
    quality_score = (
        0.3 * min(laplacian_var / 1000, 1.0) +  # Sharpness (cap at 1000)
        0.3 * (contrast / 80) +                   # Contrast (normalized)
        0.2 * dynamic_range +                     # Dynamic range
        0.2 * (entropy / 8)                       # Entropy (normalized)
    )
    
    return {
        'sharpness': laplacian_var,
        'contrast': contrast,
        'mean_intensity': mean_intensity,
        'dynamic_range': dynamic_range,
        'entropy': float(entropy),
        'quality_score': quality_score
    }


def classify_images_by_quality(image_paths: List[Dict], 
                                threshold_percentile: float = 75) -> Tuple[List[Dict], List[Dict]]:
    """
    Classify images into high and low quality groups based on quality metrics.
    
    Parameters:
    -----------
    image_paths : List[Dict]
        List of image info dictionaries
    threshold_percentile : float
        Percentile threshold for high quality classification
        
    Returns:
    --------
    Tuple[List[Dict], List[Dict]]
        (high_quality_images, low_quality_images)
    """
    quality_scores = []
    
    for img_info in image_paths:
        img = load_image(img_info['path'])
        quality = estimate_image_quality(img)
        img_info['quality'] = quality
        quality_scores.append(quality['quality_score'])
    
    # Determine threshold
    threshold = np.percentile(quality_scores, threshold_percentile)
    
    high_quality = [info for info in image_paths 
                    if info['quality']['quality_score'] >= threshold]
    low_quality = [info for info in image_paths 
                   if info['quality']['quality_score'] < threshold]
    
    return high_quality, low_quality


def create_subset(source_dir: str, dest_dir: str, 
                  images_per_category: int = 250,
                  seed: int = 42) -> Dict[str, int]:
    """
    Create a subset of images from the source directory.
    
    Parameters:
    -----------
    source_dir : str
        Source directory (e.g., OCT/train)
    dest_dir : str
        Destination directory for subset
    images_per_category : int
        Number of images per category
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, int]
        Number of images copied per category
    """
    import random
    import shutil
    
    random.seed(seed)
    categories = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    copied_counts = {}
    
    for category in categories:
        src = os.path.join(source_dir, category)
        dst = os.path.join(dest_dir, category)
        os.makedirs(dst, exist_ok=True)
        
        if not os.path.exists(src):
            print(f"Warning: Source directory {src} does not exist")
            copied_counts[category] = 0
            continue
        
        # Get all images
        images = [f for f in os.listdir(src) 
                  if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        
        # Random selection
        selected = random.sample(images, min(images_per_category, len(images)))
        
        # Copy files
        for img in selected:
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))
        
        copied_counts[category] = len(selected)
        print(f"Copied {len(selected)} images from {category}")
    
    return copied_counts


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image
        
    Returns:
    --------
    np.ndarray
        Normalized image
    """
    if img.max() == img.min():
        return np.zeros_like(img, dtype=np.uint8)
    
    normalized = ((img - img.min()) / (img.max() - img.min()) * 255)
    return normalized.astype(np.uint8)

