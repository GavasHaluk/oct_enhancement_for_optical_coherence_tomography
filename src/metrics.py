"""
Metrics Calculator for OCT Image Enhancement Evaluation

This module provides comprehensive metrics for evaluating image enhancement:
1. Standard comparative metrics (SSIM, Contrast Improvement, Edge Preservation)
2. Quality-based SSIM: Compare enhanced images to high-quality references
3. Sharpness and entropy metrics
4. FID (Fréchet Inception Distance) for distribution similarity

Note: FID requires TensorFlow/Keras for feature extraction. If not available,
a simplified version using image statistics is provided.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import linalg
from typing import Dict, List, Tuple, Optional
import warnings


class MetricsCalculator:
    """
    Calculate image quality metrics for enhancement evaluation.
    """
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Structural Similarity Index (SSIM).
        
        Measures structural similarity between original and enhanced images.
        Range: -1 to 1 (1 = identical)
        For enhancement: 0.7-0.95 is typical for good enhancement
        
        Parameters:
        -----------
        original : np.ndarray
            Original image
        enhanced : np.ndarray
            Enhanced image
            
        Returns:
        --------
        float
            SSIM value
        """
        # Ensure same size
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
        
        return ssim(original, enhanced, data_range=255)
    
    @staticmethod
    def calculate_contrast_improvement(original: np.ndarray, 
                                        enhanced: np.ndarray) -> float:
        """
        Contrast Improvement Index (CII).
        
        Ratio of standard deviations - measures contrast enhancement.
        > 1.0 indicates improvement in contrast.
        
        Parameters:
        -----------
        original : np.ndarray
            Original image
        enhanced : np.ndarray
            Enhanced image
            
        Returns:
        --------
        float
            Contrast improvement ratio
        """
        std_orig = np.std(original.astype(np.float64))
        std_enh = np.std(enhanced.astype(np.float64))
        
        if std_orig == 0:
            return 0.0
        
        return std_enh / std_orig
    
    @staticmethod
    def calculate_edge_preservation(original: np.ndarray, 
                                     enhanced: np.ndarray) -> float:
        """
        Edge Preservation Index (EPI).
        
        Measures correlation between edge maps of original and enhanced images.
        Values close to 1.0 indicate edges are well preserved.
        
        Parameters:
        -----------
        original : np.ndarray
            Original image
        enhanced : np.ndarray
            Enhanced image
            
        Returns:
        --------
        float
            Edge preservation correlation
        """
        # Detect edges in both images
        edges_orig = cv2.Canny(original, 50, 150)
        edges_enh = cv2.Canny(enhanced, 50, 150)
        
        # Flatten for correlation
        edges_orig_flat = edges_orig.flatten().astype(np.float64)
        edges_enh_flat = edges_enh.flatten().astype(np.float64)
        
        # Handle edge case of zero variance
        if np.std(edges_orig_flat) == 0 or np.std(edges_enh_flat) == 0:
            return 0.0
        
        correlation = np.corrcoef(edges_orig_flat, edges_enh_flat)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def calculate_sharpness(img: np.ndarray) -> float:
        """
        Sharpness using Laplacian variance.
        
        Higher values indicate sharper images.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        float
            Sharpness value
        """
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return float(laplacian.var())
    
    @staticmethod
    def calculate_entropy(img: np.ndarray) -> float:
        """
        Shannon entropy - measures information content.
        
        Higher entropy indicates better use of dynamic range.
        Max theoretical value is 8 (for 256 gray levels).
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        float
            Entropy value
        """
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        # Avoid log(0)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Peak Signal-to-Noise Ratio (PSNR).
        
        Measures the ratio between the maximum possible power of a signal
        and the power of corrupting noise.
        
        Parameters:
        -----------
        original : np.ndarray
            Original image
        enhanced : np.ndarray
            Enhanced image
            
        Returns:
        --------
        float
            PSNR value in dB
        """
        mse = np.mean((original.astype(np.float64) - enhanced.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return float(psnr)
    
    @staticmethod
    def calculate_mean_gradient(img: np.ndarray) -> float:
        """
        Mean gradient magnitude.
        
        Another measure of image sharpness/detail.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        float
            Mean gradient value
        """
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        return float(np.mean(gradient))
    
    def calculate_all_metrics(self, original: np.ndarray, 
                               enhanced: np.ndarray) -> Dict[str, float]:
        """
        Calculate all metrics and return as dictionary.
        
        Parameters:
        -----------
        original : np.ndarray
            Original image
        enhanced : np.ndarray
            Enhanced image
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of all metrics
        """
        sharpness_orig = self.calculate_sharpness(original)
        sharpness_enh = self.calculate_sharpness(enhanced)
        entropy_orig = self.calculate_entropy(original)
        entropy_enh = self.calculate_entropy(enhanced)
        
        metrics = {
            # Comparative metrics
            'SSIM': self.calculate_ssim(original, enhanced),
            'PSNR': self.calculate_psnr(original, enhanced),
            'Contrast_Improvement_Index': self.calculate_contrast_improvement(original, enhanced),
            'Edge_Preservation_Index': self.calculate_edge_preservation(original, enhanced),
            
            # Individual quality metrics
            'Sharpness_Original': sharpness_orig,
            'Sharpness_Enhanced': sharpness_enh,
            'Entropy_Original': entropy_orig,
            'Entropy_Enhanced': entropy_enh,
            'Mean_Gradient_Original': self.calculate_mean_gradient(original),
            'Mean_Gradient_Enhanced': self.calculate_mean_gradient(enhanced),
            
            # Derived metrics
            'Sharpness_Ratio': sharpness_enh / (sharpness_orig + 1e-10),
            'Entropy_Gain': entropy_enh - entropy_orig
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print formatted metrics summary."""
        print("\n" + "=" * 60)
        print("IMAGE ENHANCEMENT METRICS")
        print("=" * 60)
        print(f"\n{'Metric':<35} {'Value':>15}")
        print("-" * 50)
        print(f"{'SSIM (Structure Similarity)':<35} {metrics['SSIM']:>15.4f}")
        print(f"{'PSNR (dB)':<35} {metrics['PSNR']:>15.2f}")
        print(f"{'Contrast Improvement':<35} {metrics['Contrast_Improvement_Index']:>15.3f}x")
        print(f"{'Edge Preservation':<35} {metrics['Edge_Preservation_Index']:>15.4f}")
        print(f"{'Sharpness Improvement':<35} {metrics['Sharpness_Ratio']:>15.3f}x")
        print(f"{'Entropy Gain':<35} {metrics['Entropy_Gain']:>+15.4f}")
        print("=" * 60 + "\n")


class QualityBasedSSIM:
    """
    Quality-based SSIM comparison.
    
    Compares enhanced images to high-quality reference images from the same
    category to measure how much the enhancement brings them closer to
    high-quality examples.
    
    This approach is useful when you don't have ground truth "clean" images
    but can identify high vs low quality images in the dataset.
    """
    
    def __init__(self):
        self.high_quality_refs = {}  # Category -> list of high quality images
    
    def set_references(self, category: str, 
                       reference_images: List[np.ndarray]) -> None:
        """
        Set high-quality reference images for a category.
        
        Parameters:
        -----------
        category : str
            Category name (e.g., 'NORMAL', 'DME')
        reference_images : List[np.ndarray]
            List of high-quality reference images
        """
        self.high_quality_refs[category] = reference_images
    
    def calculate_similarity_to_references(self, 
                                            image: np.ndarray,
                                            category: str) -> Dict[str, float]:
        """
        Calculate SSIM between an image and all reference images.
        
        Parameters:
        -----------
        image : np.ndarray
            Image to compare
        category : str
            Category of the image
            
        Returns:
        --------
        Dict[str, float]
            Mean, max, and median SSIM to references
        """
        if category not in self.high_quality_refs:
            return {'mean_ssim': 0.0, 'max_ssim': 0.0, 'median_ssim': 0.0}
        
        references = self.high_quality_refs[category]
        if not references:
            return {'mean_ssim': 0.0, 'max_ssim': 0.0, 'median_ssim': 0.0}
        
        ssim_scores = []
        for ref in references:
            # Resize if needed
            if ref.shape != image.shape:
                ref_resized = cv2.resize(ref, (image.shape[1], image.shape[0]))
            else:
                ref_resized = ref
            
            score = ssim(image, ref_resized, data_range=255)
            ssim_scores.append(score)
        
        return {
            'mean_ssim_to_refs': float(np.mean(ssim_scores)),
            'max_ssim_to_refs': float(np.max(ssim_scores)),
            'median_ssim_to_refs': float(np.median(ssim_scores))
        }
    
    def calculate_improvement(self, 
                               original: np.ndarray,
                               enhanced: np.ndarray,
                               category: str) -> Dict[str, float]:
        """
        Calculate how much enhancement improves similarity to references.
        
        Parameters:
        -----------
        original : np.ndarray
            Original low-quality image
        enhanced : np.ndarray
            Enhanced image
        category : str
            Category of the image
            
        Returns:
        --------
        Dict[str, float]
            SSIM improvements
        """
        orig_scores = self.calculate_similarity_to_references(original, category)
        enh_scores = self.calculate_similarity_to_references(enhanced, category)
        
        return {
            'original_mean_ssim': orig_scores['mean_ssim_to_refs'],
            'enhanced_mean_ssim': enh_scores['mean_ssim_to_refs'],
            'ssim_improvement': enh_scores['mean_ssim_to_refs'] - orig_scores['mean_ssim_to_refs'],
            'original_max_ssim': orig_scores['max_ssim_to_refs'],
            'enhanced_max_ssim': enh_scores['max_ssim_to_refs'],
            'max_ssim_improvement': enh_scores['max_ssim_to_refs'] - orig_scores['max_ssim_to_refs']
        }


class SimplifiedFID:
    """
    Simplified Fréchet Inception Distance (FID) calculation.
    
    FID measures the distance between the feature distributions of two sets
    of images. Lower FID indicates more similar distributions.
    
    This simplified version uses basic image statistics instead of deep
    features from InceptionV3. For a full implementation, use the
    pytorch-fid or tensorflow-fid packages.
    
    The formula is:
    FID = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    
    where mu1, mu2 are means and C1, C2 are covariance matrices.
    """
    
    @staticmethod
    def extract_features(img: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from image using statistics.
        
        Uses multiple image statistics as features:
        - Histogram features
        - Gradient features
        - Texture features (using Laplacian)
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
            
        Returns:
        --------
        np.ndarray
            Feature vector
        """
        features = []
        
        # Histogram features (16 bins)
        hist = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
        hist = hist / hist.sum()
        features.extend(hist)
        
        # Gradient statistics
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        features.extend([
            np.mean(gradient),
            np.std(gradient),
            np.percentile(gradient, 25),
            np.percentile(gradient, 50),
            np.percentile(gradient, 75)
        ])
        
        # Laplacian (texture) features
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        features.extend([
            np.mean(np.abs(laplacian)),
            np.std(laplacian),
            float(laplacian.var())
        ])
        
        # Global statistics
        features.extend([
            np.mean(img),
            np.std(img),
            np.percentile(img, 10),
            np.percentile(img, 90)
        ])
        
        return np.array(features)
    
    @staticmethod
    def calculate_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and covariance of feature vectors.
        
        Parameters:
        -----------
        features : np.ndarray
            Array of shape (n_samples, n_features)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (mean, covariance)
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    @staticmethod
    def calculate_fid(mu1: np.ndarray, sigma1: np.ndarray,
                      mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """
        Calculate Fréchet distance between two Gaussian distributions.
        
        Parameters:
        -----------
        mu1, sigma1 : Mean and covariance of first distribution
        mu2, sigma2 : Mean and covariance of second distribution
        
        Returns:
        --------
        float
            FID score (lower is better)
        """
        # Mean difference
        diff = mu1 - mu2
        
        # Product of covariance matrices
        try:
            # Compute sqrt of sigma1 @ sigma2
            covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
            
            # Handle numerical errors
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
            
        except Exception:
            # Fallback to simpler calculation if matrix operations fail
            fid = np.sum(diff**2) + np.trace(sigma1) + np.trace(sigma2)
        
        return float(fid)
    
    def compute_fid_between_sets(self, 
                                  images1: List[np.ndarray],
                                  images2: List[np.ndarray]) -> float:
        """
        Compute FID between two sets of images.
        
        Parameters:
        -----------
        images1 : List[np.ndarray]
            First set of images (e.g., original images)
        images2 : List[np.ndarray]
            Second set of images (e.g., enhanced images)
            
        Returns:
        --------
        float
            FID score
        """
        if len(images1) < 2 or len(images2) < 2:
            warnings.warn("Need at least 2 images per set for FID calculation")
            return float('inf')
        
        # Extract features for all images
        features1 = np.array([self.extract_features(img) for img in images1])
        features2 = np.array([self.extract_features(img) for img in images2])
        
        # Calculate statistics
        mu1, sigma1 = self.calculate_statistics(features1)
        mu2, sigma2 = self.calculate_statistics(features2)
        
        # Calculate FID
        return self.calculate_fid(mu1, sigma1, mu2, sigma2)


class QualityMetrics:
    """
    Combined quality metrics including quality-based SSIM and FID.
    
    Provides comprehensive evaluation metrics:
    - Quality-based SSIM comparison with high-quality references
    - FID for measuring distribution similarity
    """
    
    def __init__(self):
        self.basic_metrics = MetricsCalculator()
        self.quality_ssim = QualityBasedSSIM()
        self.fid_calculator = SimplifiedFID()
    
    def setup_quality_references(self, 
                                  high_quality_images: Dict[str, List[np.ndarray]]) -> None:
        """
        Set up high-quality reference images for each category.
        
        Parameters:
        -----------
        high_quality_images : Dict[str, List[np.ndarray]]
            Dictionary mapping category names to lists of high-quality images
        """
        for category, images in high_quality_images.items():
            self.quality_ssim.set_references(category, images)
    
    def evaluate_enhancement(self,
                              original: np.ndarray,
                              enhanced: np.ndarray,
                              category: str = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of image enhancement.
        
        Parameters:
        -----------
        original : np.ndarray
            Original image
        enhanced : np.ndarray
            Enhanced image
        category : str, optional
            Image category for quality-based SSIM
            
        Returns:
        --------
        Dict[str, float]
            All metrics
        """
        # Basic metrics
        metrics = self.basic_metrics.calculate_all_metrics(original, enhanced)
        
        # Quality-based SSIM if references are available
        if category and category in self.quality_ssim.high_quality_refs:
            quality_metrics = self.quality_ssim.calculate_improvement(
                original, enhanced, category
            )
            metrics.update(quality_metrics)
        
        return metrics
    
    def evaluate_batch(self,
                       originals: List[np.ndarray],
                       enhanced: List[np.ndarray],
                       high_quality_refs: List[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate a batch of images including FID calculation.
        
        Parameters:
        -----------
        originals : List[np.ndarray]
            List of original images
        enhanced : List[np.ndarray]
            List of enhanced images
        high_quality_refs : List[np.ndarray], optional
            List of high-quality reference images
            
        Returns:
        --------
        Dict[str, float]
            Batch metrics including FID
        """
        # Calculate FID between original and enhanced
        fid_orig_enh = self.fid_calculator.compute_fid_between_sets(
            originals, enhanced
        )
        
        metrics = {
            'FID_original_vs_enhanced': fid_orig_enh
        }
        
        # If high-quality references provided, calculate FID improvements
        if high_quality_refs and len(high_quality_refs) >= 2:
            fid_orig_ref = self.fid_calculator.compute_fid_between_sets(
                originals, high_quality_refs
            )
            fid_enh_ref = self.fid_calculator.compute_fid_between_sets(
                enhanced, high_quality_refs
            )
            
            metrics.update({
                'FID_original_vs_refs': fid_orig_ref,
                'FID_enhanced_vs_refs': fid_enh_ref,
                'FID_improvement': fid_orig_ref - fid_enh_ref  # Positive = improvement
            })
        
        return metrics

