"""
Pipeline 1: Noise Reduction and Contrast Enhancement

This pipeline focuses on:
1. Noise Filtering: Gaussian, median, bilateral, and Wiener filters
2. Contrast Enhancement: Histogram equalization and CLAHE
3. Brightness Adjustments: Gamma correction and logarithmic transform
"""

import cv2
import numpy as np
from scipy.signal import wiener
from typing import Optional, Literal


class NoiseFilters:
    """
    Noise reduction filters for OCT images.
    Implements multiple spatial filtering techniques to reduce speckle noise.
    """
    
    @staticmethod
    def gaussian_filter(img: np.ndarray, 
                        kernel_size: int = 5, 
                        sigma: float = 1.0) -> np.ndarray:
        """
        Gaussian blur for noise reduction.
        
        Good for general smoothing but may blur edges.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        kernel_size : int
            Size of Gaussian kernel (must be odd)
        sigma : float
            Standard deviation of Gaussian
            
        Returns:
        --------
        np.ndarray
            Filtered image
        """
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def median_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Median filter - particularly effective for salt-and-pepper noise.
        
        Preserves edges better than Gaussian while removing impulse noise.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        kernel_size : int
            Size of median kernel (must be odd)
            
        Returns:
        --------
        np.ndarray
            Filtered image
        """
        return cv2.medianBlur(img, kernel_size)
    
    @staticmethod
    def bilateral_filter(img: np.ndarray, 
                         d: int = 9, 
                         sigma_color: float = 75, 
                         sigma_space: float = 75) -> np.ndarray:
        """
        Bilateral filter - preserves edges while smoothing.
        
        This is often the best choice for OCT images as it reduces
        speckle noise while maintaining retinal layer boundaries.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        d : int
            Diameter of pixel neighborhood
        sigma_color : float
            Filter sigma in the color space
        sigma_space : float
            Filter sigma in the coordinate space
            
        Returns:
        --------
        np.ndarray
            Filtered image
        """
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    @staticmethod
    def wiener_filter(img: np.ndarray, 
                      window_size: int = 5,
                      noise_variance: Optional[float] = None) -> np.ndarray:
        """
        Wiener filter for noise reduction.
        
        Optimal filter in the sense of minimizing mean square error.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        window_size : int
            Size of the local window
        noise_variance : float, optional
            Noise variance (estimated if None)
            
        Returns:
        --------
        np.ndarray
            Filtered image
        """
        # Apply 2D Wiener filtering
        filtered = wiener(img.astype(np.float64), 
                          mysize=(window_size, window_size),
                          noise=noise_variance)
        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    @staticmethod
    def non_local_means(img: np.ndarray,
                        h: float = 10,
                        template_window: int = 7,
                        search_window: int = 21) -> np.ndarray:
        """
        Non-local means denoising.
        
        Advanced denoising that uses similarity between patches.
        Slower but often produces excellent results.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        h : float
            Filter strength
        template_window : int
            Size of template patch
        search_window : int
            Size of search area
            
        Returns:
        --------
        np.ndarray
            Denoised image
        """
        return cv2.fastNlMeansDenoising(img, None, h, 
                                         template_window, search_window)


class ContrastEnhancement:
    """
    Contrast enhancement methods for OCT images.
    Improves visibility of retinal features and abnormalities.
    """
    
    @staticmethod
    def histogram_equalization(img: np.ndarray) -> np.ndarray:
        """
        Standard histogram equalization.
        
        Spreads out intensity values to use full dynamic range.
        May over-enhance noise in some regions.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        return cv2.equalizeHist(img)
    
    @staticmethod
    def clahe(img: np.ndarray, 
              clip_limit: float = 2.0, 
              tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Applies histogram equalization locally with clipping to prevent
        over-amplification of noise. Best for medical images.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        clip_limit : float
            Threshold for contrast limiting (2.0-4.0 typical)
        tile_grid_size : tuple
            Size of grid for histogram equalization
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, 
                                     tileGridSize=tile_grid_size)
        return clahe_obj.apply(img)
    
    @staticmethod
    def adaptive_clahe(img: np.ndarray) -> np.ndarray:
        """
        CLAHE with parameters adapted to image characteristics.
        
        Automatically adjusts clip limit based on image contrast.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        # Analyze image contrast
        std = np.std(img)
        
        # Lower clip limit for high contrast images
        # Higher clip limit for low contrast images
        if std < 30:
            clip_limit = 3.0
        elif std < 50:
            clip_limit = 2.0
        else:
            clip_limit = 1.5
        
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, 
                                     tileGridSize=(8, 8))
        return clahe_obj.apply(img)


class BrightnessAdjustments:
    """
    Intensity transformation methods for brightness optimization.
    """
    
    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Gamma correction for brightness adjustment.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        gamma : float
            Gamma value (< 1: brighten, > 1: darken)
            
        Returns:
        --------
        np.ndarray
            Adjusted image
        """
        # Build lookup table for efficiency
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(256)]).astype(np.uint8)
        return cv2.LUT(img, table)
    
    @staticmethod
    def log_transform(img: np.ndarray, c: float = None) -> np.ndarray:
        """
        Logarithmic transformation.
        
        Expands dark regions and compresses bright regions.
        Useful for images with large dynamic range.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        c : float, optional
            Scaling constant (computed automatically if None)
            
        Returns:
        --------
        np.ndarray
            Transformed image
        """
        if c is None:
            c = 255 / np.log(1 + np.max(img))
        
        log_transformed = c * np.log(1 + img.astype(np.float32))
        return np.uint8(np.clip(log_transformed, 0, 255))
    
    @staticmethod
    def adaptive_gamma(img: np.ndarray) -> np.ndarray:
        """
        Automatically determine and apply appropriate gamma correction.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
            
        Returns:
        --------
        np.ndarray
            Adjusted image
        """
        mean_intensity = np.mean(img)
        
        # Target mean intensity around 128 (middle of range)
        if mean_intensity < 100:
            # Dark image - brighten
            gamma = 0.6 + (mean_intensity / 250)
        elif mean_intensity > 160:
            # Bright image - darken
            gamma = 1.0 + ((mean_intensity - 128) / 255)
        else:
            # Good brightness - minimal adjustment
            gamma = 1.0
        
        return BrightnessAdjustments.gamma_correction(img, gamma)


class Pipeline1:
    """
    Complete Pipeline 1: Noise Reduction + Contrast Enhancement + Brightness
    
    Combines filtering, contrast enhancement, and brightness adjustment
    in an optimal sequence for OCT image enhancement.
    """
    
    def __init__(self):
        self.noise_filters = NoiseFilters()
        self.contrast = ContrastEnhancement()
        self.brightness = BrightnessAdjustments()
    
    def process(self, 
                img: np.ndarray, 
                filter_type: Literal['gaussian', 'median', 'bilateral', 
                                     'wiener', 'nlm'] = 'bilateral',
                contrast_type: Literal['hist_eq', 'clahe', 
                                       'adaptive_clahe', None] = 'clahe',
                gamma: Optional[float] = None,
                auto_brightness: bool = False) -> np.ndarray:
        """
        Apply Pipeline 1 processing.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        filter_type : str
            Type of noise filter to use
        contrast_type : str or None
            Type of contrast enhancement
        gamma : float, optional
            Gamma correction value (None for no gamma)
        auto_brightness : bool
            Use adaptive brightness adjustment
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        result = img.copy()
        
        # Step 1: Noise filtering
        if filter_type == 'gaussian':
            result = self.noise_filters.gaussian_filter(result)
        elif filter_type == 'median':
            result = self.noise_filters.median_filter(result)
        elif filter_type == 'bilateral':
            result = self.noise_filters.bilateral_filter(result)
        elif filter_type == 'wiener':
            result = self.noise_filters.wiener_filter(result)
        elif filter_type == 'nlm':
            result = self.noise_filters.non_local_means(result)
        
        # Step 2: Contrast enhancement
        if contrast_type == 'hist_eq':
            result = self.contrast.histogram_equalization(result)
        elif contrast_type == 'clahe':
            result = self.contrast.clahe(result)
        elif contrast_type == 'adaptive_clahe':
            result = self.contrast.adaptive_clahe(result)
        
        # Step 3: Brightness adjustment
        if gamma is not None:
            result = self.brightness.gamma_correction(result, gamma)
        elif auto_brightness:
            result = self.brightness.adaptive_gamma(result)
        
        return result
    
    def process_all_combinations(self, img: np.ndarray) -> dict:
        """
        Process image with all filter combinations and return results.
        
        Useful for comparing different processing strategies.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        dict
            Dictionary of results with combination names as keys
        """
        filter_types = ['gaussian', 'median', 'bilateral', 'wiener']
        contrast_types = ['hist_eq', 'clahe']
        
        results = {}
        
        for f_type in filter_types:
            for c_type in contrast_types:
                name = f"{f_type}_{c_type}"
                results[name] = self.process(img, 
                                            filter_type=f_type,
                                            contrast_type=c_type)
        
        return results

