"""
Pipeline 3: Comprehensive Enhancement Workflow

This pipeline combines the most effective techniques from Pipeline 1 and 2:
1. Multi-stage Processing: Sequential application of best filters
2. Adaptive Processing: Automatic parameter adjustment based on image characteristics
3. Final Sharpening: Additional sharpening pass for optimal results
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any

from .pipeline1 import Pipeline1, NoiseFilters, ContrastEnhancement, BrightnessAdjustments
from .pipeline2 import Pipeline2, EdgeDetection, MorphologicalOps, BoundaryEnhancement


class Pipeline3:
    """
    Comprehensive Enhancement Pipeline combining Pipeline 1 and Pipeline 2.
    
    Provides both fixed-parameter and adaptive processing modes.
    """
    
    def __init__(self):
        self.pipeline1 = Pipeline1()
        self.pipeline2 = Pipeline2()
        self.noise_filters = NoiseFilters()
        self.contrast = ContrastEnhancement()
        self.brightness = BrightnessAdjustments()
        self.edge_detector = EdgeDetection()
        self.morph = MorphologicalOps()
        self.boundary = BoundaryEnhancement()
    
    def process(self, 
                img: np.ndarray,
                filter_type: str = 'bilateral',
                contrast_type: str = 'clahe',
                edge_method: str = 'canny',
                final_sharpen: bool = True) -> np.ndarray:
        """
        Complete multi-stage processing with fixed parameters.
        
        Steps:
        1. Noise reduction (from Pipeline 1)
        2. Contrast enhancement (from Pipeline 1)
        3. Edge detection and boundary enhancement (from Pipeline 2)
        4. Final sharpening (optional)
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        filter_type : str
            Type of noise filter ('gaussian', 'median', 'bilateral', 'wiener', 'nlm')
        contrast_type : str
            Type of contrast enhancement ('hist_eq', 'clahe', 'adaptive_clahe')
        edge_method : str
            Edge detection method ('sobel', 'prewitt', 'canny', 'log')
        final_sharpen : bool
            Apply final sharpening pass
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        # Stage 1 & 2: Apply Pipeline 1 (noise reduction + contrast)
        enhanced = self.pipeline1.process(img, 
                                          filter_type=filter_type,
                                          contrast_type=contrast_type)
        
        # Stage 3: Apply Pipeline 2 (edge enhancement)
        result, edges = self.pipeline2.process(enhanced,
                                               edge_method=edge_method,
                                               refine=True,
                                               sharpen_strength=0.3)
        
        # Stage 4: Final sharpening (optional)
        if final_sharpen:
            result = self._apply_final_sharpening(result)
        
        return result
    
    def _apply_final_sharpening(self, img: np.ndarray, 
                                 strength: float = 0.3) -> np.ndarray:
        """
        Apply final sharpening using unsharp masking.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        strength : float
            Sharpening strength (0.0 - 1.0)
            
        Returns:
        --------
        np.ndarray
            Sharpened image
        """
        # Use a mild sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # Blend original with sharpened
        result = cv2.addWeighted(img, 1 - strength, sharpened, strength, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def analyze_image(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image to determine optimal processing parameters.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of image characteristics and recommended parameters
        """
        # Calculate image statistics
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        
        # Estimate noise level using Laplacian variance
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Calculate histogram entropy
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Dynamic range
        dynamic_range = np.max(img) - np.min(img)
        
        # Determine processing recommendations
        recommendations = {}
        
        # Noise filter recommendation
        if laplacian_var > 500:
            recommendations['filter_type'] = 'bilateral'
            recommendations['filter_reason'] = 'High noise detected'
        elif laplacian_var > 200:
            recommendations['filter_type'] = 'gaussian'
            recommendations['filter_reason'] = 'Moderate noise detected'
        else:
            recommendations['filter_type'] = 'median'
            recommendations['filter_reason'] = 'Low noise, preserving edges'
        
        # Contrast recommendation
        if std_intensity < 30:
            recommendations['contrast_type'] = 'clahe'
            recommendations['contrast_reason'] = 'Low contrast - CLAHE recommended'
            recommendations['clahe_clip'] = 3.0
        elif std_intensity < 50:
            recommendations['contrast_type'] = 'clahe'
            recommendations['contrast_reason'] = 'Moderate contrast'
            recommendations['clahe_clip'] = 2.0
        else:
            recommendations['contrast_type'] = 'hist_eq'
            recommendations['contrast_reason'] = 'Good contrast - histogram eq sufficient'
            recommendations['clahe_clip'] = 1.5
        
        # Brightness recommendation
        if mean_intensity < 80:
            recommendations['gamma'] = 0.7
            recommendations['brightness_reason'] = 'Image too dark'
        elif mean_intensity > 180:
            recommendations['gamma'] = 1.3
            recommendations['brightness_reason'] = 'Image too bright'
        else:
            recommendations['gamma'] = 1.0
            recommendations['brightness_reason'] = 'Good brightness'
        
        return {
            'statistics': {
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'laplacian_variance': laplacian_var,
                'entropy': float(entropy),
                'dynamic_range': dynamic_range
            },
            'recommendations': recommendations
        }
    
    def adaptive_process(self, img: np.ndarray) -> np.ndarray:
        """
        Automatically select best parameters based on image analysis.
        
        This method analyzes the input image and chooses optimal
        processing parameters for each stage.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        # Analyze image
        analysis = self.analyze_image(img)
        recommendations = analysis['recommendations']
        
        # Apply noise filtering based on recommendation
        filter_type = recommendations['filter_type']
        if filter_type == 'bilateral':
            filtered = self.noise_filters.bilateral_filter(img)
        elif filter_type == 'gaussian':
            filtered = self.noise_filters.gaussian_filter(img)
        else:
            filtered = self.noise_filters.median_filter(img)
        
        # Apply contrast enhancement
        contrast_type = recommendations['contrast_type']
        if contrast_type == 'clahe':
            clip_limit = recommendations.get('clahe_clip', 2.0)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)
        else:
            enhanced = cv2.equalizeHist(filtered)
        
        # Apply gamma correction if needed
        gamma = recommendations.get('gamma', 1.0)
        if gamma != 1.0:
            enhanced = self.brightness.gamma_correction(enhanced, gamma)
        
        # Apply edge enhancement with Canny (generally best for OCT)
        result, _ = self.pipeline2.process(enhanced, 
                                           edge_method='canny',
                                           refine=True,
                                           sharpen_strength=0.25)
        
        # Final mild sharpening
        result = self._apply_final_sharpening(result, strength=0.2)
        
        return result
    
    def process_with_stages(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process image and return results from each stage.
        
        Useful for visualization and analysis of the enhancement pipeline.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary containing results from each stage
        """
        stages = {'original': img.copy()}
        
        # Stage 1: Noise filtering
        filtered = self.noise_filters.bilateral_filter(img)
        stages['noise_filtered'] = filtered
        
        # Stage 2: Contrast enhancement
        enhanced = self.contrast.clahe(filtered)
        stages['contrast_enhanced'] = enhanced
        
        # Stage 3: Edge detection
        edges = self.edge_detector.canny(enhanced)
        stages['edges'] = edges
        
        # Stage 4: Morphological refinement
        refined_edges = self.morph.refine_edges(edges)
        stages['refined_edges'] = refined_edges
        
        # Stage 5: Edge-based sharpening
        sharpened = self.boundary.sharpen_with_edges(enhanced, refined_edges, 0.3)
        stages['edge_sharpened'] = sharpened
        
        # Stage 6: Final sharpening
        final = self._apply_final_sharpening(sharpened)
        stages['final'] = final
        
        return stages
    
    def process_conservative(self, img: np.ndarray) -> np.ndarray:
        """
        Conservative processing that preserves more of the original image.
        
        Uses milder parameters to avoid over-processing.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        # Mild noise filtering
        filtered = self.noise_filters.bilateral_filter(img, d=5, 
                                                       sigma_color=50, 
                                                       sigma_space=50)
        
        # Conservative CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Light edge enhancement
        result, _ = self.pipeline2.process(enhanced,
                                           edge_method='canny',
                                           refine=True,
                                           sharpen_strength=0.15)
        
        return result
    
    def process_aggressive(self, img: np.ndarray) -> np.ndarray:
        """
        Aggressive processing for very low quality images.
        
        Uses stronger parameters for maximum enhancement.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        # Strong noise filtering
        filtered = self.noise_filters.non_local_means(img, h=12)
        
        # Strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Strong edge enhancement
        result, _ = self.pipeline2.process(enhanced,
                                           edge_method='canny',
                                           refine=True,
                                           sharpen_strength=0.5)
        
        # Stronger final sharpening
        result = self._apply_final_sharpening(result, strength=0.4)
        
        return result

