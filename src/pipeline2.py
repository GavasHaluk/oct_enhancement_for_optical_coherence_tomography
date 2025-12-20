"""
Pipeline 2: Edge Detection and Layer Boundary Visualization

This pipeline focuses on:
1. Edge Detection: Sobel, Prewitt, Canny, and Laplacian of Gaussian operators
2. Morphological Operations: Opening and closing for edge refinement
3. Boundary Enhancement: Using edge detection to enhance structural boundaries
"""

import cv2
import numpy as np
from typing import Tuple, Literal, Optional


class EdgeDetection:
    """
    Edge detection methods for identifying retinal layer boundaries in OCT images.
    """
    
    @staticmethod
    def sobel(img: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Sobel edge detection.
        
        Computes gradient magnitude using Sobel operators.
        Good for detecting edges in both horizontal and vertical directions.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        ksize : int
            Kernel size for Sobel operator
            
        Returns:
        --------
        np.ndarray
            Edge map (gradient magnitude)
        """
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        
        sobel = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-255
        if sobel.max() > 0:
            sobel = np.uint8(255 * sobel / sobel.max())
        else:
            sobel = np.zeros_like(img, dtype=np.uint8)
        
        return sobel
    
    @staticmethod
    def prewitt(img: np.ndarray) -> np.ndarray:
        """
        Prewitt edge detection.
        
        Similar to Sobel but with different kernel weights.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
            
        Returns:
        --------
        np.ndarray
            Edge map
        """
        # Prewitt kernels
        kernelx = np.array([[1, 0, -1], 
                           [1, 0, -1], 
                           [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1], 
                           [0, 0, 0], 
                           [-1, -1, -1]], dtype=np.float32)
        
        prewittx = cv2.filter2D(img.astype(np.float32), -1, kernelx)
        prewitty = cv2.filter2D(img.astype(np.float32), -1, kernely)
        
        prewitt = np.sqrt(prewittx**2 + prewitty**2)
        
        # Normalize to 0-255
        if prewitt.max() > 0:
            prewitt = np.uint8(255 * prewitt / prewitt.max())
        else:
            prewitt = np.zeros_like(img, dtype=np.uint8)
        
        return prewitt
    
    @staticmethod
    def canny(img: np.ndarray, 
              threshold1: int = 50, 
              threshold2: int = 150,
              aperture_size: int = 3) -> np.ndarray:
        """
        Canny edge detection.
        
        Multi-stage algorithm that produces thin, well-defined edges.
        Often produces the cleanest results for OCT images.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        threshold1 : int
            First threshold for hysteresis
        threshold2 : int
            Second threshold for hysteresis
        aperture_size : int
            Aperture size for Sobel operator
            
        Returns:
        --------
        np.ndarray
            Binary edge map
        """
        return cv2.Canny(img, threshold1, threshold2, 
                         apertureSize=aperture_size)
    
    @staticmethod
    def adaptive_canny(img: np.ndarray) -> np.ndarray:
        """
        Canny edge detection with automatically determined thresholds.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
            
        Returns:
        --------
        np.ndarray
            Binary edge map
        """
        # Compute optimal thresholds using Otsu's method concept
        median_val = np.median(img)
        sigma = 0.33
        
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))
        
        return cv2.Canny(img, lower, upper)
    
    @staticmethod
    def laplacian_of_gaussian(img: np.ndarray, 
                               sigma: float = 1.0) -> np.ndarray:
        """
        Laplacian of Gaussian (LoG) edge detection.
        
        Combines Gaussian smoothing with Laplacian to detect edges
        at multiple scales.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
        sigma : float
            Standard deviation for Gaussian blur
            
        Returns:
        --------
        np.ndarray
            Edge map
        """
        # First apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        
        # Then apply Laplacian
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Take absolute value and normalize
        log = np.abs(log)
        if log.max() > 0:
            log = np.uint8(255 * log / log.max())
        else:
            log = np.zeros_like(img, dtype=np.uint8)
        
        return log
    
    @staticmethod
    def scharr(img: np.ndarray) -> np.ndarray:
        """
        Scharr edge detection.
        
        More accurate than Sobel for small kernels.
        
        Parameters:
        -----------
        img : np.ndarray
            Input grayscale image
            
        Returns:
        --------
        np.ndarray
            Edge map
        """
        scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        
        scharr = np.sqrt(scharrx**2 + scharry**2)
        
        if scharr.max() > 0:
            scharr = np.uint8(255 * scharr / scharr.max())
        else:
            scharr = np.zeros_like(img, dtype=np.uint8)
        
        return scharr


class MorphologicalOps:
    """
    Morphological operations for edge refinement and noise removal.
    """
    
    @staticmethod
    def opening(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Morphological opening (erosion followed by dilation).
        
        Removes small noise and separates weakly connected regions.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image (typically binary edge map)
        kernel_size : int
            Size of structuring element
            
        Returns:
        --------
        np.ndarray
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                           (kernel_size, kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def closing(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Morphological closing (dilation followed by erosion).
        
        Connects nearby edges and fills small holes.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image (typically binary edge map)
        kernel_size : int
            Size of structuring element
            
        Returns:
        --------
        np.ndarray
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                           (kernel_size, kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    @staticmethod
    def refine_edges(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply both closing and opening to refine edge maps.
        
        Closes small gaps first, then removes small noise artifacts.
        
        Parameters:
        -----------
        img : np.ndarray
            Input edge map
        kernel_size : int
            Size of structuring element
            
        Returns:
        --------
        np.ndarray
            Refined edge map
        """
        # First close to connect nearby edges
        closed = MorphologicalOps.closing(img, kernel_size)
        # Then open to remove small noise
        refined = MorphologicalOps.opening(closed, kernel_size)
        return refined
    
    @staticmethod
    def dilate(img: np.ndarray, kernel_size: int = 3, 
               iterations: int = 1) -> np.ndarray:
        """
        Dilation - expands white regions.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        kernel_size : int
            Size of structuring element
        iterations : int
            Number of times to apply
            
        Returns:
        --------
        np.ndarray
            Dilated image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                           (kernel_size, kernel_size))
        return cv2.dilate(img, kernel, iterations=iterations)
    
    @staticmethod
    def erode(img: np.ndarray, kernel_size: int = 3, 
              iterations: int = 1) -> np.ndarray:
        """
        Erosion - shrinks white regions.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        kernel_size : int
            Size of structuring element
        iterations : int
            Number of times to apply
            
        Returns:
        --------
        np.ndarray
            Eroded image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                           (kernel_size, kernel_size))
        return cv2.erode(img, kernel, iterations=iterations)


class BoundaryEnhancement:
    """
    Methods to enhance structural boundaries using detected edges.
    """
    
    @staticmethod
    def sharpen_with_edges(img: np.ndarray, edges: np.ndarray, 
                           strength: float = 0.5) -> np.ndarray:
        """
        Sharpen image using detected edges.
        
        Adds edge information to the original image to emphasize boundaries.
        
        Parameters:
        -----------
        img : np.ndarray
            Original grayscale image
        edges : np.ndarray
            Edge map
        strength : float
            How much to emphasize edges (0.0 - 1.0)
            
        Returns:
        --------
        np.ndarray
            Sharpened image
        """
        # Normalize both images
        edges_norm = edges.astype(np.float32) / 255.0
        img_norm = img.astype(np.float32) / 255.0
        
        # Add edges to image
        sharpened = img_norm + (edges_norm * strength)
        sharpened = np.clip(sharpened, 0, 1)
        
        return np.uint8(sharpened * 255)
    
    @staticmethod
    def unsharp_mask(img: np.ndarray, sigma: float = 1.0, 
                     strength: float = 1.5) -> np.ndarray:
        """
        Unsharp masking for edge enhancement.
        
        Subtracts blurred version from original to enhance edges.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        sigma : float
            Gaussian blur sigma
        strength : float
            Enhancement strength
            
        Returns:
        --------
        np.ndarray
            Sharpened image
        """
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = cv2.addWeighted(img, 1.0 + strength, 
                                    blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def overlay_edges(img: np.ndarray, edges: np.ndarray, 
                      color: tuple = (0, 255, 0)) -> np.ndarray:
        """
        Overlay colored edges on grayscale image.
        
        Creates a color visualization with edges highlighted.
        
        Parameters:
        -----------
        img : np.ndarray
            Original grayscale image
        edges : np.ndarray
            Binary edge map
        color : tuple
            BGR color for edges
            
        Returns:
        --------
        np.ndarray
            Color image with edge overlay
        """
        # Convert grayscale to BGR
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Create colored edge mask
        edge_mask = edges > 0
        img_color[edge_mask] = color
        
        return img_color
    
    @staticmethod
    def high_pass_filter(img: np.ndarray, 
                         kernel_size: int = 5) -> np.ndarray:
        """
        High-pass filter to enhance fine details.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        kernel_size : int
            Size of low-pass filter kernel
            
        Returns:
        --------
        np.ndarray
            High-pass filtered image
        """
        # Create low-pass filtered version
        low_pass = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Subtract to get high-pass
        high_pass = cv2.subtract(img, low_pass)
        
        # Enhance and add back
        enhanced = cv2.add(img, high_pass)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)


class Pipeline2:
    """
    Complete Pipeline 2: Edge Detection + Morphological Refinement + Boundary Enhancement
    
    Focuses on detecting and enhancing retinal layer boundaries.
    Should be applied after noise reduction (Pipeline 1) for best results.
    """
    
    def __init__(self):
        self.edge_detector = EdgeDetection()
        self.morph = MorphologicalOps()
        self.boundary = BoundaryEnhancement()
    
    def process(self, 
                img: np.ndarray, 
                edge_method: Literal['sobel', 'prewitt', 'canny', 
                                     'adaptive_canny', 'log', 'scharr'] = 'canny',
                refine: bool = True,
                sharpen_strength: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Pipeline 2 processing.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image (preferably pre-filtered from Pipeline 1)
        edge_method : str
            Edge detection method to use
        refine : bool
            Apply morphological refinement to edges
        sharpen_strength : float
            Strength of edge-based sharpening
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (enhanced_image, edge_map)
        """
        # Step 1: Detect edges
        if edge_method == 'sobel':
            edges = self.edge_detector.sobel(img)
        elif edge_method == 'prewitt':
            edges = self.edge_detector.prewitt(img)
        elif edge_method == 'canny':
            edges = self.edge_detector.canny(img)
        elif edge_method == 'adaptive_canny':
            edges = self.edge_detector.adaptive_canny(img)
        elif edge_method == 'log':
            edges = self.edge_detector.laplacian_of_gaussian(img)
        elif edge_method == 'scharr':
            edges = self.edge_detector.scharr(img)
        else:
            raise ValueError(f"Unknown edge method: {edge_method}")
        
        # Step 2: Refine edges with morphological operations
        if refine:
            edges = self.morph.refine_edges(edges)
        
        # Step 3: Sharpen image using detected edges
        result = self.boundary.sharpen_with_edges(img, edges, sharpen_strength)
        
        return result, edges
    
    def process_with_unsharp(self, img: np.ndarray, 
                              edge_method: str = 'canny') -> np.ndarray:
        """
        Process using unsharp masking in addition to edge enhancement.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        edge_method : str
            Edge detection method
            
        Returns:
        --------
        np.ndarray
            Enhanced image
        """
        # Apply edge-based enhancement
        enhanced, _ = self.process(img, edge_method)
        
        # Apply unsharp masking for additional sharpening
        result = self.boundary.unsharp_mask(enhanced)
        
        return result

