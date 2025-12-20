"""
Visualization Module for OCT Image Enhancement Analysis

Provides functions for creating:
- Side-by-side pipeline comparisons
- Metrics comparison charts
- Histogram analyses
- Category performance visualizations
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import List, Dict, Optional, Tuple
import random

from .utils import load_image, get_all_images


# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Visualizer:
    """
    Create visualizations for OCT enhancement analysis and reporting.
    """
    
    @staticmethod
    def compare_all_pipelines(original_path: str, output_dir: str,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create side-by-side comparison of all 3 pipelines.
        
        Parameters:
        -----------
        original_path : str
            Path to original image
        output_dir : str
            Directory containing pipeline outputs
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        # Parse path info
        parts = original_path.replace('\\', '/').split('/')
        category = parts[-2]
        filename = parts[-1]
        
        # Load all images
        original = load_image(original_path)
        
        p1_path = os.path.join(output_dir, 'pipeline1', category, filename)
        p2_path = os.path.join(output_dir, 'pipeline2', category, filename)
        p3_path = os.path.join(output_dir, 'pipeline3', category, filename)
        
        images = [original]
        titles = ['Original']
        
        for path, name in [(p1_path, 'Pipeline 1\n(Noise + Contrast)'),
                           (p2_path, 'Pipeline 2\n(P1 + Edges)'),
                           (p3_path, 'Pipeline 3\n(Adaptive)')]:
            if os.path.exists(path):
                images.append(load_image(path))
                titles.append(name)
        
        # Create figure
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        
        if n == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle(f'Pipeline Comparison - {category}', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(metrics_df: pd.DataFrame,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create bar charts comparing metrics across pipelines.
        
        Parameters:
        -----------
        metrics_df : pd.DataFrame
            DataFrame with metrics for all pipelines
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        # Calculate averages by pipeline
        key_metrics = ['SSIM', 'Contrast_Improvement_Index', 
                       'Edge_Preservation_Index', 'Sharpness_Ratio']
        
        available = [m for m in key_metrics if m in metrics_df.columns]
        avg_metrics = metrics_df.groupby('pipeline')[available].mean()
        
        # Create figure
        n_metrics = len(available)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, metric in enumerate(available):
            values = avg_metrics[metric]
            bars = axes[idx].bar(range(len(values)), values, color=colors[:len(values)])
            
            axes[idx].set_xticks(range(len(values)))
            axes[idx].set_xticklabels([p.replace('pipeline', 'P') for p in values.index])
            axes[idx].set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, v) in enumerate(zip(bars, values)):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                              f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Average Metrics Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        return fig
    
    @staticmethod
    def plot_category_performance(metrics_df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Show how each pipeline performs on different categories.
        
        Parameters:
        -----------
        metrics_df : pd.DataFrame
            DataFrame with metrics for all pipelines
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        metrics_to_plot = ['SSIM', 'Contrast_Improvement_Index',
                           'Edge_Preservation_Index', 'Sharpness_Ratio']
        available = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        for idx, metric in enumerate(available[:4]):
            # Group by pipeline and category
            grouped = metrics_df.groupby(['pipeline', 'category'])[metric].mean().unstack()
            
            grouped.plot(kind='bar', ax=axes[idx], width=0.8)
            axes[idx].set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Pipeline')
            axes[idx].set_ylabel('Value')
            axes[idx].legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=0)
            
            # Update x-tick labels
            axes[idx].set_xticklabels([p.replace('pipeline', 'P') 
                                       for p in grouped.index])
        
        plt.suptitle('Performance by Category', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        return fig
    
    @staticmethod
    def create_histogram_comparison(original: np.ndarray, 
                                     enhanced: np.ndarray,
                                     title: str = '',
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Show histogram comparison before and after enhancement.
        
        Parameters:
        -----------
        original : np.ndarray
            Original image
        enhanced : np.ndarray
            Enhanced image
        title : str
            Title for the figure
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Enhanced image
        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title('Enhanced Image', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Original histogram
        axes[1, 0].hist(original.ravel(), bins=256, range=[0, 256],
                        color='#3498db', alpha=0.7, density=True)
        axes[1, 0].set_title('Original Histogram', fontweight='bold')
        axes[1, 0].set_xlabel('Pixel Intensity')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(alpha=0.3)
        
        # Enhanced histogram
        axes[1, 1].hist(enhanced.ravel(), bins=256, range=[0, 256],
                        color='#2ecc71', alpha=0.7, density=True)
        axes[1, 1].set_title('Enhanced Histogram', fontweight='bold')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(alpha=0.3)
        
        if title:
            plt.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        return fig
    
    @staticmethod
    def plot_processing_stages(stages: Dict[str, np.ndarray],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize all processing stages.
        
        Parameters:
        -----------
        stages : Dict[str, np.ndarray]
            Dictionary of stage names to images
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        n = len(stages)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).ravel()
        
        for idx, (name, img) in enumerate(stages.items()):
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(name.replace('_', ' ').title(), 
                               fontsize=11, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide unused axes
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Processing Stages', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        return fig
    
    @staticmethod
    def plot_quality_improvement(metrics_df: pd.DataFrame,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize quality-based SSIM improvement.
        
        Shows how enhancement improves similarity to high-quality references.
        
        Parameters:
        -----------
        metrics_df : pd.DataFrame
            DataFrame with quality-based metrics
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if 'ssim_improvement' not in metrics_df.columns:
            print("Quality-based metrics not available")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot of SSIM improvement by pipeline
        sns.boxplot(data=metrics_df, x='pipeline', y='ssim_improvement', 
                    ax=axes[0], palette='husl')
        axes[0].set_title('SSIM Improvement Distribution', fontweight='bold')
        axes[0].set_xlabel('Pipeline')
        axes[0].set_ylabel('SSIM Improvement')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Bar plot of mean improvement by category and pipeline
        improvement_by_cat = metrics_df.groupby(['category', 'pipeline'])['ssim_improvement'].mean().unstack()
        improvement_by_cat.plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_title('Mean SSIM Improvement by Category', fontweight='bold')
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('SSIM Improvement')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].legend(title='Pipeline')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Quality-Based Enhancement Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        return fig
    
    @staticmethod
    def generate_sample_comparisons(input_dir: str, output_dir: str,
                                     num_samples: int = 10,
                                     save_dir: str = None) -> None:
        """
        Generate comparison figures for random samples.
        
        Parameters:
        -----------
        input_dir : str
            Input images directory
        output_dir : str
            Pipeline outputs directory
        num_samples : int
            Number of samples to generate
        save_dir : str, optional
            Directory to save comparisons (defaults to output_dir/comparisons)
        """
        if save_dir is None:
            save_dir = os.path.join(output_dir, 'comparisons')
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get all images
        all_images = get_all_images(input_dir)
        
        if not all_images:
            print(f"No images found in {input_dir}")
            return
        
        # Select random samples, ensuring at least one from each category
        categories = list(set(img['category'] for img in all_images))
        selected = []
        
        # One from each category first
        for cat in categories:
            cat_images = [img for img in all_images if img['category'] == cat]
            if cat_images:
                selected.append(random.choice(cat_images))
        
        # Fill remaining with random selection
        remaining = [img for img in all_images if img not in selected]
        if remaining and len(selected) < num_samples:
            additional = random.sample(remaining, 
                                       min(num_samples - len(selected), len(remaining)))
            selected.extend(additional)
        
        print(f"\nGenerating {len(selected)} comparison figures...")
        
        for idx, img_info in enumerate(selected, 1):
            save_path = os.path.join(
                save_dir,
                f'comparison_{idx:02d}_{img_info["category"]}.png'
            )
            
            Visualizer.compare_all_pipelines(
                img_info['path'],
                output_dir,
                save_path
            )
        
        print(f"✓ Saved comparisons to: {save_dir}/")
    
    @staticmethod
    def create_summary_report(metrics_df: pd.DataFrame,
                               output_dir: str) -> None:
        """
        Generate all summary visualizations.
        
        Parameters:
        -----------
        metrics_df : pd.DataFrame
            Metrics DataFrame
        output_dir : str
            Output directory for figures
        """
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        print("\nGenerating summary visualizations...")
        
        # Metrics comparison
        Visualizer.plot_metrics_comparison(
            metrics_df,
            os.path.join(figures_dir, 'metrics_comparison.png')
        )
        print("  ✓ Metrics comparison")
        
        # Category performance
        Visualizer.plot_category_performance(
            metrics_df,
            os.path.join(figures_dir, 'category_performance.png')
        )
        print("  ✓ Category performance")
        
        # Quality improvement if available
        if 'ssim_improvement' in metrics_df.columns:
            Visualizer.plot_quality_improvement(
                metrics_df,
                os.path.join(figures_dir, 'quality_improvement.png')
            )
            print("  ✓ Quality improvement")
        
        print(f"\n✓ All visualizations saved to: {figures_dir}/")


class EdgeVisualization:
    """
    Specialized visualization for edge detection analysis.
    """
    
    @staticmethod
    def compare_edge_methods(img: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare different edge detection methods on an image.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        from .pipeline2 import EdgeDetection
        
        detector = EdgeDetection()
        
        methods = {
            'Original': img,
            'Sobel': detector.sobel(img),
            'Prewitt': detector.prewitt(img),
            'Canny': detector.canny(img),
            'Scharr': detector.scharr(img),
            'LoG': detector.laplacian_of_gaussian(img)
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(methods.items()):
            axes[idx].imshow(result, cmap='gray')
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('Edge Detection Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        return fig
    
    @staticmethod
    def compare_filter_methods(img: np.ndarray,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare different noise filtering methods on an image.
        
        Parameters:
        -----------
        img : np.ndarray
            Input image
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        from .pipeline1 import NoiseFilters
        
        filters = NoiseFilters()
        
        methods = {
            'Original': img,
            'Gaussian': filters.gaussian_filter(img),
            'Median': filters.median_filter(img),
            'Bilateral': filters.bilateral_filter(img),
            'Wiener': filters.wiener_filter(img),
            'NL-Means': filters.non_local_means(img)
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(methods.items()):
            axes[idx].imshow(result, cmap='gray')
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('Noise Filtering Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
        return fig

