"""
Batch Processor for OCT Image Enhancement

Processes multiple images through all three pipelines and collects metrics.
Includes quality classification for quality-based SSIM evaluation.
"""

import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import get_all_images, save_image, load_image, estimate_image_quality
from .pipeline1 import Pipeline1
from .pipeline2 import Pipeline2
from .pipeline3 import Pipeline3
from .metrics import MetricsCalculator, QualityBasedSSIM, SimplifiedFID, QualityMetrics


class BatchProcessor:
    """
    Process all images through enhancement pipelines and collect metrics.
    
    Features:
    - Processes images through Pipeline 1, 2, and 3
    - Calculates comprehensive metrics for each pipeline
    - Supports quality-based evaluation (comparing to high-quality references)
    - Saves processed images and metrics to disk
    """
    
    def __init__(self, input_dir: str, output_dir: str, 
                 use_quality_based_metrics: bool = True):
        """
        Initialize batch processor.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing input images (with category subfolders)
        output_dir : str
            Directory to save processed images and metrics
        use_quality_based_metrics : bool
            Whether to use quality-based SSIM comparison
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_quality_based_metrics = use_quality_based_metrics
        
        # Initialize calculators
        self.metrics_calc = MetricsCalculator()
        self.quality_metrics = QualityMetrics()
        self.fid_calc = SimplifiedFID()
        
        # Initialize pipelines
        self.p1 = Pipeline1()
        self.p2 = Pipeline2()
        self.p3 = Pipeline3()
        
        # Categories
        self.categories = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        
        # Create output directories
        self._create_output_dirs()
        
        # High and low quality image storage
        self.high_quality_images = {cat: [] for cat in self.categories}
        self.low_quality_images = {cat: [] for cat in self.categories}
    
    def _create_output_dirs(self) -> None:
        """Create output directory structure."""
        for pipeline in ['pipeline1', 'pipeline2', 'pipeline3']:
            for category in self.categories:
                path = os.path.join(self.output_dir, pipeline, category)
                os.makedirs(path, exist_ok=True)
        
        # Comparisons and metrics directories
        os.makedirs(os.path.join(self.output_dir, 'comparisons'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'metrics'), exist_ok=True)
    
    def classify_images_by_quality(self, 
                                    threshold_percentile: float = 75) -> Tuple[List[Dict], List[Dict]]:
        """
        Classify images into high and low quality groups.
        
        Parameters:
        -----------
        threshold_percentile : float
            Percentile threshold for high quality (e.g., 75 means top 25% are high quality)
            
        Returns:
        --------
        Tuple[List[Dict], List[Dict]]
            (high_quality_images, low_quality_images)
        """
        print("\nClassifying images by quality...")
        
        all_images = get_all_images(self.input_dir, self.categories)
        quality_scores = []
        
        for img_info in tqdm(all_images, desc="Analyzing image quality"):
            img = load_image(img_info['path'])
            quality = estimate_image_quality(img)
            img_info['quality'] = quality
            quality_scores.append(quality['quality_score'])
        
        # Determine threshold
        threshold = np.percentile(quality_scores, threshold_percentile)
        
        high_quality = [info for info in all_images 
                        if info['quality']['quality_score'] >= threshold]
        low_quality = [info for info in all_images 
                       if info['quality']['quality_score'] < threshold]
        
        # Group by category
        for img_info in high_quality:
            category = img_info['category']
            img = load_image(img_info['path'])
            self.high_quality_images[category].append(img)
        
        for img_info in low_quality:
            category = img_info['category']
            img = load_image(img_info['path'])
            self.low_quality_images[category].append(img)
        
        print(f"\n✓ High quality images: {len(high_quality)}")
        print(f"✓ Low quality images: {len(low_quality)}")
        
        for category in self.categories:
            print(f"  {category}: {len(self.high_quality_images[category])} high, "
                  f"{len(self.low_quality_images[category])} low")
        
        # Set up quality-based SSIM references
        if self.use_quality_based_metrics:
            self.quality_metrics.setup_quality_references(self.high_quality_images)
        
        return high_quality, low_quality
    
    def process_single_image(self, img_info: Dict) -> Optional[Dict]:
        """
        Process one image through all pipelines.
        
        Parameters:
        -----------
        img_info : Dict
            Image information dictionary
            
        Returns:
        --------
        Optional[Dict]
            Results dictionary or None if processing failed
        """
        try:
            # Load image
            img = cv2.imread(img_info['path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            results = {}
            
            # Pipeline 1: Bilateral + CLAHE (best combination for OCT)
            p1_result = self.p1.process(img, 
                                        filter_type='bilateral',
                                        contrast_type='clahe')
            results['pipeline1'] = p1_result
            
            # Pipeline 2: Apply edge enhancement on filtered image
            p2_result, edges = self.p2.process(p1_result,
                                               edge_method='canny',
                                               refine=True,
                                               sharpen_strength=0.3)
            results['pipeline2'] = p2_result
            
            # Pipeline 3: Comprehensive workflow (adaptive)
            p3_result = self.p3.adaptive_process(img)
            results['pipeline3'] = p3_result
            
            # Calculate metrics for each pipeline
            all_metrics = {}
            for pipeline_name, result in results.items():
                # Basic metrics
                metrics = self.metrics_calc.calculate_all_metrics(img, result)
                
                # Quality-based SSIM if enabled
                if self.use_quality_based_metrics:
                    category = img_info['category']
                    if category in self.quality_metrics.quality_ssim.high_quality_refs:
                        quality_metrics = self.quality_metrics.quality_ssim.calculate_improvement(
                            img, result, category
                        )
                        metrics.update(quality_metrics)
                
                all_metrics[pipeline_name] = metrics
                
                # Save processed image
                output_path = os.path.join(
                    self.output_dir,
                    pipeline_name,
                    img_info['category'],
                    img_info['filename']
                )
                save_image(result, output_path)
            
            return {
                'filename': img_info['filename'],
                'category': img_info['category'],
                'metrics': all_metrics,
                'quality_score': img_info.get('quality', {}).get('quality_score', 0)
            }
            
        except Exception as e:
            print(f"\nError processing {img_info['filename']}: {e}")
            return None
    
    def process_all(self, 
                    classify_quality: bool = True,
                    save_csv: bool = True,
                    max_workers: int = 4) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Process all images through all pipelines.
        
        Parameters:
        -----------
        classify_quality : bool
            Whether to classify images by quality first
        save_csv : bool
            Whether to save metrics to CSV
        max_workers : int
            Number of parallel workers (use 1 for debugging)
            
        Returns:
        --------
        Tuple[List[Dict], pd.DataFrame]
            (results list, metrics DataFrame)
        """
        # Get all images
        image_list = get_all_images(self.input_dir, self.categories)
        
        if not image_list:
            print(f"No images found in {self.input_dir}")
            return [], pd.DataFrame()
        
        # Classify by quality if requested
        if classify_quality and self.use_quality_based_metrics:
            high_quality, low_quality = self.classify_images_by_quality()
            # Combine lists for processing
            all_images = high_quality + low_quality
        else:
            all_images = image_list
        
        print(f"\nProcessing {len(all_images)} images through 3 pipelines...")
        print("This may take several minutes depending on your computer.\n")
        
        all_results = []
        
        # Process sequentially (safer for image processing)
        for img_info in tqdm(all_images, desc="Processing images"):
            result = self.process_single_image(img_info)
            if result:
                all_results.append(result)
        
        print(f"\n✓ Successfully processed {len(all_results)} images")
        
        # Save metrics to CSV
        if save_csv and all_results:
            df = self.save_metrics_to_csv(all_results)
            return all_results, df
        
        return all_results, pd.DataFrame()
    
    def save_metrics_to_csv(self, results: List[Dict]) -> pd.DataFrame:
        """
        Convert results to DataFrame and save to CSV.
        
        Parameters:
        -----------
        results : List[Dict]
            List of result dictionaries
            
        Returns:
        --------
        pd.DataFrame
            Metrics DataFrame
        """
        rows = []
        
        for result in results:
            base_row = {
                'filename': result['filename'],
                'category': result['category'],
                'original_quality_score': result.get('quality_score', 0)
            }
            
            for pipeline, metrics in result['metrics'].items():
                row = base_row.copy()
                row['pipeline'] = pipeline
                row.update(metrics)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'metrics', 'all_metrics.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Metrics saved to: {csv_path}")
        
        # Print summary statistics
        self.print_summary(df)
        
        return df
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """Print summary statistics from metrics DataFrame."""
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS (Average across all images)")
        print("=" * 70)
        
        # Group by pipeline
        key_metrics = ['SSIM', 'Contrast_Improvement_Index', 
                       'Edge_Preservation_Index', 'Sharpness_Ratio']
        
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        if available_metrics:
            summary = df.groupby('pipeline')[available_metrics].mean()
            print("\n" + summary.round(4).to_string())
        
        # Quality-based metrics if available
        if 'ssim_improvement' in df.columns:
            print("\n\nQUALITY-BASED SSIM IMPROVEMENT:")
            print("-" * 50)
            quality_summary = df.groupby('pipeline')['ssim_improvement'].mean()
            print(quality_summary.round(4).to_string())
        
        print("\n" + "=" * 70)
    
    def calculate_batch_fid(self, results: List[Dict]) -> Dict[str, float]:
        """
        Calculate FID scores for the batch.
        
        Compares:
        - Original vs Enhanced (for each pipeline)
        - Enhanced vs High-quality references (if available)
        
        Parameters:
        -----------
        results : List[Dict]
            Processing results
            
        Returns:
        --------
        Dict[str, float]
            FID scores
        """
        print("\nCalculating FID scores...")
        
        fid_results = {}
        
        for category in self.categories:
            # Collect images
            originals = []
            enhanced_p1 = []
            enhanced_p2 = []
            enhanced_p3 = []
            
            for result in results:
                if result['category'] == category:
                    # Load original
                    orig_path = os.path.join(self.input_dir, category, result['filename'])
                    if os.path.exists(orig_path):
                        originals.append(load_image(orig_path))
                    
                    # Load enhanced versions
                    for pipeline, storage in [('pipeline1', enhanced_p1),
                                               ('pipeline2', enhanced_p2),
                                               ('pipeline3', enhanced_p3)]:
                        enh_path = os.path.join(self.output_dir, pipeline, 
                                                category, result['filename'])
                        if os.path.exists(enh_path):
                            storage.append(load_image(enh_path))
            
            if len(originals) < 2:
                continue
            
            # Calculate FID for each pipeline
            for pipeline, enhanced in [('pipeline1', enhanced_p1),
                                        ('pipeline2', enhanced_p2),
                                        ('pipeline3', enhanced_p3)]:
                if len(enhanced) >= 2:
                    fid = self.fid_calc.compute_fid_between_sets(originals, enhanced)
                    fid_results[f'{category}_{pipeline}_fid'] = fid
            
            # FID to high quality references
            if category in self.high_quality_images and len(self.high_quality_images[category]) >= 2:
                refs = self.high_quality_images[category]
                
                fid_orig = self.fid_calc.compute_fid_between_sets(originals, refs)
                fid_results[f'{category}_original_vs_refs_fid'] = fid_orig
                
                for pipeline, enhanced in [('pipeline1', enhanced_p1),
                                            ('pipeline2', enhanced_p2),
                                            ('pipeline3', enhanced_p3)]:
                    if len(enhanced) >= 2:
                        fid = self.fid_calc.compute_fid_between_sets(enhanced, refs)
                        fid_results[f'{category}_{pipeline}_vs_refs_fid'] = fid
        
        # Save FID results
        fid_df = pd.DataFrame([fid_results])
        fid_path = os.path.join(self.output_dir, 'metrics', 'fid_scores.csv')
        fid_df.to_csv(fid_path, index=False)
        print(f"✓ FID scores saved to: {fid_path}")
        
        return fid_results


class QuickProcessor:
    """
    Quick processor for testing on a small subset of images.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.p1 = Pipeline1()
        self.p2 = Pipeline2()
        self.p3 = Pipeline3()
        self.metrics = MetricsCalculator()
    
    def process_sample(self, n_samples: int = 5) -> pd.DataFrame:
        """
        Process a small sample from each category.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples per category
            
        Returns:
        --------
        pd.DataFrame
            Metrics for sampled images
        """
        import random
        
        categories = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        all_images = get_all_images(self.input_dir, categories)
        
        # Sample images
        by_category = {cat: [] for cat in categories}
        for img_info in all_images:
            by_category[img_info['category']].append(img_info)
        
        sampled = []
        for cat in categories:
            if by_category[cat]:
                sampled.extend(random.sample(by_category[cat], 
                                            min(n_samples, len(by_category[cat]))))
        
        print(f"Processing {len(sampled)} sample images...")
        
        results = []
        for img_info in tqdm(sampled):
            img = load_image(img_info['path'])
            
            # Process through all pipelines
            p1_result = self.p1.process(img, filter_type='bilateral', contrast_type='clahe')
            p2_result, _ = self.p2.process(p1_result, edge_method='canny')
            p3_result = self.p3.adaptive_process(img)
            
            # Calculate metrics
            for name, result in [('pipeline1', p1_result),
                                  ('pipeline2', p2_result),
                                  ('pipeline3', p3_result)]:
                metrics = self.metrics.calculate_all_metrics(img, result)
                metrics['filename'] = img_info['filename']
                metrics['category'] = img_info['category']
                metrics['pipeline'] = name
                results.append(metrics)
        
        df = pd.DataFrame(results)
        print(f"\n✓ Processed {len(sampled)} images")
        
        return df

