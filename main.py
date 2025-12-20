#!/usr/bin/env python3
"""
OCT Image Enhancement Project - Main Execution Script

This script provides the main entry point for running the OCT image
enhancement pipeline. It processes images through three different
enhancement pipelines and generates comprehensive metrics and visualizations.

Usage:
    python main.py --mode full          # Full processing on all images
    python main.py --mode quick         # Quick test on sample images
    python main.py --mode visualize     # Generate visualizations only
    python main.py --mode demo          # Demo on a single image
"""

import argparse
import os
import sys
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.utils import load_image, get_all_images, estimate_image_quality
from src.pipeline1 import Pipeline1
from src.pipeline2 import Pipeline2
from src.pipeline3 import Pipeline3
from src.metrics import MetricsCalculator, QualityMetrics, SimplifiedFID
from src.batch_processor import BatchProcessor, QuickProcessor
from src.visualization import Visualizer, EdgeVisualization


def demo_single_image(data_dir: str, output_dir: str):
    """
    Demonstrate enhancement on a single image.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing OCT images
    output_dir : str
        Directory to save outputs
    """
    print("\n" + "=" * 60)
    print("OCT IMAGE ENHANCEMENT - SINGLE IMAGE DEMO")
    print("=" * 60)
    
    # Find a sample image
    all_images = get_all_images(data_dir)
    if not all_images:
        print(f"No images found in {data_dir}")
        return
    
    # Select an image from each category for demo
    categories = ['NORMAL', 'DME', 'CNV', 'DRUSEN']
    sample = None
    for cat in categories:
        cat_images = [img for img in all_images if img['category'] == cat]
        if cat_images:
            sample = cat_images[0]
            break
    
    if sample is None:
        sample = all_images[0]
    
    print(f"\nProcessing: {sample['filename']} ({sample['category']})")
    
    # Load image
    img = load_image(sample['path'])
    print(f"Image shape: {img.shape}")
    print(f"Value range: [{img.min()}, {img.max()}]")
    
    # Estimate quality
    quality = estimate_image_quality(img)
    print(f"\nOriginal image quality:")
    print(f"  Sharpness: {quality['sharpness']:.2f}")
    print(f"  Contrast: {quality['contrast']:.2f}")
    print(f"  Quality Score: {quality['quality_score']:.3f}")
    
    # Initialize pipelines
    p1 = Pipeline1()
    p2 = Pipeline2()
    p3 = Pipeline3()
    metrics = MetricsCalculator()
    
    # Process through pipelines
    print("\nProcessing through pipelines...")
    
    # Pipeline 1
    p1_result = p1.process(img, filter_type='bilateral', contrast_type='clahe')
    p1_metrics = metrics.calculate_all_metrics(img, p1_result)
    print(f"\nPipeline 1 (Noise + Contrast):")
    print(f"  SSIM: {p1_metrics['SSIM']:.4f}")
    print(f"  Contrast Improvement: {p1_metrics['Contrast_Improvement_Index']:.3f}x")
    
    # Pipeline 2
    p2_result, edges = p2.process(p1_result, edge_method='canny')
    p2_metrics = metrics.calculate_all_metrics(img, p2_result)
    print(f"\nPipeline 2 (P1 + Edges):")
    print(f"  SSIM: {p2_metrics['SSIM']:.4f}")
    print(f"  Edge Preservation: {p2_metrics['Edge_Preservation_Index']:.4f}")
    
    # Pipeline 3
    p3_result = p3.adaptive_process(img)
    p3_metrics = metrics.calculate_all_metrics(img, p3_result)
    print(f"\nPipeline 3 (Adaptive Comprehensive):")
    print(f"  SSIM: {p3_metrics['SSIM']:.4f}")
    print(f"  Sharpness Improvement: {p3_metrics['Sharpness_Ratio']:.3f}x")
    
    # Save demo outputs
    demo_dir = os.path.join(output_dir, 'demo')
    os.makedirs(demo_dir, exist_ok=True)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(p1_result, cmap='gray')
    axes[0, 1].set_title('Pipeline 1\n(Bilateral + CLAHE)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(p2_result, cmap='gray')
    axes[0, 2].set_title('Pipeline 2\n(P1 + Canny Edges)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(p3_result, cmap='gray')
    axes[1, 0].set_title('Pipeline 3\n(Adaptive)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title('Detected Edges', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Histogram comparison
    axes[1, 2].hist(img.ravel(), bins=50, alpha=0.5, label='Original', color='blue')
    axes[1, 2].hist(p3_result.ravel(), bins=50, alpha=0.5, label='Enhanced', color='green')
    axes[1, 2].set_title('Histogram Comparison', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Pixel Intensity')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.suptitle(f'OCT Enhancement Demo - {sample["category"]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(demo_dir, 'demo_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Demo visualization saved to: {save_path}")
    
    # Show processing stages for Pipeline 3
    stages = p3.process_with_stages(img)
    Visualizer.plot_processing_stages(
        stages, 
        os.path.join(demo_dir, 'processing_stages.png')
    )
    print(f"✓ Processing stages saved to: {demo_dir}/processing_stages.png")
    
    # Edge detection comparison
    EdgeVisualization.compare_edge_methods(
        p1_result,
        os.path.join(demo_dir, 'edge_methods_comparison.png')
    )
    print(f"✓ Edge methods comparison saved to: {demo_dir}/edge_methods_comparison.png")
    
    # Filter comparison
    EdgeVisualization.compare_filter_methods(
        img,
        os.path.join(demo_dir, 'filter_methods_comparison.png')
    )
    print(f"✓ Filter methods comparison saved to: {demo_dir}/filter_methods_comparison.png")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


def quick_test(data_dir: str, output_dir: str, n_samples: int = 5):
    """
    Quick test on a small sample of images.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing OCT images
    output_dir : str
        Directory to save outputs
    n_samples : int
        Number of samples per category
    """
    print("\n" + "=" * 60)
    print("OCT IMAGE ENHANCEMENT - QUICK TEST")
    print("=" * 60)
    
    processor = QuickProcessor(data_dir, output_dir)
    df = processor.process_sample(n_samples=n_samples)
    
    if df.empty:
        print("No results generated")
        return
    
    # Save metrics
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    csv_path = os.path.join(output_dir, 'metrics', 'quick_test_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Metrics saved to: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUICK TEST SUMMARY")
    print("=" * 60)
    
    summary = df.groupby('pipeline')[['SSIM', 'Contrast_Improvement_Index', 
                                       'Edge_Preservation_Index']].mean()
    print("\nAverage Metrics by Pipeline:")
    print(summary.round(4).to_string())
    
    # Generate visualization
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    Visualizer.plot_metrics_comparison(
        df,
        os.path.join(output_dir, 'figures', 'quick_test_metrics.png')
    )
    print(f"\n✓ Visualization saved to: {output_dir}/figures/quick_test_metrics.png")
    
    print("\n" + "=" * 60)
    print("QUICK TEST COMPLETE")
    print("=" * 60)


def full_processing(data_dir: str, output_dir: str, 
                    use_quality_metrics: bool = True):
    """
    Full processing on all images.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing OCT images
    output_dir : str
        Directory to save outputs
    use_quality_metrics : bool
        Whether to use quality-based SSIM comparison
    """
    print("\n" + "=" * 60)
    print("OCT IMAGE ENHANCEMENT - FULL PROCESSING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create batch processor
    processor = BatchProcessor(
        input_dir=data_dir,
        output_dir=output_dir,
        use_quality_based_metrics=use_quality_metrics
    )
    
    # Process all images
    results, metrics_df = processor.process_all(
        classify_quality=use_quality_metrics,
        save_csv=True
    )
    
    if metrics_df.empty:
        print("No results generated")
        return
    
    # Calculate FID scores
    if results:
        print("\nCalculating FID scores...")
        fid_results = processor.calculate_batch_fid(results)
        
        print("\nFID Scores (lower is better):")
        for key, value in fid_results.items():
            if 'improvement' in key.lower():
                print(f"  {key}: {value:+.4f}")
            else:
                print(f"  {key}: {value:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    Visualizer.create_summary_report(metrics_df, output_dir)
    
    # Generate sample comparisons
    Visualizer.generate_sample_comparisons(
        data_dir, output_dir, num_samples=20
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("FULL PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nTotal time: {elapsed_time/60:.1f} minutes")
    print(f"Images processed: {len(results)}")
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - pipeline1/, pipeline2/, pipeline3/: Enhanced images")
    print("  - metrics/: CSV files with metrics")
    print("  - figures/: Visualization plots")
    print("  - comparisons/: Side-by-side comparisons")


def generate_visualizations(data_dir: str, output_dir: str):
    """
    Generate visualizations from existing results.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing OCT images
    output_dir : str
        Directory containing processed outputs
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Load metrics if available
    csv_path = os.path.join(output_dir, 'metrics', 'all_metrics.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(output_dir, 'metrics', 'quick_test_metrics.csv')
    
    if os.path.exists(csv_path):
        print(f"\nLoading metrics from: {csv_path}")
        metrics_df = pd.read_csv(csv_path)
        
        # Generate summary visualizations
        Visualizer.create_summary_report(metrics_df, output_dir)
    else:
        print(f"No metrics file found. Run processing first.")
    
    # Generate sample comparisons
    print("\nGenerating sample comparisons...")
    Visualizer.generate_sample_comparisons(
        data_dir, output_dir, num_samples=20
    )
    
    print("\n✓ Visualization generation complete")


def analyze_results(output_dir: str):
    """
    Analyze and print detailed results.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing processed outputs
    """
    csv_path = os.path.join(output_dir, 'metrics', 'all_metrics.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(output_dir, 'metrics', 'quick_test_metrics.csv')
    
    if not os.path.exists(csv_path):
        print(f"No metrics file found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS ANALYSIS")
    print("=" * 70)
    
    # Overall performance
    print("\n1. OVERALL PERFORMANCE (Average across all images)")
    print("-" * 70)
    
    key_metrics = ['SSIM', 'Contrast_Improvement_Index', 
                   'Edge_Preservation_Index', 'Sharpness_Ratio']
    available = [m for m in key_metrics if m in df.columns]
    
    overall = df.groupby('pipeline')[available].mean()
    print(overall.round(4).to_string())
    
    # Performance by category
    print("\n2. PERFORMANCE BY CATEGORY")
    print("-" * 70)
    
    for category in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
        cat_data = df[df['category'] == category]
        if not cat_data.empty:
            print(f"\n{category}:")
            cat_summary = cat_data.groupby('pipeline')[['SSIM', 'Contrast_Improvement_Index']].mean()
            print(cat_summary.round(4).to_string())
    
    # Best pipeline determination
    print("\n3. BEST PIPELINE DETERMINATION")
    print("-" * 70)
    
    if 'Contrast_Improvement_Index' in overall.columns:
        best_contrast = overall['Contrast_Improvement_Index'].idxmax()
        print(f"Best Contrast Enhancement: {best_contrast}")
    
    if 'SSIM' in overall.columns:
        best_ssim = overall['SSIM'].idxmax()
        print(f"Best Structure Preservation: {best_ssim}")
    
    if 'Edge_Preservation_Index' in overall.columns:
        best_edges = overall['Edge_Preservation_Index'].idxmax()
        print(f"Best Edge Preservation: {best_edges}")
    
    # Quality-based analysis
    if 'ssim_improvement' in df.columns:
        print("\n4. QUALITY-BASED SSIM IMPROVEMENT")
        print("-" * 70)
        quality_summary = df.groupby('pipeline')['ssim_improvement'].agg(['mean', 'std', 'min', 'max'])
        print(quality_summary.round(4).to_string())
    
    # Improvement verification
    print("\n5. IMPROVEMENT VERIFICATION")
    print("-" * 70)
    
    for pipeline in df['pipeline'].unique():
        p_data = df[df['pipeline'] == pipeline]
        
        if 'Contrast_Improvement_Index' in p_data.columns:
            contrast_improved = (p_data['Contrast_Improvement_Index'] > 1.0).sum()
            contrast_pct = (contrast_improved / len(p_data)) * 100
            
            print(f"\n{pipeline}:")
            print(f"  Images with improved contrast: {contrast_pct:.1f}%")
            print(f"  Average SSIM: {p_data['SSIM'].mean():.4f}")
            
            if 'Edge_Preservation_Index' in p_data.columns:
                print(f"  Average Edge Preservation: {p_data['Edge_Preservation_Index'].mean():.4f}")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='OCT Image Enhancement Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode demo                    # Demo on single image
    python main.py --mode quick --samples 5       # Quick test with 5 samples/category
    python main.py --mode full                    # Full processing
    python main.py --mode visualize               # Generate visualizations
    python main.py --mode analyze                 # Analyze existing results
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['demo', 'quick', 'full', 'visualize', 'analyze'],
        default='demo',
        help='Processing mode'
    )
    
    parser.add_argument(
        '--data-dir',
        default=None,
        help='Path to OCT data directory (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of samples per category for quick test (default: 5)'
    )
    
    parser.add_argument(
        '--no-quality-metrics',
        action='store_true',
        help='Disable quality-based SSIM comparison'
    )
    
    args = parser.parse_args()
    
    # Auto-detect data directory
    if args.data_dir is None:
        # Look for common locations
        possible_paths = [
            '../CellData/OCT/test',  # Test set (smaller)
            '../CellData/OCT/train', # Train set (larger)
            'data/subset',           # Custom subset
            '../CellData/OCT',       # Full dataset
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            if os.path.exists(abs_path):
                args.data_dir = abs_path
                break
        
        if args.data_dir is None:
            print("Error: Could not find OCT data directory. Please specify with --data-dir")
            sys.exit(1)
    
    # Make output directory absolute
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
    
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Run appropriate mode
    if args.mode == 'demo':
        demo_single_image(args.data_dir, args.output_dir)
    
    elif args.mode == 'quick':
        quick_test(args.data_dir, args.output_dir, args.samples)
    
    elif args.mode == 'full':
        full_processing(
            args.data_dir, 
            args.output_dir,
            use_quality_metrics=not args.no_quality_metrics
        )
    
    elif args.mode == 'visualize':
        generate_visualizations(args.data_dir, args.output_dir)
    
    elif args.mode == 'analyze':
        analyze_results(args.output_dir)


if __name__ == '__main__':
    main()

